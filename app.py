import io
import math
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st

import folium
from streamlit_folium import st_folium
import simplekml


# ---- spacing rules (miles along pipeline) ----
HARD_MIN_MI = 0.40
HARD_MAX_MI = 1.40

# "about a mile" window you care about for normal rhythm
PREF_LO_MI = 0.90
PREF_HI_MI = 1.10
TARGET_MI = 1.00

# end-of-line handling
ALWAYS_END_AGM = True
TAIL_MIN_MI = HARD_MIN_MI

# reachability / truck rule
REACHABLE_GAP_METERS = 201.168  # 1/8 mile

# Mapbox scoring / anti-weird-routing heuristic
MAX_DRIVE_STRAIGHT_RATIO = 2.75

# Candidate sampling (speed)
N_CAND_PREF = 11   # 0.40–1.10 window
N_CAND_WIDE = 9    # 1.10–1.40 window
N_CAND_SHORT = 7   # 0.40–0.90 window

# "acceptable road crossing" proxy (meters): route end must be this close to the target point
# Tune: lower = stricter (more likely to short-step at crossings)
ACCEPTABLE_END_GAP_M = 350.0


@dataclass
class Agm:
    idx: int
    along_mi: float
    lon: float
    lat: float


def mi_to_m(mi: float) -> float:
    return mi * 1609.344


def m_to_mi(m: float) -> float:
    return m / 1609.344


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def agm_label(idx: int) -> str:
    return f"{idx * 10:03d}"


def parse_first_linestring_coords_from_kml(kml_text: str) -> List[Tuple[float, float]]:
    root = ET.fromstring(kml_text)

    def local(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    linestring_coords = None
    for el in root.iter():
        if local(el.tag) == "LineString":
            for child in el.iter():
                if local(child.tag) == "coordinates" and child.text:
                    linestring_coords = child.text.strip()
                    break
        if linestring_coords:
            break

    if not linestring_coords:
        raise ValueError("Could not find a LineString/coordinates in the KML inside your KMZ.")

    coords: List[Tuple[float, float]] = []
    for token in linestring_coords.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) < 2:
            continue
        coords.append((float(parts[0]), float(parts[1])))

    if len(coords) < 2:
        raise ValueError("LineString had fewer than 2 coordinate points.")
    return coords


def kmz_to_centerline_coords(kmz_bytes: bytes) -> List[Tuple[float, float]]:
    with zipfile.ZipFile(io.BytesIO(kmz_bytes), "r") as z:
        kml_name = None
        for name in z.namelist():
            if name.lower().endswith(".kml"):
                kml_name = name
                break
        if not kml_name:
            raise ValueError("KMZ did not contain a .kml file.")
        kml_text = z.read(kml_name).decode("utf-8", errors="replace")
    return parse_first_linestring_coords_from_kml(kml_text)


def cumulative_distances_m(coords: List[Tuple[float, float]]) -> List[float]:
    d = [0.0]
    total = 0.0
    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        total += haversine_m(lon1, lat1, lon2, lat2)
        d.append(total)
    return d


def interpolate_point(coords: List[Tuple[float, float]], cum_m: List[float], target_m: float) -> Tuple[float, float]:
    if target_m <= 0:
        return coords[0]
    if target_m >= cum_m[-1]:
        return coords[-1]

    lo, hi = 0, len(cum_m) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum_m[mid] < target_m:
            lo = mid + 1
        else:
            hi = mid
    i = max(1, lo)

    m0 = cum_m[i - 1]
    m1 = cum_m[i]
    lon0, lat0 = coords[i - 1]
    lon1, lat1 = coords[i]

    if m1 == m0:
        return (lon1, lat1)

    t = (target_m - m0) / (m1 - m0)
    lon = lon0 + (lon1 - lon0) * t
    lat = lat0 + (lat1 - lat0) * t
    return (lon, lat)


@st.cache_data(show_spinner=False)
def mapbox_directions_driving(token: str, a: Tuple[float, float], b: Tuple[float, float]) -> Dict[str, Any]:
    (alon, alat) = a
    (blon, blat) = b
    url = (
        "https://api.mapbox.com/directions/v5/mapbox/driving/"
        f"{alon},{alat};{blon},{blat}"
        f"?geometries=geojson&overview=full&access_token={token}"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def route_metrics(token: str, a_lonlat: Tuple[float, float], b_lonlat: Tuple[float, float]):
    straight_m = haversine_m(a_lonlat[0], a_lonlat[1], b_lonlat[0], b_lonlat[1])
    straight_mi = straight_m / 1609.344

    try:
        data = mapbox_directions_driving(token, a_lonlat, b_lonlat)
    except Exception:
        return (float("inf"), float("inf"), False, straight_mi)

    try:
        route = data["routes"][0]
        dist_m = float(route.get("distance", float("inf")))
        driving_mi = dist_m / 1609.344

        coords = route.get("geometry", {}).get("coordinates", [])
        if coords:
            end_lon, end_lat = coords[-1]
            end_gap = haversine_m(end_lon, end_lat, b_lonlat[0], b_lonlat[1])
        else:
            end_gap = float("inf")

        return (driving_mi, end_gap, True, straight_mi)
    except Exception:
        return (float("inf"), float("inf"), False, straight_mi)


def linspace_clamped(lo: float, hi: float, n: int) -> List[float]:
    if n <= 1:
        return [round((lo + hi) / 2, 4)]
    step = (hi - lo) / (n - 1)
    return [round(lo + i * step, 4) for i in range(n)]


def score_tuple(
    reachable: bool,
    end_gap_m: float,
    drive_straight_ratio: float,
    abs_from_target: float,
    driving_mi: float,
) -> Tuple:
    ratio_penalty = 0.0
    if math.isfinite(drive_straight_ratio) and drive_straight_ratio > MAX_DRIVE_STRAIGHT_RATIO:
        ratio_penalty = (drive_straight_ratio - MAX_DRIVE_STRAIGHT_RATIO) * 5000.0

    return (
        0 if reachable else 1,
        end_gap_m + ratio_penalty,
        abs_from_target,
        driving_mi,
    )


def best_in_band(token: str, prev: Agm, center_coords: List[Tuple[float, float]], cum_m: List[float], lo: float, hi: float, n: int) -> Optional[Tuple]:
    if lo > hi:
        return None

    s_list = linspace_clamped(lo, hi, n)
    best = None

    for s in s_list:
        lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(s))
        driving_mi, end_gap_m, ok, straight_mi = route_metrics(token, (prev.lon, prev.lat), (lon, lat))

        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        if ok and math.isfinite(driving_mi) and straight_mi > 1e-6:
            ratio = driving_mi / straight_mi
        else:
            ratio = float("inf")

        step_mi = s - prev.along_mi
        abs_from_target = abs(step_mi - TARGET_MI)

        stuple = score_tuple(reachable, end_gap_m, ratio, abs_from_target, driving_mi)
        cand = (stuple, s, lon, lat, end_gap_m, reachable, step_mi)

        if best is None or cand[0] < best[0]:
            best = cand

    return best


def acceptable_crossing(best: Optional[Tuple]) -> bool:
    if best is None:
        return False
    end_gap_m = best[4]
    return math.isfinite(end_gap_m) and end_gap_m <= ACCEPTABLE_END_GAP_M


def pick_next_agm(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev: Agm,
    end_along_mi: float,
) -> Optional[Agm]:
    prev_along = prev.along_mi
    if prev_along >= end_along_mi - 1e-6:
        return None

    # 0.40–0.90 (short crossings)
    band_short_lo = prev_along + HARD_MIN_MI
    band_short_hi = min(prev_along + PREF_LO_MI, end_along_mi)

    # 0.40–1.10 (main band: includes short + up to 1.10)
    band_main_lo = prev_along + HARD_MIN_MI
    band_main_hi = min(prev_along + PREF_HI_MI, end_along_mi)

    # 1.10–1.40 (wide band)
    band_wide_lo = max(prev_along + PREF_HI_MI, prev_along + HARD_MIN_MI)
    band_wide_hi = min(prev_along + HARD_MAX_MI, end_along_mi)

    best_main = best_in_band(token, prev, center_coords, cum_m, band_main_lo, band_main_hi, N_CAND_PREF)
    best_short = best_in_band(token, prev, center_coords, cum_m, band_short_lo, band_short_hi, N_CAND_SHORT)
    best_wide = best_in_band(token, prev, center_coords, cum_m, band_wide_lo, band_wide_hi, N_CAND_WIDE)

    # 1) If we have an acceptable crossing <= 1.10, take it.
    if acceptable_crossing(best_main):
        _, s, lon, lat, _, _, _ = best_main
        return Agm(idx=prev.idx + 1, along_mi=s, lon=lon, lat=lat)

    # 2) If wide would push >1.10, but we have acceptable crossings in 0.40–0.90, take short.
    if best_wide is not None and best_wide[6] > 1.10 and acceptable_crossing(best_short):
        _, s, lon, lat, _, _, _ = best_short
        return Agm(idx=prev.idx + 1, along_mi=s, lon=lon, lat=lat)

    # 3) Otherwise use best main if present (even if not "acceptable crossing")
    if best_main is not None:
        _, s, lon, lat, _, _, _ = best_main
        return Agm(idx=prev.idx + 1, along_mi=s, lon=lon, lat=lat)

    # 4) Finally allow wide
    if best_wide is not None:
        _, s, lon, lat, _, _, _ = best_wide
        return Agm(idx=prev.idx + 1, along_mi=s, lon=lon, lat=lat)

    return None


def build_kmz(center_coords: List[Tuple[float, float]], agms: List[Agm]) -> bytes:
    kml = simplekml.Kml()
    f_agms = kml.newfolder(name="AGMs")
    f_center = kml.newfolder(name="Centerline")

    ls = f_center.newlinestring(name="Centerline")
    ls.coords = [(lon, lat) for (lon, lat) in center_coords]
    ls.style.linestyle.color = simplekml.Color.red
    ls.style.linestyle.width = 3.0

    for a in agms:
        label = agm_label(a.idx)
        p = f_agms.newpoint(name=f"AGM {label}", coords=[(a.lon, a.lat)])
        p.description = f"AGM {label} at {a.along_mi:.2f} miles along line"

    kml_bytes = kml.kml().encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml_bytes)
    buf.seek(0)
    return buf.read()


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    cols = [
        "Starting AGM",
        "Next AGM",
        "Segment Pipeline Distance",
        "Cumulative Pipeline Distance",
        "Segment Driving Distance",
        "Cumulative Driving Distance",
        "Reachable by truck",
        "ATV?",
    ]
    out = io.StringIO()
    out.write(",".join(cols) + "\n")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            s = "" if v is None else str(v)
            if "," in s or '"' in s or "\n" in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        out.write(",".join(vals) + "\n")
    return out.getvalue().encode("utf-8")


def generate_full_line(token: str, center_coords: List[Tuple[float, float]]):
    cum_m = cumulative_distances_m(center_coords)
    total_mi = m_to_mi(cum_m[-1])

    lon0, lat0 = interpolate_point(center_coords, cum_m, 0.0)
    agms: List[Agm] = [Agm(idx=0, along_mi=0.0, lon=lon0, lat=lat0)]

    while True:
        prev = agms[-1]
        remaining = total_mi - prev.along_mi

        if remaining <= 1e-6:
            break

        if ALWAYS_END_AGM and remaining < TAIL_MIN_MI:
            break

        nxt = pick_next_agm(token, center_coords, cum_m, prev, total_mi)
        if nxt is None:
            break

        if nxt.along_mi <= prev.along_mi + 1e-6:
            break

        agms.append(nxt)

        if len(agms) > 20000:
            break

    if ALWAYS_END_AGM:
        lonE, latE = interpolate_point(center_coords, cum_m, cum_m[-1])
        if agms[-1].along_mi < total_mi - 1e-6:
            agms.append(Agm(idx=agms[-1].idx + 1, along_mi=total_mi, lon=lonE, lat=latE))
        else:
            agms[-1] = Agm(idx=agms[-1].idx, along_mi=total_mi, lon=lonE, lat=latE)

    rows: List[Dict[str, Any]] = []
    cum_pipe = 0.0
    cum_drive = 0.0

    for i in range(len(agms) - 1):
        a = agms[i]
        b = agms[i + 1]

        pipe_seg = b.along_mi - a.along_mi
        cum_pipe += pipe_seg

        driving_mi, end_gap_m, ok, _ = route_metrics(token, (a.lon, a.lat), (b.lon, b.lat))
        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        if math.isfinite(driving_mi):
            cum_drive += driving_mi

        rows.append(
            {
                "Starting AGM": agm_label(a.idx),
                "Next AGM": agm_label(b.idx),
                "Segment Pipeline Distance": round(pipe_seg, 3),
                "Cumulative Pipeline Distance": round(cum_pipe, 3),
                "Segment Driving Distance": "" if not math.isfinite(driving_mi) else round(driving_mi, 2),
                "Cumulative Driving Distance": "" if not math.isfinite(driving_mi) else round(cum_drive, 2),
                "Reachable by truck": "Yes" if reachable else "",
                "ATV?": "Yes" if (not reachable) else "",
            }
        )

    summary = {
        "total_line_mi": round(total_mi, 3),
        "agm_count": len(agms),
        "hop_count": max(0, len(agms) - 1),
        "min_pipeline_spacing_mi": "" if len(agms) < 2 else round(
            min(agms[i + 1].along_mi - agms[i].along_mi for i in range(len(agms) - 1)), 3
        ),
        "max_pipeline_spacing_mi": "" if len(agms) < 2 else round(
            max(agms[i + 1].along_mi - agms[i].along_mi for i in range(len(agms) - 1)), 3
        ),
    }

    kmz_out = build_kmz(center_coords, agms)
    csv_out = rows_to_csv_bytes(rows)
    return agms, rows, summary, kmz_out, csv_out


st.set_page_config(page_title="AGM Planner", layout="wide")
st.title("AGM Planner (prefer crossings in 0.40–1.10; short-step 0.40–0.90 when needed)")

token = st.secrets.get("MAPBOX_TOKEN", "")
if not token:
    st.error('Missing Mapbox token in Streamlit Secrets. Add:\n\nMAPBOX_TOKEN = "sk.your_secret_token_here"')
    st.stop()

if "results" not in st.session_state:
    st.session_state["results"] = None

if isinstance(st.session_state.get("results"), dict):
    r = st.session_state["results"]
    if not {"agms", "rows", "kmz_bytes", "csv_bytes"}.issubset(r.keys()):
        st.session_state["results"] = None

uploaded = st.file_uploader("Upload centerline KMZ (LineString)", type=["kmz"])

c1, c2 = st.columns([1, 1])
with c1:
    run_btn = st.button("Generate AGMs for FULL line")
with c2:
    if st.button("Clear results"):
        st.session_state["results"] = None

if uploaded and run_btn:
    try:
        with st.spinner("Parsing KMZ..."):
            center = kmz_to_centerline_coords(uploaded.read())

        with st.spinner("Generating AGMs..."):
            agms, rows, summary, kmz_out, csv_out = generate_full_line(token, center)

        st.session_state["results"] = {
            "center": center,
            "agms": agms,
            "rows": rows,
            "summary": summary,
            "kmz_bytes": kmz_out,
            "csv_bytes": csv_out,
        }
    except Exception as e:
        st.session_state["results"] = None
        st.error("Something went wrong.")
        st.exception(e)

results = st.session_state["results"]
if results is None:
    st.info("Upload a KMZ, then click Generate.")
else:
    center = results["center"]
    agms = results["agms"]
    rows = results["rows"]
    summary = results["summary"]

    st.subheader("Summary")
    st.json(summary)

    st.subheader("Output table")
    st.dataframe(rows, width="stretch")

    st.subheader("Map")
    mid = agms[len(agms) // 2]
    m = folium.Map(location=[mid.lat, mid.lon], zoom_start=11, tiles="OpenStreetMap")
    folium.PolyLine([(lat, lon) for (lon, lat) in center], color="#FF0000", weight=3, opacity=0.9).add_to(m)

    for a in agms:
        folium.CircleMarker(
            location=[a.lat, a.lon],
            radius=5,
            color="#000000",
            weight=1,
            fill=True,
            fill_color="#FFEA00",
            fill_opacity=0.95,
            tooltip=f"AGM {agm_label(a.idx)}",
        ).add_to(m)

    st_folium(m, width="stretch", height=600)

    st.subheader("Downloads")
    st.download_button(
        label="Download KMZ (AGMs + Centerline)",
        data=results["kmz_bytes"],
        file_name="agm_output.kmz",
        mime="application/vnd.google-earth.kmz",
    )
    st.download_button(
        label="Download CSV table",
        data=results["csv_bytes"],
        file_name="agm_output.csv",
        mime="text/csv",
    )
