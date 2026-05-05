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


# ---- spacing rules ----
HARD_MIN_MI = 0.40
PREF_MAX_MI = 1.10
HARD_MAX_MI = 1.40

TARGET_MI = 1.00  # only a tiebreaker now (NOT the main objective)

# always add last AGM
ALWAYS_END_AGM = True

# reachability rule (truck within 1/8 mile of target)
REACHABLE_GAP_METERS = 201.168

# parking heuristic: when aiming for a candidate, try slightly downstream first
PARK_OFFSET_MI = 0.02  # ~105 ft

# candidate sampling / cost controls
CANDIDATE_STEP_MI = 0.05
MAX_CANDIDATES_PER_BAND = 21  # set to 11 for faster

# "decent road-ish option" threshold inside 0.40–1.10
# If nothing inside the preferred band gets within this gap, we allow 1.10–1.40.
PREFERRED_BAND_GOOD_GAP_METERS = 400.0


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


def route_to_lonlat(token: str, a_lonlat: Tuple[float, float], b_lonlat: Tuple[float, float]) -> Tuple[float, float, bool]:
    try:
        data = mapbox_directions_driving(token, a_lonlat, b_lonlat)
    except Exception:
        return (float("inf"), float("inf"), False)

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

        return (driving_mi, end_gap, True)
    except Exception:
        return (float("inf"), float("inf"), False)


def make_s_values(start_s: float, end_s: float) -> List[float]:
    vals: List[float] = []
    s = start_s
    while s <= end_s + 1e-9:
        vals.append(round(s, 4))
        s += CANDIDATE_STEP_MI

    if len(vals) > MAX_CANDIDATES_PER_BAND:
        step = max(1, len(vals) // MAX_CANDIDATES_PER_BAND)
        vals = vals[::step]
    return vals


def best_parking_variant_for_candidate(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev_agm: Agm,
    cand_along_mi: float,
) -> Tuple[float, float, str, float, float, bool]:
    """
    For this candidate along-mile, try downstream first, then none, then upstream.
    Return the best (lon,lat, used_offset, driving_mi, end_gap_m, ok).
    """
    total_mi = m_to_mi(cum_m[-1])

    variants = [
        ("downstream", min(total_mi, cand_along_mi + PARK_OFFSET_MI)),
        ("none", cand_along_mi),
        ("upstream", max(0.0, cand_along_mi - PARK_OFFSET_MI)),
    ]

    best = None
    for label, s in variants:
        lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(s))
        driving_mi, end_gap_m, ok = route_to_lonlat(token, (prev_agm.lon, prev_agm.lat), (lon, lat))
        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        # Prefer reachable immediately (downstream is tested first)
        if reachable:
            return (lon, lat, label, driving_mi, end_gap_m, ok)

        cur = (end_gap_m, driving_mi, ok, lon, lat, label)
        if best is None or cur[0] < best[0] or (cur[0] == best[0] and cur[1] < best[1]):
            best = cur

    assert best is not None
    return (best[3], best[4], best[5], best[1], best[0], best[2])


def pick_next_agm(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev: Agm,
    end_along_mi: float,
) -> Optional[Agm]:
    """
    Your requested rule:
    - Prefer road crossings in 0.40–1.10 (i.e., prefer smallest end-gap in that band)
    - Only if that band has no decent road-ish option, allow 1.10–1.40
    """
    hard_min = prev.along_mi + HARD_MIN_MI
    if hard_min > end_along_mi:
        return None

    band1_lo = hard_min
    band1_hi = min(prev.along_mi + PREF_MAX_MI, end_along_mi)

    band2_lo = max(band1_hi, prev.along_mi + PREF_MAX_MI)
    band2_hi = min(prev.along_mi + HARD_MAX_MI, end_along_mi)

    def score_band(lo: float, hi: float) -> List[Tuple]:
        if lo > hi:
            return []
        s_values = make_s_values(lo, hi)
        scored = []
        for s in s_values:
            lon, lat, used_offset, driving_mi, end_gap_m, ok = best_parking_variant_for_candidate(
                token, center_coords, cum_m, prev, s
            )
            step_mi = s - prev.along_mi
            reachable = (end_gap_m <= REACHABLE_GAP_METERS)

            # IMPORTANT scoring: road/access first, then ~1 mile tiebreak
            scored.append(
                (
                    0 if reachable else 1,          # reachable first
                    end_gap_m,                      # then closest-to-road
                    driving_mi,                     # then shortest drive
                    abs(step_mi - TARGET_MI),       # then closest-to-1-mile spacing
                    s, lon, lat, used_offset, ok
                )
            )
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        return scored

    band1 = score_band(band1_lo, band1_hi)
    if band1:
        # if we have a "good enough" road-ish option in preferred band, take it.
        if band1[0][1] <= PREFERRED_BAND_GOOD_GAP_METERS:
            best = band1[0]
            return Agm(idx=prev.idx + 1, along_mi=best[4], lon=best[5], lat=best[6])

        # even if not "good", we still prefer band1 unless band2 is clearly better
        band2 = score_band(band2_lo, band2_hi)
        if not band2:
            best = band1[0]
            return Agm(idx=prev.idx + 1, along_mi=best[4], lon=best[5], lat=best[6])

        # allow band2 only if it significantly improves road access
        # (smaller end-gap wins; if tie, band scoring already handled)
        if band2[0][1] + 1e-6 < band1[0][1]:
            best = band2[0]
            return Agm(idx=prev.idx + 1, along_mi=best[4], lon=best[5], lat=best[6])

        best = band1[0]
        return Agm(idx=prev.idx + 1, along_mi=best[4], lon=best[5], lat=best[6])

    # no band1 candidates (rare), fall back to band2
    band2 = score_band(band2_lo, band2_hi)
    if not band2:
        return None
    best = band2[0]
    return Agm(idx=prev.idx + 1, along_mi=best[4], lon=best[5], lat=best[6])


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

    # generate interior points
    while True:
        prev = agms[-1]
        nxt = pick_next_agm(token, center_coords, cum_m, prev, total_mi)
        if nxt is None:
            break

        # prevent duplicates / non-progress
        if nxt.along_mi <= prev.along_mi + 1e-6:
            break

        # if we're near the end, stop and we'll add the end AGM
        if ALWAYS_END_AGM and (total_mi - nxt.along_mi) < HARD_MIN_MI:
            break

        agms.append(nxt)

        if len(agms) > 20000:
            break

    # force end AGM
    if ALWAYS_END_AGM:
        lonE, latE = interpolate_point(center_coords, cum_m, cum_m[-1])
        if agms[-1].along_mi < total_mi - 1e-6:
            agms.append(Agm(idx=agms[-1].idx + 1, along_mi=total_mi, lon=lonE, lat=latE))
        else:
            agms[-1] = Agm(idx=agms[-1].idx, along_mi=total_mi, lon=lonE, lat=latE)

    # build one output table
    rows: List[Dict[str, Any]] = []
    cum_pipe = 0.0
    cum_drive = 0.0

    for i in range(len(agms) - 1):
        a = agms[i]
        b = agms[i + 1]

        pipe_seg = b.along_mi - a.along_mi
        cum_pipe += pipe_seg

        driving_mi, end_gap_m, ok = route_to_lonlat(token, (a.lon, a.lat), (b.lon, b.lat))
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
        "min_pipeline_spacing_mi": "" if len(agms) < 2 else round(min(agms[i + 1].along_mi - agms[i].along_mi for i in range(len(agms) - 1)), 3),
        "max_pipeline_spacing_mi": "" if len(agms) < 2 else round(max(agms[i + 1].along_mi - agms[i].along_mi for i in range(len(agms) - 1)), 3),
    }

    kmz_out = build_kmz(center_coords, agms)
    csv_out = rows_to_csv_bytes(rows)
    return agms, rows, summary, kmz_out, csv_out


st.set_page_config(page_title="AGM Planner", layout="wide")
st.title("AGM Planner (prefer road crossings in 0.40–1.10)")

token = st.secrets.get("MAPBOX_TOKEN", "")
if not token:
    st.error('Missing Mapbox token in Streamlit Secrets. Add:\n\nMAPBOX_TOKEN = "sk.your_secret_token_here"')
    st.stop()

if "results" not in st.session_state:
    st.session_state["results"] = None

uploaded = st.file_uploader("Upload centerline KMZ (LineString)", type=["kmz"])
run_btn = st.button("Generate AGMs for FULL line")

if uploaded and run_btn:
    try:
        with st.spinner("Parsing KMZ..."):
            center = kmz_to_centerline_coords(uploaded.read())

        with st.spinner("Generating AGMs (prefers road crossings)..."):
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
