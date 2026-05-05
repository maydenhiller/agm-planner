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
HARD_MAX_MI = 1.40
PREF_MIN_MI = 0.90
PREF_MAX_MI = 1.10
TARGET_MI = 1.00

# ensure last AGM always added
ALWAYS_END_AGM = True

# reachability rule
REACHABLE_GAP_METERS = 201.168  # 1/8 mile

# parking heuristic
PARK_OFFSET_MI = 0.02  # downstream favored first

# candidate sampling / cost controls
CANDIDATE_STEP_MI = 0.05
MAX_CANDIDATES_PER_BAND = 21  # keep this modest for speed


@dataclass
class Agm:
    idx: int  # 0,1,2,... => 000,010,020...
    along_mi: float
    lon: float
    lat: float


@dataclass
class Hop:
    from_idx: int
    to_idx: int
    from_along_mi: float
    to_along_mi: float
    pipeline_segment_mi: float
    driving_mi: float
    end_gap_m: float
    reachable: bool
    mapbox_ok: bool
    used_offset: str  # downstream|none|upstream


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
        lon = float(parts[0])
        lat = float(parts[1])
        coords.append((lon, lat))

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


def route_to_lonlat(token: str, a: Agm, b_lonlat: Tuple[float, float]) -> Tuple[float, float, bool]:
    try:
        data = mapbox_directions_driving(token, (a.lon, a.lat), b_lonlat)
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


def pick_next_agm_along(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev: Agm,
    end_along_mi: float,
) -> Optional[Agm]:
    """
    Pick next AGM by scoring candidates near TARGET_MI.
    Primary preference: reachable (end gap <= 1/8 mile),
    then closeness to 1.00 mile spacing,
    then smaller driving distance, then smaller end gap.
    """
    hard_min = prev.along_mi + HARD_MIN_MI
    hard_max = min(prev.along_mi + HARD_MAX_MI, end_along_mi)
    if hard_min > end_along_mi:
        return None

    pref_min = max(prev.along_mi + PREF_MIN_MI, hard_min)
    pref_max = min(prev.along_mi + PREF_MAX_MI, hard_max)

    bands: List[Tuple[str, float, float]] = []
    if pref_min <= pref_max:
        bands.append(("preferred", pref_min, pref_max))
    bands.append(("hard", hard_min, hard_max))

    best_cand = None

    for band_name, lo, hi in bands:
        s_values = make_s_values(lo, hi)
        scored = []

        for s in s_values:
            lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(s))
            driving_mi, end_gap_m, ok = route_to_lonlat(token, prev, (lon, lat))
            reachable = (end_gap_m <= REACHABLE_GAP_METERS)
            step_mi = s - prev.along_mi
            scored.append((reachable, abs(step_mi - TARGET_MI), driving_mi, end_gap_m, ok, s, lon, lat))

        # If in preferred band and any reachable exists, only consider reachable
        if band_name == "preferred" and any(x[0] for x in scored):
            scored = [x for x in scored if x[0]]

        scored.sort(key=lambda x: (0 if x[0] else 1, x[1], x[2], x[3]))

        if scored:
            r = scored[0]
            best_cand = Agm(idx=prev.idx + 1, along_mi=r[5], lon=r[6], lat=r[7])
            # If we got a preferred-band candidate, return immediately.
            if band_name == "preferred":
                return best_cand

            # For hard band, only use it if preferred band didn't exist OR had no options.
            return best_cand

    return best_cand


def adjust_for_parking(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev: Agm,
    next_along_mi: float,
) -> Tuple[Tuple[float, float], str, float, float, bool]:
    """
    Try downstream first, then none, then upstream.
    Returns chosen lonlat + hop metrics.
    """
    total_mi = m_to_mi(cum_m[-1])

    candidates = [
        ("downstream", min(total_mi, next_along_mi + PARK_OFFSET_MI)),
        ("none", next_along_mi),
        ("upstream", max(0.0, next_along_mi - PARK_OFFSET_MI)),
    ]

    best = None
    for label, s in candidates:
        lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(s))
        driving_mi, end_gap_m, ok = route_to_lonlat(token, prev, (lon, lat))
        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        if reachable:
            return ((lon, lat), label, driving_mi, end_gap_m, ok)

        cur = (end_gap_m, driving_mi, ok, (lon, lat), label)
        if best is None or cur[0] < best[0] or (cur[0] == best[0] and cur[1] < best[1]):
            best = cur

    assert best is not None
    return (best[3], best[4], best[1], best[0], best[2])


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


def generate_full_line(
    token: str,
    center_coords: List[Tuple[float, float]],
) -> Tuple[List[Agm], List[Hop], List[Dict[str, Any]], Dict[str, Any], bytes, bytes]:
    cum_m = cumulative_distances_m(center_coords)
    total_mi = m_to_mi(cum_m[-1])

    # Start AGM 000 at start of line
    lon0, lat0 = interpolate_point(center_coords, cum_m, 0.0)
    agms_raw: List[Agm] = [Agm(idx=0, along_mi=0.0, lon=lon0, lat=lat0)]

    # Place interior AGMs
    while True:
        prev = agms_raw[-1]
        nxt = pick_next_agm_along(token, center_coords, cum_m, prev, total_mi)
        if nxt is None:
            break

        # Stop if we'd be forced into a tiny tail and we plan to always place end AGM
        if ALWAYS_END_AGM and (total_mi - nxt.along_mi) < PREF_MIN_MI:
            break

        agms_raw.append(nxt)

        # safety to avoid infinite loops
        if len(agms_raw) > 20000:
            break

    # Always add end AGM
    if ALWAYS_END_AGM:
        lonE, latE = interpolate_point(center_coords, cum_m, cum_m[-1])
        if agms_raw[-1].along_mi < total_mi:
            agms_raw.append(Agm(idx=agms_raw[-1].idx + 1, along_mi=total_mi, lon=lonE, lat=latE))
        else:
            agms_raw[-1] = Agm(idx=agms_raw[-1].idx, along_mi=total_mi, lon=lonE, lat=latE)

    # Now compute hops + adjust pins for parking
    agms_adj = [Agm(idx=a.idx, along_mi=a.along_mi, lon=a.lon, lat=a.lat) for a in agms_raw]
    hops: List[Hop] = []
    rows: List[Dict[str, Any]] = []

    cum_pipeline = 0.0
    cum_driving = 0.0

    for i in range(len(agms_raw) - 1):
        a = agms_adj[i]
        b = agms_raw[i + 1]  # authoritative along_mi

        (lonlat, used_offset, driving_mi, end_gap_m, ok) = adjust_for_parking(
            token, center_coords, cum_m, a, b.along_mi
        )
        agms_adj[i + 1] = Agm(idx=b.idx, along_mi=b.along_mi, lon=lonlat[0], lat=lonlat[1])

        pipe_seg = b.along_mi - agms_raw[i].along_mi
        cum_pipeline += pipe_seg

        if math.isfinite(driving_mi):
            cum_driving += driving_mi

        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        hops.append(
            Hop(
                from_idx=agms_raw[i].idx,
                to_idx=b.idx,
                from_along_mi=agms_raw[i].along_mi,
                to_along_mi=b.along_mi,
                pipeline_segment_mi=pipe_seg,
                driving_mi=driving_mi,
                end_gap_m=end_gap_m,
                reachable=reachable,
                mapbox_ok=ok,
                used_offset=used_offset,
            )
        )

        rows.append(
            {
                "Starting AGM": agm_label(agms_raw[i].idx),
                "Next AGM": agm_label(b.idx),
                "Segment Pipeline Distance": round(pipe_seg, 3),
                "Cumulative Pipeline Distance": round(cum_pipeline, 3),
                "Segment Driving Distance": "" if not math.isfinite(driving_mi) else round(driving_mi, 2),
                "Cumulative Driving Distance": "" if not math.isfinite(driving_mi) else round(cum_driving, 2),
                "Reachable by truck": "Yes" if reachable else "",
                "ATV?": "Yes" if (not reachable) else "",
            }
        )

    summary = {
        "total_line_mi": round(total_mi, 3),
        "agm_count": len(agms_adj),
        "hop_count": len(hops),
        "avg_pipeline_spacing_mi": "" if len(hops) == 0 else round(sum(h.pipeline_segment_mi for h in hops) / len(hops), 3),
        "min_pipeline_spacing_mi": "" if len(hops) == 0 else round(min(h.pipeline_segment_mi for h in hops), 3),
        "max_pipeline_spacing_mi": "" if len(hops) == 0 else round(max(h.pipeline_segment_mi for h in hops), 3),
    }

    kmz_out = build_kmz(center_coords, agms_adj)
    csv_out = rows_to_csv_bytes(rows)
    return agms_adj, hops, rows, summary, kmz_out, csv_out


st.set_page_config(page_title="AGM Planner", layout="wide")
st.title("AGM Planner (full line, road-aware, end AGM, KMZ + table)")

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

        with st.spinner("Generating AGMs (prefers ~1.00 mile spacing)..."):
            agms_adj, hops, rows, summary, kmz_out, csv_out = generate_full_line(token, center)

        st.session_state["results"] = {
            "center": center,
            "agms": agms_adj,
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
    st.info("Upload a KMZ, then click the button to process the whole line.")
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
        label = agm_label(a.idx)
        folium.CircleMarker(
            location=[a.lat, a.lon],
            radius=5,
            color="#000000",
            weight=1,
            fill=True,
            fill_color="#FFEA00",
            fill_opacity=0.95,
            tooltip=f"AGM {label}",
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
