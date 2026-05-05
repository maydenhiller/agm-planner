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


# ---- Your spacing rules ----
MIN_STEP_MI = 0.40
MAX_STEP_MI = 1.40
TARGET_STEP_MI = 1.00  # strongly prefer ~1 mile

# Reachability rule
REACHABLE_GAP_METERS = 201.168  # 1/8 mile

# Parking/side heuristic: try placing AGM slightly downstream/upstream along centerline
PARK_OFFSET_MI = 0.02  # ~105 feet; adjust later if you want (0.01-0.03 typical)


@dataclass
class Agm:
    idx: int  # 0,1,2,... used for label 000,010,...
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
    used_offset: str  # "downstream" | "none" | "upstream"


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


def agm_label(idx: int) -> str:
    # AGM 1 -> 000, AGM 2 -> 010, ...
    return f"{idx * 10:03d}"


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


def route_hop(token: str, a: Agm, b_lonlat: Tuple[float, float]) -> Tuple[float, float, bool]:
    """
    Returns (driving_miles, end_gap_meters, mapbox_ok)
    """
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


def choose_next_along(prev_along: float, end_along: float) -> Optional[float]:
    """
    Place next AGM purely by pipeline distance, preferring TARGET_STEP_MI,
    but constrained to [MIN_STEP_MI, MAX_STEP_MI] and line end.
    """
    lo = prev_along + MIN_STEP_MI
    hi = min(prev_along + MAX_STEP_MI, end_along)
    if lo > end_along:
        return None

    target = prev_along + TARGET_STEP_MI

    # clamp target into [lo, hi]
    if target < lo:
        return lo
    if target > hi:
        return hi
    return target


def generate_agms_full_line_by_distance(
    center_coords: List[Tuple[float, float]],
) -> Tuple[List[Agm], List[float], float]:
    """
    Returns (agms, cum_m, total_mi) without Mapbox calls.
    """
    cum_m = cumulative_distances_m(center_coords)
    total_mi = m_to_mi(cum_m[-1])
    end_along = total_mi

    # AGM 000 at start
    lon0, lat0 = interpolate_point(center_coords, cum_m, 0.0)
    agms: List[Agm] = [Agm(idx=0, along_mi=0.0, lon=lon0, lat=lat0)]

    while True:
        prev = agms[-1]
        nxt_along = choose_next_along(prev.along_mi, end_along)
        if nxt_along is None:
            break

        # stop if we can't fit another minimum after this (prevents tiny tail points)
        if end_along - nxt_along < MIN_STEP_MI:
            # if we're close to the end, we stop instead of adding a short last segment
            break

        lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(nxt_along))
        agms.append(Agm(idx=prev.idx + 1, along_mi=nxt_along, lon=lon, lat=lat))

    return agms, cum_m, total_mi


def adjust_next_point_for_parking(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    prev_agm: Agm,
    next_along_mi: float,
) -> Tuple[Tuple[float, float], str, float, float, bool]:
    """
    Try routing to:
    1) downstream offset (preferred)
    2) no offset
    3) upstream offset
    Pick first reachable; otherwise pick smallest end-gap.

    Returns (lonlat, used_offset, driving_mi, end_gap_m, mapbox_ok)
    """
    candidates: List[Tuple[str, float]] = [
        ("downstream", next_along_mi + PARK_OFFSET_MI),
        ("none", next_along_mi),
        ("upstream", next_along_mi - PARK_OFFSET_MI),
    ]

    best = None

    for label, s in candidates:
        s_clamped = max(0.0, min(s, m_to_mi(cum_m[-1])))
        lon, lat = interpolate_point(center_coords, cum_m, mi_to_m(s_clamped))
        driving_mi, end_gap_m, ok = route_hop(token, prev_agm, (lon, lat))
        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        result = (reachable, end_gap_m, driving_mi, ok, (lon, lat), label)

        # prefer reachable immediately (and downstream is tested first)
        if reachable:
            return (result[4], result[5], result[2], result[1], result[3])

        # keep best non-reachable by smallest end gap, then driving distance
        if best is None:
            best = result
        else:
            if result[1] < best[1] or (result[1] == best[1] and result[2] < best[2]):
                best = result

    # fallback
    assert best is not None
    return (best[4], best[5], best[2], best[1], best[3])


def compute_hops_and_table(
    token: str,
    center_coords: List[Tuple[float, float]],
    cum_m: List[float],
    agms: List[Agm],
) -> Tuple[List[Hop], List[Dict[str, Any]], Dict[str, Any], List[Agm]]:
    """
    Computes hop metrics and also returns a single table (list of dict rows).
    Also returns adjusted AGM points (for KMZ pins) for parking heuristic.
    """
    adjusted_agms = [Agm(idx=a.idx, along_mi=a.along_mi, lon=a.lon, lat=a.lat) for a in agms]

    hops: List[Hop] = []
    rows: List[Dict[str, Any]] = []

    cum_pipeline = 0.0
    cum_driving = 0.0
    used_overpreferred = 0  # kept for future; with this approach it should be 0 most of the time

    for i in range(len(agms) - 1):
        a = adjusted_agms[i]
        b_raw = agms[i + 1]  # along distance is authoritative

        # adjust next point (pin location) for "parking side", favor downstream
        b_lonlat, used_offset, driving_mi, end_gap_m, ok = adjust_next_point_for_parking(
            token, center_coords, cum_m, a, b_raw.along_mi
        )
        adjusted_agms[i + 1] = Agm(idx=b_raw.idx, along_mi=b_raw.along_mi, lon=b_lonlat[0], lat=b_lonlat[1])

        pipeline_seg = b_raw.along_mi - agms[i].along_mi
        cum_pipeline += pipeline_seg

        if math.isfinite(driving_mi):
            cum_driving += driving_mi

        reachable = (end_gap_m <= REACHABLE_GAP_METERS)

        hop = Hop(
            from_idx=agms[i].idx,
            to_idx=b_raw.idx,
            from_along_mi=agms[i].along_mi,
            to_along_mi=b_raw.along_mi,
            pipeline_segment_mi=pipeline_seg,
            driving_mi=driving_mi,
            end_gap_m=end_gap_m,
            reachable=reachable,
            mapbox_ok=ok,
            used_offset=used_offset,
        )
        hops.append(hop)

        rows.append(
            {
                "Starting AGM": agm_label(agms[i].idx),
                "Next AGM": agm_label(b_raw.idx),
                "Segment Pipeline Distance": round(pipeline_seg, 3),
                "Cumulative Pipeline Distance": round(cum_pipeline, 3),
                "Segment Driving Distance": "" if not math.isfinite(driving_mi) else round(driving_mi, 2),
                "Cumulative Driving Distance": "" if not math.isfinite(driving_mi) else round(cum_driving, 2),
                "Reachable by truck": "Yes" if reachable else "",
                "ATV?": "Yes" if (not reachable) else "",
            }
        )

    total_driving = sum(h.driving_mi for h in hops if math.isfinite(h.driving_mi))
    four_wheeler_hops = [h for h in hops if not h.reachable]

    summary = {
        "total_line_mi": round(m_to_mi(cum_m[-1]), 3),
        "agm_count": len(agms),
        "hop_count": len(hops),
        "total_driving_mi": round(total_driving, 2),
        "four_wheeler_hop_count": len(four_wheeler_hops),
    }

    return hops, rows, summary, adjusted_agms


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
            # basic CSV escaping
            if "," in s or '"' in s or "\n" in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        out.write(",".join(vals) + "\n")
    return out.getvalue().encode("utf-8")


st.set_page_config(page_title="AGM Planner", layout="wide")
st.title("AGM Planner (full line, ~1 mile spacing, KMZ + table)")

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
            kmz_bytes = uploaded.read()
            center = kmz_to_centerline_coords(kmz_bytes)

        with st.spinner("Placing AGMs by pipeline distance (~1 mile)..."):
            agms, cum_m, total_mi = generate_agms_full_line_by_distance(center)

        with st.spinner("Computing driving hops (Mapbox)..."):
            hops, rows, summary, adjusted_agms = compute_hops_and_table(token, center, cum_m, agms)

        kmz_out = build_kmz(center, adjusted_agms)
        csv_out = rows_to_csv_bytes(rows)

        st.session_state["results"] = {
            "center": center,
            "agms_raw": agms,
            "agms_adjusted": adjusted_agms,
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
    agms_adj = results["agms_adjusted"]
    rows = results["rows"]
    summary = results["summary"]

    st.subheader("Summary")
    st.json(summary)

    st.subheader("Output table")
    st.dataframe(rows, width="stretch")

    st.subheader("Map")
    mid = agms_adj[len(agms_adj) // 2]
    m = folium.Map(location=[mid.lat, mid.lon], zoom_start=11, tiles="OpenStreetMap")

    folium.PolyLine([(lat, lon) for (lon, lat) in center], color="#FF0000", weight=3, opacity=0.9).add_to(m)

    for a in agms_adj:
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
