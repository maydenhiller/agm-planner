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


MIN_STEP_MI = 0.40
PREF_MAX_STEP_MI = 1.10
MAX_STEP_MI = 1.40

REACHABLE_GAP_METERS = 201.168  # 1/8 mile in meters

CANDIDATE_STEP_MI = 0.05
MAX_CANDIDATES_PER_BAND = 40


@dataclass
class Agm:
    idx: int         # 0,1,2,... used for labelling 000,010,020,...
    along_mi: float
    lon: float
    lat: float


@dataclass
class Hop:
    from_idx: int
    to_idx: int
    from_along_mi: float
    to_along_mi: float
    driving_mi: float
    end_gap_m: float
    reachable: bool
    mapbox_ok: bool


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


def hop_between(token: str, a: Agm, b: Agm) -> Hop:
    try:
        data = mapbox_directions_driving(token, (a.lon, a.lat), (b.lon, b.lat))
    except Exception:
        return Hop(
            from_idx=a.idx,
            to_idx=b.idx,
            from_along_mi=a.along_mi,
            to_along_mi=b.along_mi,
            driving_mi=float("inf"),
            end_gap_m=float("inf"),
            reachable=False,
            mapbox_ok=False,
        )

    try:
        route = data["routes"][0]
        dist_m = float(route.get("distance", float("inf")))
        driving_mi = dist_m / 1609.344

        coords = route.get("geometry", {}).get("coordinates", [])
        if coords:
            end_lon, end_lat = coords[-1]
            end_gap = haversine_m(end_lon, end_lat, b.lon, b.lat)
        else:
            end_gap = float("inf")

        reachable = end_gap <= REACHABLE_GAP_METERS

        return Hop(
            from_idx=a.idx,
            to_idx=b.idx,
            from_along_mi=a.along_mi,
            to_along_mi=b.along_mi,
            driving_mi=driving_mi,
            end_gap_m=end_gap,
            reachable=reachable,
            mapbox_ok=True,
        )
    except Exception:
        return Hop(
            from_idx=a.idx,
            to_idx=b.idx,
            from_along_mi=a.along_mi,
            to_along_mi=b.along_mi,
            driving_mi=float("inf"),
            end_gap_m=float("inf"),
            reachable=False,
            mapbox_ok=False,
        )


def make_candidate_s_values(start_mi: float, end_mi: float) -> List[float]:
    vals: List[float] = []
    s = start_mi
    while s <= end_mi + 1e-9:
        vals.append(round(s, 4))
        s += CANDIDATE_STEP_MI

    if len(vals) > MAX_CANDIDATES_PER_BAND:
        step = max(1, len(vals) // MAX_CANDIDATES_PER_BAND)
        vals = vals[::step]
    return vals


def choose_next_agm(
    token: str,
    center_coords: List[Tuple[float, float]],
    center_cum_m: List[float],
    prev: Agm,
    segment_end_mi: float,
) -> Optional[Tuple[Agm, Hop, bool]]:
    window_min = prev.along_mi + MIN_STEP_MI
    pref_max = min(prev.along_mi + PREF_MAX_STEP_MI, segment_end_mi)
    window_max = min(prev.along_mi + MAX_STEP_MI, segment_end_mi)

    if window_min > segment_end_mi:
        return None

    bands: List[Tuple[str, List[float]]] = []
    if window_min <= pref_max:
        bands.append(("preferred", make_candidate_s_values(window_min, pref_max)))
    if window_max > pref_max:
        over_start = max(window_min, pref_max + CANDIDATE_STEP_MI)
        if over_start <= window_max:
            bands.append(("over", make_candidate_s_values(over_start, window_max)))

    for band_name, s_values in bands:
        scored: List[Tuple[Agm, Hop]] = []
        for s in s_values:
            lon, lat = interpolate_point(center_coords, center_cum_m, mi_to_m(s))
            cand = Agm(idx=prev.idx + 1, along_mi=s, lon=lon, lat=lat)
            hop = hop_between(token, prev, cand)
            scored.append((cand, hop))

        if not scored:
            continue

        if band_name == "preferred":
            reachable_any = any(h.reachable for _, h in scored)
            if reachable_any:
                scored = [(c, h) for (c, h) in scored if h.reachable]

        scored.sort(
            key=lambda ch: (
                0 if ch[1].reachable else 1,
                ch[1].driving_mi,
                ch[1].end_gap_m,
            )
        )

        chosen_cand, chosen_hop = scored[0]
        used_over = band_name == "over"

        if band_name == "preferred":
            return (chosen_cand, chosen_hop, False)
        return (chosen_cand, chosen_hop, used_over)

    return None


def generate_agms_full_line(
    token: str,
    center_coords: List[Tuple[float, float]],
) -> Tuple[List[Agm], List[Hop], Dict[str, Any]]:
    cum_m = cumulative_distances_m(center_coords)
    total_mi = m_to_mi(cum_m[-1])

    seg_start = 0.0
    seg_end = total_mi

    first_lon, first_lat = interpolate_point(center_coords, cum_m, mi_to_m(seg_start))
    agms: List[Agm] = [Agm(idx=0, along_mi=seg_start, lon=first_lon, lat=first_lat)]
    hops: List[Hop] = []
    used_over_count = 0

    while True:
        prev = agms[-1]
        nxt = choose_next_agm(token, center_coords, cum_m, prev, seg_end)
        if not nxt:
            break
        next_agm, hop, used_over = nxt
        agms.append(next_agm)
        hops.append(hop)
        if used_over:
            used_over_count += 1

        if seg_end - next_agm.along_mi < MIN_STEP_MI:
            break

    total_driving = sum(h.driving_mi for h in hops if math.isfinite(h.driving_mi))
    four_wheeler_hops = [h for h in hops if not h.reachable]

    summary = {
        "segment_start_mi": round(seg_start, 3),
        "segment_end_mi": round(seg_end, 3),
        "total_line_mi": round(total_mi, 3),
        "agm_count": len(agms),
        "hop_count": len(hops),
        "total_driving_mi": round(total_driving, 2),
        "four_wheeler_hop_count": len(four_wheeler_hops),
        "used_over_preferred_count": used_over_count,
    }
    return agms, hops, summary


def agm_label(idx: int) -> str:
    # AGM 1 -> 000, AGM 2 -> 010, AGM 3 -> 020, ...
    return f"{idx * 10:03d}"


def build_kmz(center_coords: List[Tuple[float, float]], agms: List[Agm]) -> bytes:
    kml = simplekml.Kml()
    f_agms = kml.newfolder(name="AGMs")
    f_center = kml.newfolder(name="Centerline")

    ls = f_center.newlinestring(name="Centerline")
    ls.coords = [(lon, lat) for (lon, lat) in center_coords]

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


st.set_page_config(page_title="AGM Planner", layout="wide")
st.title("AGM Planner (full line, AGMs + KMZ)")

token = st.secrets.get("MAPBOX_TOKEN", "")
if not token:
    st.error('Missing Mapbox token in Streamlit Secrets. Add:\n\nMAPBOX_TOKEN = "sk.your_secret_token_here"')
    st.stop()

if "agm_results" not in st.session_state:
    st.session_state["agm_results"] = None

uploaded = st.file_uploader("Upload centerline KMZ (LineString)", type=["kmz"])

run_btn = st.button("Generate AGMs for FULL line")

if uploaded and run_btn:
    try:
        with st.spinner("Parsing KMZ..."):
            kmz_bytes = uploaded.read()
            center = kmz_to_centerline_coords(kmz_bytes)

        with st.spinner("Generating AGMs and calling Mapbox..."):
            agms, hops, summary = generate_agms_full_line(token, center)

        kmz_out = build_kmz(center, agms)

        st.session_state["agm_results"] = {
            "center": center,
            "agms": agms,
            "hops": hops,
            "summary": summary,
            "kmz_bytes": kmz_out,
        }
    except Exception as e:
        st.session_state["agm_results"] = None
        st.error("Something went wrong while generating AGMs.")
        st.exception(e)

results = st.session_state["agm_results"]
if results is None:
    if not uploaded:
        st.info("Upload a KMZ to begin, then click the button to process the entire line.")
else:
    center = results["center"]
    agms = results["agms"]
    hops = results["hops"]
    summary = results["summary"]
    kmz_out = results["kmz_bytes"]

    st.subheader("Summary (full line)")
    st.json(summary)

    st.subheader("AGMs")
    agm_rows = []
    for a in agms:
        agm_rows.append(
            {
                "AGM_label": agm_label(a.idx),
                "along_mi": round(a.along_mi, 3),
                "lon": a.lon,
                "lat": a.lat,
            }
        )
    st.dataframe(agm_rows, width="stretch")

    st.subheader("Hops (AGM → AGM)")
    hop_rows = []
    for h in hops:
        hop_rows.append(
            {
                "from_label": agm_label(h.from_idx),
                "to_label": agm_label(h.to_idx),
                "from_along_mi": round(h.from_along_mi, 3),
                "to_along_mi": round(h.to_along_mi, 3),
                "drive_mi": None if not math.isfinite(h.driving_mi) else round(h.driving_mi, 2),
                "end_gap_m": None if not math.isfinite(h.end_gap_m) else int(round(h.end_gap_m)),
                "reachable": h.reachable,
                "4_wheeler_likely": (not h.reachable),
                "mapbox_ok": h.mapbox_ok,
            }
        )
    st.dataframe(hop_rows, width="stretch")

    st.subheader("Map")
    mid = agms[len(agms) // 2]
    m = folium.Map(location=[mid.lat, mid.lon], zoom_start=11, tiles="OpenStreetMap")

    folium.PolyLine([(lat, lon) for (lon, lat) in center], color="#00E5FF", weight=4, opacity=0.9).add_to(m)

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
            tooltip=f"AGM {label} @ {a.along_mi:.2f} mi",
        ).add_to(m)

    st_folium(m, width="stretch", height=600)

    st.subheader("Download KMZ")
    st.download_button(
        label="Download KMZ (AGMs + Centerline)",
        data=kmz_out,
        file_name="agm_output.kmz",
        mime="application/vnd.google-earth.kmz",
    )
