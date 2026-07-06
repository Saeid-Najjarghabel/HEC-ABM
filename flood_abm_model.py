
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:17:45 2025

@author: saeid
"""

import os
import math
import random
import pickle
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from pyproj import Transformer
from matplotlib.animation import FFMpegWriter
import osmnx as ox
import networkx as nx
import h5py
import copy
import re

#====================================================================================================================
# ================================<< SETTINGS (Agent and HEC-RAS) >>=================================================
#====================================================================================================================

HEC_HDF_PATH = r"E:\Doctoral\PhD Thesis\Proposal\MATLAB code\Rating Curve\HECRAS-2D-T100_24h desined hydrograph_5hwarmup- with break line-With Bridges\HEC2D.p06.hdf"
HEC_FLOW_AREAS = ["Perimeter 1"]
PREFER_NATIVE_DEPTH = True
# --- Time window 
START_CLOCK = pd.Timestamp("2014-01-01 03:00:00")  
WINDOW_HOURS = 24

# --- Display clock for plots/results
DISPLAY_OFFSET_HOURS = 3  # 03:00 + 5h = 08:00
DISPLAY_START_CLOCK  = START_CLOCK + pd.Timedelta(hours=DISPLAY_OFFSET_HOURS)

STEP_MINUTES = 30  # ABM step (used for agent clock + movement)
STEP_HOURS = STEP_MINUTES / 60.0
time_step_increment = STEP_HOURS

# Kingston gauging station (from EA website)
KINGSTON_LAT = 51.409
KINGSTON_LON = -0.304

CLOSURE_DEPTH = 0.30               # default closure threshold for normal roads (m)

SPECIAL_OPEN_ROADS = {
    "Kingston Bridge": 10.0,
    "Hampton Court Way": 10.0,
}


FORCE_OPEN_FIRST_N_STEPS = 2   
FORCE_OPEN_LAST_N_STEPS  = 2   

time_step_increment = STEP_MINUTES / 60.0   # hours
scaling_factor = 200               # Each agent represents 200 people


ENABLE_ROAD_CLOSURES = True   


# ===================== LTDS-inspired population parameters ===================

# Age structure – London LTDS (all trips, collapsed to 3 groups)
age_types = ['Children', 'Adults', 'Seniors']
age_probs  = [0.20,   0.62,    0.18]

# Gender – almost balanced, very slight female majority
gender_types = ['Male', 'Female']
gender_probs = [0.49,  0.51]

# Employment status among 16+ (London, LTDS-style split)
employment_status_types = ['Employed', 'Unemployed', 'Student']
employment_probs_weekday = [0.65,       0.10,         0.25]
employment_probs_weekend = [0.65,       0.10,         0.25]

# Main travel modes around Kingston (outer SW London, LTDS modal share style)
travel_modes = ['Walkers', 'Cyclists', 'PTP', 'Drivers']

# Weekday mode shares (all trips, residents of an outer-London area like Kingston)
travel_mode_probs_weekday = {
    'Walkers': 0.27,   # walk all the way
    'Cyclists': 0.10,
    'PTP': 0.25,   # bus 
    'Drivers': 0.38    # car driver + passenger
}

# Weekend mode shares – more leisure & local trips → more walking, slightly less PT
travel_mode_probs_weekend = {
    'Walkers': 0.35,
    'Cyclists': 0.08,
    'PTP': 0.27,
    'Drivers': 0.30
}

# Speed Assumptions (km/hour)
base_speeds = {
    'PTP': 20, 
    'Drivers': 25, 
    'Walkers': 4, 
    'Cyclists': 12
}

# Social Network Parameters
k_neighbors = 4        # each agent connected to k nearest neighbors
p_rewire = 0.2         # rewiring probability
lambda_social = 0.7    # RPI Coefficient (0.7 = 70% individual, 30% social)

# ============================ CRS & TRANSFORMERS =============================
BNG = "EPSG:27700"
WGS84 = "EPSG:4326"
to_wgs84 = Transformer.from_crs(BNG, WGS84, always_xy=True)    # (E,N) -> (lon,lat)
to_bng   = Transformer.from_crs(WGS84, BNG, always_xy=True)    # (lon,lat) -> (E,N)

# ================= LONDON TRAVEL DEMAND PARAMETERS (LTDS–inspired) ===========

# Trip purpose probabilities (approx., normalised to 1.0)
WEEKDAY_PURPOSE_PROBS = {
    "Work": 0.28,
    "Education": 0.14,
    "Shopping": 0.18,
    "Leisure": 0.22,
    "Other": 0.18,
}

WEEKEND_PURPOSE_PROBS = {
    "Work": 0.05,      # very few commutes at weekend
    "Education": 0.02,
    "Shopping": 0.32,
    "Leisure": 0.45,
    "Other": 0.16,
}

# ================= LTDS trip-length distribution (London) ====================
LONDON_TRIP_LENGTH_PROFILES = {
    "2019/20": {
        "Under 1km": 0.33,
        "1-2km":     0.16,
        "2-5km":     0.21,
        "5-10km":    0.15,
        "10-20km":   0.10,
        "Over 20km": 0.04,
    },
    "2022/23": {
        "Under 1km": 0.35,
        "1-2km":     0.18,
        "2-5km":     0.20,
        "5-10km":    0.14,
        "10-20km":   0.09,
        "Over 20km": 0.05,
    },
    "2023/24": {
        "Under 1km": 0.5,
        "1-2km":     0.10,
        "2-5km":     0.85,
        #"4-10km":    0.15,
        #"10-20km":   0.09,
        #"Over 20km": 0.04,
    },
}

TRIP_LENGTH_PROFILE_YEAR = "2023/24"   
MAX_TRIP_KM = 6.0                     

TRIP_LENGTH_BINS = [
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
]

_profile = LONDON_TRIP_LENGTH_PROFILES[TRIP_LENGTH_PROFILE_YEAR]
_kept = [_profile["Under 1km"], _profile["1-2km"], _profile["2-5km"]]
_s = sum(_kept)
TRIP_LENGTH_PROBS = [p / _s for p in _kept]  


# Time-of-day windows (centre hour, weight) for each purpose and day type.
TIME_WINDOWS = {
    "weekday": {
        "Work": [
            (8.0, 3.0),   # 07–09
            (9.0, 1.5),
            (17.0, 3.0),  # 16–18
            (18.0, 2.0),
        ],
        "Education": [
            (8.0, 4.0),
            (15.5, 3.0),
        ],
        "Shopping": [
            (11.0, 2.0),
            (14.0, 2.0),
            (16.0, 1.0),
        ],
        "Leisure": [
            (12.0, 1.0),
            (18.0, 2.0),
            (20.0, 2.0),
        ],
        "Other": [
            (10.0, 1.5),
            (15.0, 1.5),
        ],
    },
    "weekend": {
        "Work": [
            (9.0, 1.0),
        ],
        "Education": [
            (10.0, 1.0),
        ],
        "Shopping": [
            (11.0, 2.0),
            (14.0, 2.5),
            (16.0, 1.5),
        ],
        "Leisure": [
            (11.0, 2.0),
            (15.0, 2.5),
            (19.0, 2.0),
        ],
        "Other": [
            (10.0, 1.5),
            (17.0, 1.5),
        ],
    }
}

# Average number of trips per person per day (Poisson mean)
AVERAGE_TRIPS_PER_DAY = 2
STEP_HOURS = STEP_MINUTES / 60.0

# ============================ RPI parameters =================================
T_DEPTH = 0.30     # tolerance threshold (m)  
K_SIG   = 10.0     # sigmoid slope
ALPHA   = 0.70     # weight of depth perception
BETA    = 0.30     # weight of official communication (alpha + beta ≈ 1)


#====================================================================================================================
# ==========================================<< Main Helper Functions >>==============================================
#====================================================================================================================

def sample_age():
    return np.random.choice(age_types, p=age_probs)

def sample_gender():
    return np.random.choice(gender_types, p=gender_probs)

def sample_employment(day_type: str):
    if day_type == "weekend":
        probs = employment_probs_weekend
    else:
        probs = employment_probs_weekday
    return np.random.choice(employment_status_types, p=probs)

def sample_travel_mode(day_type: str, age: str = None):
    if day_type == "weekend":
        probs_dict = travel_mode_probs_weekend
    else:
        probs_dict = travel_mode_probs_weekday

    modes = list(probs_dict.keys())
    probs = np.array([probs_dict[m] for m in modes], dtype=float)

    # If Child: remove driving
    if age == "Children" and "Drivers" in modes:
        i = modes.index("Drivers")
        probs[i] = 0.0
        probs = probs / probs.sum()

    return np.random.choice(modes, p=probs)

def go(p: float) -> bool:
    """Bernoulli(p) helper."""
    return random.random() < p

def sample_from_probs(items, probs):
    """Return one item from `items` according to `probs`."""
    r = random.random()
    cum = 0.0
    for item, p in zip(items, probs):
        cum += p
        if r <= cum:
            return item
    return items[-1]

def time_reached(now_h, now_m, tgt_h, tgt_m):
    """Return True if (now) has reached or passed (target) time."""
    return (now_h > tgt_h) or (now_h == tgt_h and now_m >= tgt_m)

def sample_trip_purpose(day_type: str) -> str:
    """Sample a trip purpose based on weekday/weekend probabilities."""
    if day_type == "weekend":
        items = list(WEEKEND_PURPOSE_PROBS.keys())
        probs = list(WEEKEND_PURPOSE_PROBS.values())
    else:
        items = list(WEEKDAY_PURPOSE_PROBS.keys())
        probs = list(WEEKDAY_PURPOSE_PROBS.values())
    return sample_from_probs(items, probs)

def sample_departure_time(purpose: str, day_type: str,
                          min_h: float = 0.0,
                          max_h: float = 23.5,
                          sigma: float = 0.6) -> float:
    """
    Sample departure time (hours) using TIME_WINDOWS mixture.
    - min_h/max_h constrain the valid time range (e.g., morning commute only)
    - sigma controls spread around each centre (hours)
    """
    windows = TIME_WINDOWS[day_type].get(purpose, TIME_WINDOWS[day_type]["Other"])
    centres = [c for (c, w) in windows]
    weights = np.array([w for (c, w) in windows], dtype=float)
    weights = weights / weights.sum()

    centre = random.choices(centres, weights=weights, k=1)[0]

    # sample around centre
    t = np.random.normal(loc=centre, scale=sigma)

    # clip + snap
    t = max(min_h, min(max_h, t))
    t = round(t / STEP_HOURS) * STEP_HOURS
    t = max(min_h, min(max_h, t))
    return t

def sample_trip_length_bin() -> tuple:
    """Return a (d_min_km, d_max_km) bin, always within < 5 km."""
    return sample_from_probs(TRIP_LENGTH_BINS, TRIP_LENGTH_PROBS)

def transform_coordinates(x, y):
    """From BNG (E,N) -> WGS84 (lat, lon)."""
    try:
        lon, lat = to_wgs84.transform(x, y)
        return lat, lon
    except Exception as e:
        print(f"Error transforming coordinates: {e}")
        return None, None

def wgs84_to_bng(lon, lat):
    """From WGS84 (lon, lat) -> BNG (E,N)."""
    E, N = to_bng.transform(lon, lat)
    return float(E), float(N)

def get_astar_path(G, origin_lat, origin_lon, dest_lat, dest_lon):
    """Finds the shortest path on graph G using A*."""
    try:
        orig_node = ox.distance.nearest_nodes(G, X=origin_lon, Y=origin_lat)
        dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

        path = nx.astar_path(G, orig_node, dest_node, weight='length')
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]

        total_distance = 0
        for u, v in zip(path[:-1], path[1:]):
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                first_key = list(edge_data.keys())[0]
                total_distance += min(d.get('length', 0) for d in edge_data.values())


        return coords, total_distance
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], 0
    except Exception:
        return [], 0

def choose_destination_with_length_bin(agent, purpose: str):

    
    if purpose == "Work":
        cat_keys = ["Work"]
    elif purpose == "Education":
        cat_keys = ["School"]
    elif purpose in ["Shopping", "Other"]:
        cat_keys = ["Recreation"]
    else:  # Leisure
        cat_keys = ["Recreation"]

    candidates = []
    for ck in cat_keys:
        if ck in categories:
            xs = categories[ck]["X"]
            ys = categories[ck]["Y"]
            for x, y in zip(xs, ys):
                candidates.append((x, y))

    if not candidates:
        return None  

    sampled_bin = sample_trip_length_bin()

    def bin_rank(b):
        return b[1]  
    bins_sorted = sorted(TRIP_LENGTH_BINS, key=bin_rank)  
    start_idx = bins_sorted.index(sampled_bin)

    fallback_bins = [bins_sorted[start_idx:][0]] + list(reversed(bins_sorted[:start_idx]))

    def try_find_in_bin(d_min, d_max, n_tries=100):
        best = None  
        best_dist = None

        for _ in range(n_tries):
            lat_c, lon_c = random.choice(candidates)

            _, dist_m = get_astar_path(G_base, agent["x"], agent["y"], lat_c, lon_c)
            if dist_m <= 0:
                continue

            dist_km = dist_m / 1000.0

            
            if dist_km > MAX_TRIP_KM:
                continue

            
            if d_min <= dist_km <= d_max:
                return {"X": lat_c, "Y": lon_c}, dist_km

     
            if best is None or dist_km < best_dist:
                best = (lat_c, lon_c)
                best_dist = dist_km

        if best is not None:
            lat_c, lon_c = best
            return {"X": lat_c, "Y": lon_c}, best_dist
        return None

    
    for (dmin, dmax) in fallback_bins:
        out = try_find_in_bin(dmin, dmax, n_tries=100)
        if out is not None:
            return out

    
    for _ in range(100):
        lat_c, lon_c = random.choice(candidates)
        _, dist_m = get_astar_path(G_base, agent["x"], agent["y"], lat_c, lon_c)
        if dist_m > 0 and (dist_m / 1000.0) <= MAX_TRIP_KM:
            return {"X": lat_c, "Y": lon_c}, dist_m / 1000.0

    return None

def generate_daily_trips_for_agent(agent, day_type: str):

    if agent.get("trips_generated", False):
        return

    trips = []

    if day_type == "weekday":

        if agent["employment_status"] == "Employed":

            dep_time = sample_departure_time("Work", "weekday", min_h=6.0, max_h=11.0, sigma=0.6)
            dep_step = int(round(dep_time / STEP_HOURS))

            trips.append({
                "purpose": "Work",
                "dep_time_h": dep_time,
                "dep_step": dep_step,
                "assigned": False,
                "completed": False,
                "direction": "outbound"
            })

            ret_time = sample_departure_time("Work", "weekday", min_h=15.0, max_h=20.0, sigma=0.7)
            ret_step = int(round(ret_time / STEP_HOURS))

            trips.append({
                "purpose": "Home",
                "dep_time_h": ret_time,
                "dep_step": ret_step,
                "assigned": False,
                "completed": False,
                "direction": "return"
            })

        # ---- STUDENT → EDUCATION ----
        elif agent["employment_status"] == "Student":

            # Outbound: school/university morning
            dep_time = sample_departure_time("Education", "weekday", min_h=7.0, max_h=10.5, sigma=0.5)
            dep_step = int(round(dep_time / STEP_HOURS))

            trips.append({
                "purpose": "Education",
                "dep_time_h": dep_time,
                "dep_step": dep_step,
                "assigned": False,
                "completed": False,
                "direction": "outbound"
            })

            # Return: afternoon/evening
            ret_time = sample_departure_time("Education", "weekday", min_h=14.5, max_h=19.5, sigma=0.7)
            ret_step = int(round(ret_time / STEP_HOURS))

            trips.append({
                "purpose": "Home",
                "dep_time_h": ret_time,
                "dep_step": ret_step,
                "assigned": False,
                "completed": False,
                "direction": "return"
            })

        # ---- UNEMPLOYED → MIDDAY ACTIVITIES ----
        elif agent["employment_status"] == "Unemployed":

            purpose = np.random.choice(["Shopping", "Leisure", "Other"], p=[0.45, 0.45, 0.10])

            
            dep_time = sample_departure_time(purpose, "weekday", min_h=9.0, max_h=20.0, sigma=0.9)
            dep_step = int(round(dep_time / STEP_HOURS))

            trips.append({
                "purpose": purpose,
                "dep_time_h": dep_time,
                "dep_step": dep_step,
                "assigned": False,
                "completed": False,
                "direction": "outbound"
            })

            # Return home later (still cap to 23:00)
            ret_time = dep_time + np.random.uniform(1.0, 3.0)
            ret_time = float(np.clip(ret_time, dep_time + STEP_HOURS, 23.0))
            ret_time = round(ret_time / STEP_HOURS) * STEP_HOURS
            ret_step = int(round(ret_time / STEP_HOURS))

            trips.append({
                "purpose": "Home",
                "dep_time_h": ret_time,
                "dep_step": ret_step,
                "assigned": False,
                "completed": False,
                "direction": "return"
            })


    n_extra = np.random.poisson(lam=2)

    for _ in range(n_extra):
        purpose = sample_trip_purpose(day_type)

        if purpose in ["Work", "Education"]:
            continue  # avoid duplicate commute

        dep_time_h = sample_departure_time(purpose, day_type, min_h=6.0, max_h=23.0, sigma=0.9)
        dep_step = int(round(dep_time_h / STEP_HOURS))

        trips.append({
            "purpose": purpose,
            "dep_time_h": dep_time_h,
            "dep_step": dep_step,
            "assigned": False,
            "completed": False,
            "direction": "other"
        })

    trips.sort(key=lambda t: t["dep_step"])

    agent["daily_trips"] = trips
    agent["trips_generated"] = True

#----------- Helper for Exposure matrix and RRI---------------------------------

def step_to_clock_str(step_idx: int, start_hour: int, start_min: int, step_minutes: int) -> str:
    """Convert timestep index to clock label 6-to-6 window."""
    total_min = start_hour * 60 + start_min + step_idx * step_minutes
    total_min = total_min % (24 * 60)
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"

def safe_norm(s):
    return str(s).strip()

def compute_rri_active_matrix(E_gt: np.ndarray,
                              A_gt: np.ndarray,
                              min_active_total: int = 10,
                              min_active_group: int = 3,
                              return_nan_when_low: bool = True,
                              eps: float = 1e-9) -> np.ndarray:

    G, T = E_gt.shape
    assert A_gt.shape == (G, T), "A_gt must have same shape as E_gt (G,T)."

    E_t = np.sum(E_gt, axis=0)  # (T,)
    A_t = np.sum(A_gt, axis=0)  # (T,)

    # Rates among active
    r_g = E_gt / (A_gt + eps)           # (G,T)
    r_t = E_t / (A_t + eps)             # (T,)
    RRI = r_g / (r_t[None, :] + eps)    # (G,T)

    
    if return_nan_when_low:
        
        bad_t = (A_t < min_active_total) | (r_t < eps)
        RRI[:, bad_t] = np.nan

       
        bad_g = (A_gt < min_active_group)
        RRI[bad_g] = np.nan
    else:
       
        bad_t = (A_t < min_active_total) | (r_t < eps)
        RRI[:, bad_t] = 1.0
        bad_g = (A_gt < min_active_group)
        RRI[bad_g] = 1.0

    return RRI


def _label_to_hour(x):
    """Convert label like '06:00' or '6' or 6 to an integer hour 0..23."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return int(round(float(x)))
    s = str(x).strip()
    # '06:00'
    m = re.match(r"^(\d{1,2})\s*:\s*\d{2}$", s)
    if m:
        return int(m.group(1))
    # '6' or '06'
    m = re.match(r"^(\d{1,2})$", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse hour from col label: {x}")

def plot_heatmap(matrix: np.ndarray,
                 row_labels,
                 col_labels,
                 #title: str,
                 cbar_label: str,
                 out_png: str = None,
                 vmin=None, vmax=None,
                 start_hour: int = 6):


    # Convert labels to numeric hours
    hours = np.array([int(str(lbl)[:2]) for lbl in col_labels])

    # Find index where start_hour occurs
    if start_hour not in hours:
        raise ValueError(f"start_hour {start_hour} not found in col_labels")

    start_idx = list(hours).index(start_hour)

    
    matrix_rot = np.roll(matrix, -start_idx, axis=1)
    labels_rot = col_labels[start_idx:] + col_labels[:start_idx]
    
    
    keep_hours = []
    for lbl in labels_rot:
        h = _label_to_hour(lbl)
        if start_hour <= h <= 23 or 0 <= h < 2:
            keep_hours.append(True)
        else:
            keep_hours.append(False)
    
    keep_hours = np.array(keep_hours, dtype=bool)
    matrix_rot = matrix_rot[:, keep_hours]
    labels_rot = [lbl for lbl, keep in zip(labels_rot, keep_hours) if keep]

    
    fig, ax = plt.subplots(figsize=(14, 4.8))
    
    
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color=cmap(0))   
    
    
    vmax_auto = np.nanpercentile(matrix_rot, 95)
    vmax = min(vmax, vmax_auto) if vmax is not None else vmax_auto
    
    im = ax.imshow(
        matrix_rot,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


    #ax.set_title(title, fontsize=14)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    step = max(1, len(labels_rot) // 12)
    ax.set_xticks(np.arange(0, len(labels_rot), step))
    ax.set_xticklabels([labels_rot[i] for i in range(0, len(labels_rot), step)],
                       rotation=45, ha="right")

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Group")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")

    plt.close(fig)

#====================================================================================================================
# ==========================================<< HEC-RAS 2D Model >>===================================================
#====================================================================================================================

class HECRAS2DReader:
    def __init__(self, hdf_path, flow_areas=None, prefer_native_depth=True):
        if not os.path.isfile(hdf_path):
            raise FileNotFoundError(f"HEC-RAS HDF not found: {hdf_path}")
        self.hdf_path = hdf_path
        self.prefer_native_depth = prefer_native_depth
        self._load(flow_areas)

    def _read_strings(self, dset):
        out = []
        for v in dset[:]:
            if isinstance(v, (bytes, bytearray)):
                out.append(v.decode("utf-8", errors="ignore").strip())
            elif isinstance(v, str):
                out.append(v.strip())
            else:
                out.append(str(v))
        return out

    def _first_key(self, group, keys):
        for k in keys:
            if k in group:
                return k
        return None

    def _load_time_axis(self, hdf, uts_root):
      
      tds_path = f"{uts_root}/Time Date Stamp"
      if tds_path in hdf:
          ts = self._read_strings(hdf[tds_path])
          parsed = []
          for s in ts:
              s2 = " ".join(s.strip().upper().split())
              parsed.append(pd.to_datetime(s2, format="%d%b%Y %H:%M:%S", errors="coerce"))
          t = pd.DatetimeIndex(parsed)
  
          if t.isna().any():
              # fallback parser
              t = pd.DatetimeIndex(pd.to_datetime(ts, errors="coerce"))
  
          if t.isna().any():
              bad = np.where(pd.isna(t))[0][:10]
              raise RuntimeError(f"Failed to parse some Time Date Stamp values. Bad idx example: {bad}")
  
          return t
  
      
      if f"{uts_root}/Time" in hdf:
          T = hdf[f"{uts_root}/Time"]
          if np.issubdtype(T.dtype, np.number):
              tv = T[:]
              base_key = f"{uts_root}/Time Date of First Value"
              base = pd.to_datetime(self._read_strings(hdf[base_key])[0]) if base_key in hdf else pd.Timestamp("1970-01-01")
              return pd.DatetimeIndex(base + pd.to_timedelta(tv, unit="s"))
          else:
              return pd.to_datetime(self._read_strings(T))
  
      if f"{uts_root}/Time Values" in hdf:
          tv = hdf[f"{uts_root}/Time Values"][:]
          base = pd.to_datetime(self._read_strings(hdf[f"{uts_root}/Time Date of First Value"])[0])
          return pd.DatetimeIndex(base + pd.to_timedelta(tv, unit="s"))
  
      raise KeyError("No recognized time dataset found (expected Time Date Stamp).")


    def _load(self, flow_areas):
        self.areas = {}
        with h5py.File(self.hdf_path, "r") as hdf:
            ob_root = "/Results/Unsteady/Output/Output Blocks"
            blocks = list(hdf[ob_root].keys())
            if not blocks:
                raise RuntimeError("No Output Blocks in HDF.")
            base_block = f"{ob_root}/{blocks[0]}"
            uts = f"{base_block}/Unsteady Time Series"
            self.times = self._load_time_axis(hdf, uts)

            if "2D Flow Areas" not in hdf[uts]:
                raise RuntimeError("No '2D Flow Areas' in unsteady results.")
            two_d_root = f"{uts}/2D Flow Areas"
            area_names = list(hdf[two_d_root].keys())
            if flow_areas:
                area_names = [a for a in area_names if a in flow_areas]
                if not area_names:
                    raise RuntimeError(f"Requested areas not found in HDF: {flow_areas}")

            for area in area_names:
                geom_root = f"/Geometry/2D Flow Areas/{area}"
                if geom_root not in hdf:
                    continue
                g = hdf[geom_root]

                # Cell centres
                cx = cy = None
                if "Cells Center Coordinate" in g:
                    centers = g["Cells Center Coordinate"][:]
                    cx = centers[:, 0]
                    cy = centers[:, 1]
                else:
                    xk = self._first_key(g, ["Cells Center X Coordinate", "Cells X Coordinates", "Cell Centroid X"])
                    yk = self._first_key(g, ["Cells Center Y Coordinate", "Cells Y Coordinates", "Cell Centroid Y"])
                    if xk and yk:
                        cx = g[xk][:]
                        cy = g[yk][:]

                # Perimeter
                per_xy = None
                if "Perimeter" in g:
                    per_xy = g["Perimeter"][:]

                # Elevations
                elev = None
                elev_k = self._first_key(
                    g,
                    ["Cells Minimum Elevation", "Cells Elevation", "Cell Elevation", "Cells Ground Elevation"]
                )
                if elev_k:
                    elev = g[elev_k][:]

                # Results
                area_grp = hdf[two_d_root][area]
                stage_key = self._first_key(area_grp, ["Water Surface", "Stage"])
                if not stage_key:
                    raise RuntimeError(f"No Stage/Water Surface for area '{area}'.")
                depth_key = self._first_key(area_grp, ["Depth"])

                stage_ds = area_grp[stage_key]
                n_times = len(self.times)
                n_cells_from_stage = stage_ds.shape[1] if stage_ds.shape[0] == n_times else stage_ds.shape[0]

                n_cells = n_cells_from_stage
                if cx is not None and cy is not None:
                    n_cells = min(n_cells, len(cx), len(cy))

                self.areas[area] = {
                    "n_cells": int(n_cells),
                    "cx": cx,
                    "cy": cy,
                    "elev": elev,
                    "perimeter": per_xy,
                    "stage_path": f"{two_d_root}/{area}/{stage_key}",
                    "depth_path": f"{two_d_root}/{area}/{depth_key}" if depth_key else None,
                }

            if not self.areas:
                raise RuntimeError("No 2D areas loaded from HDF.")

    @staticmethod
    def _point_in_polygon(x, y, poly_xy):
        xs, ys = poly_xy[:, 0], poly_xy[:, 1]
        n = len(xs)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = xs[i], ys[i]
            xj, yj = xs[j], ys[j]
            inter = ((yi > y) != (yj > y)) and (x < (xj - xi) *
                                                (y - yi) / ((yj - yi) + 1e-12) + xi)
            if inter:
                inside = not inside
            j = i
        return inside

    def _nearest_index(self, cx, cy, x, y):
        dx = cx - x
        dy = cy - y
        return int(np.argmin(dx * dx + dy * dy))

    def _select_area(self, preferred=None):
        if preferred and preferred in self.areas:
            return preferred
        return next(iter(self.areas.keys()))

    def get_depth_wse_at_points(self, points_EN, t_index, area=None):
        area = self._select_area(area)
        info = self.areas[area]
        depths = np.zeros(len(points_EN), dtype=float)
        wses = np.zeros(len(points_EN), dtype=float)

        with h5py.File(self.hdf_path, "r") as hdf:
            stage = hdf[info["stage_path"]][:]
            if stage.shape[0] == len(self.times):
                wse_t = stage[t_index, :]
            else:
                wse_t = stage[:, t_index]

            dep_t = None
            if self.prefer_native_depth and info["depth_path"] is not None:
                dep = hdf[info["depth_path"]][:]
                dep_t = dep[t_index, :] if dep.shape[0] == len(self.times) else dep[:, t_index]

            cx, cy = info["cx"], info["cy"]
            elev = info["elev"]
            per = info["perimeter"]

            for i, (E, N) in enumerate(points_EN):
                if per is not None and not self._point_in_polygon(E, N, per):
                    depths[i] = 0.0
                    wses[i] = 0.0
                    continue

                if cx is None or cy is None:
                    depths[i] = 0.0
                    wses[i] = 0.0
                    continue

                idx = self._nearest_index(cx, cy, E, N)
                idx = max(0, min(idx, info["n_cells"] - 1))

                w = float(wse_t[idx])
                if dep_t is not None:
                    d = float(dep_t[idx])
                else:
                    if elev is not None and idx < len(elev):
                        d = max(0.0, w - float(elev[idx]))
                    else:
                        d = 0.0
                wses[i] = w
                depths[i] = d

        return depths, wses

    def get_depth_field(self, t_index, area=None):
        area = self._select_area(area)
        info = self.areas[area]

        with h5py.File(self.hdf_path, "r") as hdf:
            stage = hdf[info["stage_path"]][:]
            if stage.shape[0] == len(self.times):
                wse_t = stage[t_index, :]
            else:
                wse_t = stage[:, t_index]

            dep_t = None
            if self.prefer_native_depth and info["depth_path"] is not None:
                dep = hdf[info["depth_path"]][:]
                dep_t = dep[t_index, :] if dep.shape[0] == len(self.times) else dep[:, t_index]

        cx, cy = info["cx"], info["cy"]
        if cx is None or cy is None:
            raise RuntimeError("No cell centers in geometry — cannot build a flood map.")

        if dep_t is not None:
            D = np.asarray(dep_t, dtype=float)
        else:
            elev = info["elev"]
            if elev is None:
                D = np.zeros_like(wse_t, dtype=float)
            else:
                D = np.maximum(0.0, np.asarray(wse_t, dtype=float) - np.asarray(elev, dtype=float))

        n_cells = info["n_cells"]
        E = np.asarray(cx[:n_cells], dtype=float)
        N = np.asarray(cy[:n_cells], dtype=float)
        D = np.asarray(D[:n_cells], dtype=float)
        return E, N, D

    def get_extent_EN(self, area=None):
        area = self._select_area(area)
        info = self.areas[area]
        if info.get("perimeter") is not None:
            per = info["perimeter"]
            xmin = float(np.min(per[:, 0]))
            xmax = float(np.max(per[:, 0]))
            ymin = float(np.min(per[:, 1]))
            ymax = float(np.max(per[:, 1]))
            return xmin, xmax, ymin, ymax
        cx, cy = info["cx"], info["cy"]
        if cx is None or cy is None:
            raise RuntimeError("No perimeter or centers to compute extent.")
        return float(np.min(cx)), float(np.max(cx)), float(np.min(cy)), float(np.max(cy))

#====================================================================================================================
# ==========================================<< Trasportation Model >>================================================
#====================================================================================================================

def get_gap(road, car_index):
    length = len(road)
    for i in range(1, length):
        next_pos = (car_index + i) % length
        if road[next_pos] != -1:
            return i - 1
    return length - 1

def na_sch_step(road, max_velocity, p_randomization):
    length = len(road)
    car_indices = np.where(road != -1)[0]

    for i in car_indices:
        gap = get_gap(road, i)
        road[i] = min(road[i] + 1, max_velocity)
        road[i] = min(road[i], gap)

    for i in car_indices:
        if np.random.rand() < p_randomization:
            road[i] = max(0, road[i] - 1)

    new_road = -1 * np.ones_like(road)
    for i in car_indices:
        new_pos = (i + road[i]) % length
        new_road[new_pos] = road[i]

    return new_road

def get_traffic_factor_from_nasch(density, base_speed_kph, road_length_cells):
    """Runs a NaSch simulation to get a dynamic traffic factor."""
    MAX_VELOCITY = 5
    P_RANDOMIZATION = 0.5
    WARM_UP_STEPS = 20
    SIM_STEPS = 10
    CELL_LENGTH_METERS = 7.5
    TIME_STEP_SECONDS = 1

    free_flow_nasch_kph = (MAX_VELOCITY * CELL_LENGTH_METERS / TIME_STEP_SECONDS) * 3.6

    num_cars = int(density * road_length_cells)
    if num_cars == 0:
        return 1.0

    road = -1 * np.ones(road_length_cells, dtype=int)
    car_positions = np.random.choice(road_length_cells, num_cars, replace=False)
    road[car_positions] = np.random.randint(0, MAX_VELOCITY + 1, num_cars)

    for _ in range(WARM_UP_STEPS):
        road = na_sch_step(road, MAX_VELOCITY, P_RANDOMIZATION)

    total_velocity = 0
    for _ in range(SIM_STEPS):
        road = na_sch_step(road, MAX_VELOCITY, P_RANDOMIZATION)
        car_velocities = road[road != -1]
        if len(car_velocities) > 0:
            total_velocity += np.mean(car_velocities)

    avg_velocity_nasch = total_velocity / SIM_STEPS if SIM_STEPS > 0 else 0

    if avg_velocity_nasch <= 0.1:
        return 10.0  # Gridlock

    avg_speed_kph = (avg_velocity_nasch * CELL_LENGTH_METERS / TIME_STEP_SECONDS) * 3.6

    traffic_factor = free_flow_nasch_kph / avg_speed_kph

    return max(1.0, traffic_factor)

def _norm_name(x):
    if x is None:
        return ""
    if isinstance(x, (list, tuple, set)):
        x = next(iter(x), "")
    return str(x).strip().lower()

def is_special_always_open_road(edge_name: str) -> bool:
    n = _norm_name(edge_name)
    # robust match (OSM names sometimes include extra words)
    return ("kingston bridge" in n) or ("hampton court way" in n)

def get_edge_threshold(edge_name: str, default_thr: float) -> float:
    # exact-key map first 
    for k, thr in SPECIAL_OPEN_ROADS.items():
        if _norm_name(k) == _norm_name(edge_name):
            return float(thr)
    # substring match fallback
    if is_special_always_open_road(edge_name):
        return 10.0
    return float(default_thr)

def get_closed_edges(edges_gdf_bng, hec, t_idx, default_threshold=0.30, n_samples=5):

    sampled_points, edge_ids = [], []
    for idx, row in edges_gdf_bng.iterrows():
        line = row.geometry
        if line is None:
            continue
        distances = np.linspace(0, line.length, n_samples)
        pts = [line.interpolate(d) for d in distances]
        for p in pts:
            sampled_points.append((p.x, p.y))
            edge_ids.append(idx)  # (u, v, key) in MultiIndex

    if not sampled_points:
        return set()

    depths, _ = hec.get_depth_wse_at_points(sampled_points, t_idx, area=HEC_FLOW_AREAS[0])

    # max depth per edge
    edge_max_depth = {}
    for depth, eid in zip(depths, edge_ids):
        d = float(depth)
        if d > edge_max_depth.get(eid, 0.0):
            edge_max_depth[eid] = d

    flooded_edges = set()
    for eid, dmax in edge_max_depth.items():
        # edge name from edges_gdf_bng (BNG gdf keeps same columns as edges_gdf)
        try:
            edge_name = edges_gdf_bng.loc[eid].get("name", None)
        except Exception:
            edge_name = None

        thr = get_edge_threshold(edge_name, default_threshold)
        if dmax > thr:
            flooded_edges.add(eid)

    return flooded_edges

def apply_road_closures_to_graph(G_base, flooded_edge_indices):
    G_t = G_base.copy()
    for eid in flooded_edge_indices:
        try:
            u, v, k = eid
            if G_t.has_edge(u, v, k):
                G_t.remove_edge(u, v, k)
            if G_t.has_edge(v, u, k):
                G_t.remove_edge(v, u, k)
        except Exception:
            continue
    return G_t

def get_water_level_at_kingston(hec_reader, t_index, area=None):
    """
    Returns water surface elevation (stage) at Kingston station from HEC-RAS 2D.
    """
    # Convert station to BNG (same CRS as HEC cells)
    E, N = wgs84_to_bng(KINGSTON_LON, KINGSTON_LAT)

    # Query depth + WSE at that single point
    depths, wses = hec_reader.get_depth_wse_at_points([(E, N)], t_index, area=area)


    H_t = float(wses[0])   # water surface elevation at Kingston
    D_t = float(depths[0])

    return H_t, D_t


# ==============================================================================================
# =====================<<  RPI (Risk Priority Index) >>=======================
# ==============================================================================================

def clamp01(x: float) -> float:
    """Clamp x into [0,1]."""
    return max(0.0, min(1.0, float(x)))


def depth_perception_sigmoid(D: float, T: float = T_DEPTH, k: float = K_SIG) -> float:
    """
    f(D) = 1 / (1 + exp(-k(D - T)))
    Nonlinear perception: small depths ~0, near T ~0.5, high depths -> 1.
    """
    D = max(0.0, float(D))
    # avoid overflow for extreme k(D-T)
    z = -k * (D - T)
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(z))


def compute_M_i(agent: dict) -> float:
    """
    Demographic amplifier:
      M_i = 1 + AgeFactor + EmploymentFactor

      Age < 12 or > 65 => +0.1
      Unemployed/financially vulnerable => +0.1

    """
    age_cat = agent.get("age", "Adults")
    emp     = agent.get("employment_status", "Employed")

    age_factor = 0.1 if age_cat in ("Children", "Seniors") else 0.0
    emp_factor = 0.1 if emp == "Unemployed" else 0.0

    return 1.0 + age_factor + emp_factor


def get_official_warning_C_t(hec_reader, t_index, area=None):
    """
    Gauge-based institutional warning rule using Kingston station water level.

      C_t = 0   if H_t < 5.15
      C_t = 0.5 if 5.15 <= H_t < 5.77
      C_t = 1   if H_t >= 5.77
    """

    try:
        H_t, D_t = get_water_level_at_kingston(hec_reader, t_index, area=area)

        # --- EA thresholds from Kingston station ---
        if H_t < 5.15:
            C_t = 0.0
        elif 5.15 <= H_t < 5.77:
            C_t = 0.5
        else:
            C_t = 1.0

        return C_t

    except Exception:
        # Safe fallback
        return 0.0


def RiskPerceptionIndex_HECRAS(agents, hec_reader, t_index,
                              alpha: float = ALPHA,
                              beta: float = BETA,
                              T: float = T_DEPTH,
                              k: float = K_SIG,
                              day_type: str = "weekday"):
    """
    NEW RPI:
      RPI_{i,t} = Clamp( ((alpha*f(D_{i,t}) + beta*C_t) * M_i), 0, 1 )
    where:
      f(D) is sigmoid depth perception
      C_t is official warning (0..1)
      M_i is demographic amplifier (>=1)
    """
    # --- points for HEC sampling ---
    pts_EN = []
    for ag in agents:
        lat = ag['x']  # Agent stores lat in x
        lon = ag['y']  # Agent stores lon in y
        E, N = wgs84_to_bng(lon, lat)
        pts_EN.append((E, N))

    depths, wses = hec_reader.get_depth_wse_at_points(pts_EN, t_index, area=HEC_FLOW_AREAS[0])


    # --- official communication value for this timestep ---
    C_t = get_official_warning_C_t(hec_reader, t_index)

    num_agents_at_risk = 0

    for i, agent in enumerate(agents):
        depth = max(0.0, float(depths[i]))
        agent['water_depth'] = depth

        if depth > CLOSURE_DEPTH:
            num_agents_at_risk += 1

        # 1) Nonlinear depth perception
        fD = depth_perception_sigmoid(depth, T=T, k=k)

        # 2) Demographic amplifier
        M_i = compute_M_i(agent)
        agent["M_i"] = M_i  

        # 3) Weighted hazard + warning, then amplify, then clamp
        raw = (alpha * fD + beta * C_t) * M_i
        agent['RPI_individual'] = raw  
        agent['RPI'] = clamp01(raw)

        
        agent['f_depth'] = fD
        agent['C_t'] = C_t

    return num_agents_at_risk, agents


def save_workspace(filename, scenario_results):
    """
    Save only the scenario_results dict to disk.
    This avoids iterating over globals() and associated issues.
    """
    workspace = {
        "scenario_results": scenario_results
    }

    with open(filename, "wb") as f:
        pickle.dump(workspace, f)

    print(f" Workspace saved to {filename}")


#------------------------------------------------------------------------------------------------------------
#-------------------------------<< Initialisation Phase>>----------------------------------------------------
#------------------------------------------------------------------------------------------------------------
print("Loading static data (Population, Buildings, Roads)...")
try:
    population_data = gpd.read_file('Agents_in_FigureBBox.shp')
    building_data = gpd.read_file('Buildings_in_FigureBBox.shp')
    
    # --------------------- Load density points as population seeds -----------------------------
    
    # --------------------- Prepare residential buildings (homes) -----------------------------
    # building_data already loaded from Buildings_in_FigureBBox.shp
    homes_bld = building_data[building_data["use"] == "RESIDENTIAL ONLY"].copy()
    
    
    population_bng = population_data.to_crs(BNG)
    homes_bng = homes_bld.to_crs(BNG)
    
    # Keep only geometry + id fields
    population_bng = population_bng[["geometry"]].copy()
    homes_bng = homes_bng[["geometry"]].copy()
    

    pop_to_home = gpd.sjoin_nearest(
        population_bng,
        homes_bng,
        how="left",
        distance_col="dist_to_home_m"
    )
    

    home_geom = homes_bng.loc[pop_to_home["index_right"]].geometry.reset_index(drop=True)
    home_cent = home_geom.centroid
    
    # centroid coords in BNG -> WGS84
    home_lon, home_lat = to_wgs84.transform(home_cent.x.values, home_cent.y.values)
    
    
    pop_cent = population_bng.geometry.centroid
    pop_lon, pop_lat = to_wgs84.transform(pop_cent.x.values, pop_cent.y.values)
    
    home_lat = np.where(np.isnan(home_lat), pop_lat, home_lat)
    home_lon = np.where(np.isnan(home_lon), pop_lon, home_lon)
    
    
    HOME_LAT_FOR_POPPOINT = home_lat
    HOME_LON_FOR_POPPOINT = home_lon


    # --------------------- Load Road Network -----------------------------
    center_point = (51.41, -0.33)
    radius = 5000
    custom_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    G_base = ox.graph_from_point(center_point, dist=radius,
                                 custom_filter=custom_filter, simplify=True)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_base)
    edges_bng = edges_gdf.to_crs(BNG)  # For flood checks

    if edges_gdf.crs.is_geographic:
        edges_utm = edges_gdf.to_crs("EPSG:32630")
        total_road_length_meters = edges_utm.geometry.length.sum()
    else:
        total_road_length_meters = edges_gdf.geometry.length.sum()

    CELL_LENGTH_METERS = 7.5
    TOTAL_ROAD_CELLS = int(total_road_length_meters / CELL_LENGTH_METERS)
    print(f"Total road network length: {total_road_length_meters:.2f} meters")
    print(f"Equivalent NaSch cells: {TOTAL_ROAD_CELLS}")
    
    
    edges_bng2 = edges_bng.copy()
    
    def is_main_highway(hw):
        # hw can be string or list in OSMnx
        if isinstance(hw, (list, tuple, set)):
            hw = hw[0] if len(hw) else None
        return hw in {"motorway", "trunk", "primary", "secondary"}
    
    
    main_edges_bng = edges_bng2[edges_bng2["highway"].apply(is_main_highway)].copy()
    
    def sample_line_points(line, n_samples=10):
        ds = np.linspace(0, line.length, n_samples)
        return [(line.interpolate(d).x, line.interpolate(d).y) for d in ds]
    
    
    road_pts_EN = []                 # sampled points on closed roads (BNG)
    road_pts_edge_ids = []           # mapping each point -> edge id (u,v,key)
    
    
##-----------------------------------------------------------------------------

    # Initialize categories
    categories = {
        'Home': {'X': [], 'Y': []},
        'School': {'X': [], 'Y': []},
        'Work': {'X': [], 'Y': []},
        'Recreation': {'X': [], 'Y': []}
    }

    for _, row in building_data.iterrows():
        building_type = row['use']
        xx = row['X_Coordina']
        yy = row['Y_Coordina']
        lat, lon = transform_coordinates(xx, yy)
        if lat is None:
            continue

        if building_type == 'RESIDENTIAL ONLY':
            categories['Home']['X'].append(lat)
            categories['Home']['Y'].append(lon)
        elif building_type == 'COMMUNITY - EDUCATIONAL':
            categories['School']['X'].append(lat)
            categories['School']['Y'].append(lon)
        elif building_type in [
            'RETAIL WITH OFFICE/RESIDENTIAL ABOVE', 'OFFICE ONLY',
            'COMMUNITY - GOVERNMENTAL (CENTRAL AND LOCAL)',
            'GENERAL COMMERCIAL - MIXED USE', 'RETAIL ONLY',
            'RETAIL - WITH MORE RECENT EXTENSIONS OF DIFFERENT TYPE CONSTRUCTION/AGE'
        ]:
            categories['Work']['X'].append(lat)
            categories['Work']['Y'].append(lon)
        elif building_type in ['RECREATION AND LEISURE', 'COMMUNITY - RELIGIOUS']:
            categories['Recreation']['X'].append(lat)
            categories['Recreation']['Y'].append(lon)

    # Entry points for external agents
    entry_points = [
        {"name": "North West Entrance", "x": 51.4480, "y": -0.3935},
        {"name": "North Entrance", "x": 51.4505, "y": -0.3400},
        {"name": "North East Entrance", "x": 51.4485, "y": -0.2780},
        {"name": "East Entrance", "x": 51.4250, "y": -0.2650},
        {"name": "South East Entrance", "x": 51.3800, "y": -0.2700},
        {"name": "South Entrance", "x": 51.3700, "y": -0.3100},
        {"name": "South West Entrance", "x": 51.3700, "y": -0.3700},
        {"name": "West Entrance", "x": 51.4000, "y": -0.3950},
        {"name": "A3 Entrance", "x": 51.4150, "y": -0.3850},
        {"name": "M4 Junction", "x": 51.4300, "y": -0.3600},
        {"name": "Twickenham Entrance", "x": 51.4400, "y": -0.3450},
        {"name": "Wandsworth Entrance", "x": 51.4450, "y": -0.3200},
        {"name": "Battersea Entrance", "x": 51.4700, "y": -0.2900},
        {"name": "Richmond Entrance", "x": 51.4600, "y": -0.3050},
        {"name": "Tooting Entrance", "x": 51.4200, "y": -0.3900},
        {"name": "Sutton Entrance", "x": 51.3600, "y": -0.3200},
        {"name": "Croydon Entrance", "x": 51.3700, "y": -0.2800},
        {"name": "Lewisham Entrance", "x": 51.3850, "y": -0.2600},
        {"name": "Fulham Entrance", "x": 51.4700, "y": -0.2200},
        {"name": "Epsom Entrance", "x": 51.3400, "y": -0.3200}
    ]

    hec = HECRAS2DReader(
        HEC_HDF_PATH,
        flow_areas=HEC_FLOW_AREAS,
        prefer_native_depth=PREFER_NATIVE_DEPTH
    )
    print(f"[HEC] Loaded {len(hec.areas)} 2D area(s). HDF timesteps = {len(hec.times)}")
    
    # ------------------ filter HEC timesteps to a 24h window ------------------
    end_clock = START_CLOCK + pd.Timedelta(hours=WINDOW_HOURS)
    mask = (hec.times >= START_CLOCK) & (hec.times < end_clock)
    hec_indices = np.where(mask)[0].tolist()
    
    if len(hec_indices) == 0:
        raise RuntimeError(f"No HEC timesteps found in window {START_CLOCK} to {end_clock}")
    
    used_times = hec.times[hec_indices]
    total_time_steps = len(hec_indices)     # <-- IMPORTANT: ABM loop length becomes HEC-driven
    
    print(f"[HEC] Window steps found: {total_time_steps}")
    print(f"[HEC] Window range: {used_times.min()} -> {used_times.max()}")
    
    if len(used_times) >= 2:
        dt_min = np.median(np.diff(used_times.values).astype("timedelta64[m]").astype(int))
        print(f"[HEC] Median timestep in window: {dt_min} minutes")


except Exception as e:
    print(f"!!! CRITICAL ERROR during Initialization: {e} !!!")
    raise

#-------------------------------<< Agent Creation >>---------------------------------------------------------

def create_agents():
    """Creates a new list of agents with initial properties + daily-trip flags."""
    agents = []

    # Local residents from population_data
    for idx, row in population_data.iterrows():
        # Use assigned home building lat/lon (not the random density point)
        x = float(HOME_LAT_FOR_POPPOINT[idx])   # agent lat
        y = float(HOME_LON_FOR_POPPOINT[idx])   # agent lon
        
        age = sample_age()
        
        agents.append(
            {
                "x": x,
                "y": y,
                "home_lat": x,
                "home_lon": y,
                "activity": "Home",
                "target": None,
                "speed": None,
                "age": age,
                "travelMode": sample_travel_mode(day_type="weekday", age=age),
                "path": None,
                "stay_until": None,
                "traffic_factor": None,
                "agent_type": "Local",
                "movement_history": None,
                "DistanceToTarget": 0.0,
                "has_reached_target": False,
                "flood_risk": None,
                "gender": sample_gender(),
                "employment_status": sample_employment(day_type="weekday"),
                "I_media": random.uniform(0.1, 0.3),
                "demographic_modifier": None,
                "location_risk_factor": None,
                "RPI": None,
                "trust_official": random.uniform(0.5, 1.0),
                "cancelled_trip": False,
                "baseline_dist": 0.0,
                "flooded_dist": 0.0,
                "excess_dist": 0.0,
                "excess_time": 0.0,
                "unserved": False,
                "neighbor_mean_rpi_t_minus_1": 0.0,
                # LTDS trip planning
                "trips_generated": False,
                "daily_trips": [],
                "trip_active": False,
                "trip_departure_step": None,
                "trip_departure_label": None,
                "current_baseline_time_hr": 0.0,
                "current_flooded_time_hr": 0.0,
                "previous_activity": "Home",
                "waiting_to_depart": False,
                "staying": False,
                "cooldown_steps": 0,              
                "resume_activity": None,          
                "resume_target": None,            
                "unserved_state": False,
                "unserved_since_step": None,   



            }
        )

    # External agents from entry_points
    for entry in entry_points:
        for _ in range(2):
            age = sample_age()
            agents.append(
                {
                    "x": entry["x"],
                    "y": entry["y"],
                    "home_lat": entry["x"],
                    "home_lon": entry["y"],

                    "activity": "Home",
                    "target": None,
                    "speed": None,
                    "age": age,
                    "travelMode": sample_travel_mode(day_type="weekday", age=age),
                    "path": None,
                    "movement_history": [],
                    "stay_until": None,
                    "DistanceToTarget": 0.0,
                    "has_reached_target": False,
                    "origin_entry": entry["name"],
                    "agent_type": random.choice(["visitor", "commuter"]),
                    "flood_risk": None,
                    "gender": sample_gender(),
                    "employment_status": sample_employment(day_type="weekday"),
                    "I_media": random.uniform(0.1, 0.3),
                    "demographic_modifier": None,
                    "location_risk_factor": None,
                    "RPI": None,
                    "trust_official": random.uniform(0.5, 1.0),
                    "cancelled_trip": False,
                    "baseline_dist": 0.0,
                    "flooded_dist": 0.0,
                    "excess_dist": 0.0,
                    "excess_time": 0.0,
                    "unserved": False,
                    "neighbor_mean_rpi_t_minus_1": 0.0,
                    "trips_generated": False,
                    "daily_trips": [],
                    "trip_active": False,
                    "trip_departure_step": None,
                    "trip_departure_label": None,
                    "current_baseline_time_hr": 0.0,
                    "current_flooded_time_hr": 0.0,
                    "previous_activity": "Home",
                    "waiting_to_depart": False,
                    "staying": False,
                    "cooldown_steps": 0,              
                    "resume_activity": None,          
                    "resume_target": None,            
                    "unserved_state": False,          
                    "unserved_since_step": None,      



                }
            )



        for agent in agents:
            # If Child: force Student + forbid driving
            if agent["age"] == "Children":
                agent["employment_status"] = "Student"
        
                # forbid driving for children
                if agent["travelMode"] == "Drivers":
                    agent["travelMode"] = np.random.choice(
                        ["Walkers", "Cyclists", "PTP"],
                        p=[0.55, 0.10, 0.35]   
                    )
        
            # speed after fixing mode
            agent["speed"] = base_speeds[agent["travelMode"]]
        
            agent["demographic_modifier"] = compute_M_i(agent)


    # Social network
    N = len(agents)
    G_social = nx.watts_strogatz_graph(N, k_neighbors, p_rewire)
    for i, agent in enumerate(agents):
        nbrs = list(G_social.neighbors(i))
        agent["neighbors"] = nbrs
        agent["social_weights"] = {j: 1 / len(nbrs) for j in nbrs} if nbrs else {}

    # Initial RPI
    _, agents = RiskPerceptionIndex_HECRAS(agents, hec, 0)

    print(f"Created {len(agents)} total agents.")
    return agents


#--------------------------------------------------------------------------------------------------------------
#---------------<< Activity generation & update (LTDS-based) >>-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def update_agent_activity(agent, current_hour, current_minute, day_type: str):
    """
    Convert planned trips into actual activities (Home -> Work, etc.)
    and also handle return trips when stay_until is reached.
    """
    # ensure the agent has a daily plan
    generate_daily_trips_for_agent(agent, day_type)

    now_h = current_hour + current_minute / 60.0
    current_step = int(round(now_h / STEP_HOURS)) 
    
    agent["waiting_to_depart"] = False
    agent["staying"] = False
    
    # If agent is currently in an activity (has stay_until), they are "staying"
    if agent.get("stay_until") is not None:
        sh, sm = agent["stay_until"]
        if not time_reached(current_hour, current_minute, sh, sm):
            agent["staying"] = True
            return  # nothing else should start while staying

    for trip in agent["daily_trips"]:
        if trip["assigned"] or trip["completed"]:
            continue
        
        # Find first upcoming trip that isn't assigned/completed
        if current_step < trip["dep_step"] and current_step >= (trip["dep_step"] - 1):
            agent["waiting_to_depart"] = True


        # >>> exact step trigger
        if current_step == trip["dep_step"]:
            purpose = trip["purpose"]
            rpi = agent.get("RPI", 0.0)
            
            if purpose == "Home" and trip.get("direction") == "return":
                agent["activity"] = "Home"
                
                agent["target"] = {"X": agent["home_lat"], "Y": agent["home_lon"]}  # stored at creation

                agent["has_reached_target"] = False
                agent["cancelled_trip"] = False
                agent["baseline_dist"] = 0.0
                agent["flooded_dist"] = 0.0
                agent["excess_dist"] = 0.0
                agent["excess_time"] = 0.0
                agent["trip_active"] = False
                agent["trip_departure_step"] = None
                agent["trip_departure_label"] = None
                
                trip["assigned"] = True
                break
                
            # RPI thresholds: cancel trip if risk too high
            if purpose in ["Work", "Education"] and rpi > 0.9:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue
            if purpose in ["Shopping", "Leisure", "Other"] and rpi > 0.8:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue

            # choose destination with realistic trip length
            dest_info = choose_destination_with_length_bin(agent, purpose)
            if dest_info is None:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue

            target, dist_km = dest_info
            agent["target"] = target

            # set activity label based on purpose
            if purpose == "Work":
                agent["activity"] = "Work"
            elif purpose == "Education":
                agent["activity"] = "School"
            elif purpose == "Shopping":
                agent["activity"] = "Shop"
            elif purpose == "Leisure":
                agent["activity"] = "Recreation"
            else:
                agent["activity"] = random.choice(["Recreation", "Shop"])

            # store origin activity for this trip
            agent["previous_activity"] = "Home" if agent["activity"] != "Home" else "Home"

            # length of stay: rough rule-of-thumb by purpose
            if agent["activity"] in ["Work", "School"]:
                stay_hours = random.uniform(3.0, 5.0)
            elif agent["activity"] == "Shop":
                stay_hours = random.uniform(0.5, 1.0)
            else:
                stay_hours = random.uniform(1.0, 2.0)

            end_time = now_h + stay_hours
            end_time = min(end_time, 23.9)
            stay_h = int(end_time)
            stay_m = int(round((end_time - stay_h) * 60))
            agent["stay_until"] = [stay_h, stay_m]

            # reset movement flags for new trip
            agent["has_reached_target"] = False
            agent["cancelled_trip"] = False
            agent["baseline_dist"] = 0.0
            agent["flooded_dist"] = 0.0
            agent["excess_dist"] = 0.0
            agent["excess_time"] = 0.0
            agent["trip_active"] = False   # will be set when path is found
            agent["trip_departure_step"] = None
            agent["trip_departure_label"] = None

            trip["assigned"] = True
            break  # only start one new trip per timestep

    # 2) Return trips: when stay_until reached at destination -> go Home (or Exit for visitors)
    if agent.get("stay_until") is not None:
        sh, sm = agent["stay_until"]
        if time_reached(current_hour, current_minute, sh, sm):
            agent["stay_until"] = None



def plot_agents_snapshot(
    agents_t,
    hec,
    t_idx,
    title_time_str,
    edges_gdf=None,
    out_png=None,
    hec_area=None,
    rpi_thr=0.8,
    depth_thr=0.30,
    lonlim=(-0.36, -0.29),
    latlim=(51.37, 51.44),
):
    """
    One snapshot map:
      - Flood depth scatter (HEC 2D)
      - Roads
      - Agents meeting: (RPI >= rpi_thr) OR (water_depth >= depth_thr)
      - Marker shape = travel mode
      - Color = age group
    """

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    #ax.set_title(f"Agents with RPI>{rpi_thr} or Water depth>{depth_thr} m", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(*lonlim)
    ax.set_ylim(*latlim)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_edgecolor('black')

    # ---------- Flood map ----------
    try:
        E, N, D = hec.get_depth_field(t_idx, area=hec_area)
        lon, lat = to_wgs84.transform(E, N)

        flood_mask = D > 0.01
        lon_f, lat_f, D_f = lon[flood_mask], lat[flood_mask], D[flood_mask]

        sc = ax.scatter(
            lon_f, lat_f, c=D_f, s=3,
            cmap="Blues", vmin=0.1, vmax=6.0,
            alpha=0.9, zorder=1
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Water Depth (m)")
        cbar.outline.set_linewidth(1.2)
        cbar.outline.set_edgecolor("black")
    except Exception as e:
        sc = None

    # ---------- Roads ----------
    if edges_gdf is not None:
        try:
            edges_gdf.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.7, zorder=2)
        except Exception:
            pass

    try:
        pts_EN = []
        for ag in agents_t:
            E_ag, N_ag = wgs84_to_bng(ag["y"], ag["x"])
            pts_EN.append((E_ag, N_ag))
        depths, _ = hec.get_depth_wse_at_points(pts_EN, t_idx, area=hec_area)

        for ag, d in zip(agents_t, depths):
            ag["water_depth"] = float(max(0.0, d))
    except Exception:
        # If fails, keep existing agent["water_depth"] if present
        pass

    # ---------- Filter agents of interest ----------
    selected = [
        ag for ag in agents_t
        if (ag.get("RPI", 0.0) >= rpi_thr) or (ag.get("water_depth", 0.0) >= depth_thr)
    ]

    # ---------- Styles ----------
    age_color = {
        "Children":  "tab:blue",
        "Adults":  "tab:orange",
        "Seniors": "tab:green"
    }
    mode_marker = {
        "Walkers": "o",
        "Cyclists": "^",
        "PTP": "s",
        "Drivers": "D"
    }

    # ---------- Plot agents (by mode to get correct legend) ----------
    for mode, mk in mode_marker.items():
        subset_mode = [ag for ag in selected if ag.get("travelMode") == mode]
        if not subset_mode:
            continue

        xs = [ag["y"] for ag in subset_mode]  # lon
        ys = [ag["x"] for ag in subset_mode]  # lat
        cs = [age_color.get(ag.get("age", "Adults"), "tab:orange") for ag in subset_mode]

        ax.scatter(
            xs, ys,
            s=55,
            marker=mk,
            c=cs,
            edgecolor="black",
            linewidths=0.8,
            alpha=0.95,
            zorder=10
        )

    # ---------- Time label ----------
    ax.text(
        0.02, 0.98, f"Time: {title_time_str}",
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=1.0),
        zorder=50
    )

    # ---------- Legends (Age + Travel Mode) ----------
    age_handles = [
        plt.Line2D([0],[0], marker='o', color='w', label='Children',
                   markerfacecolor=age_color["Children"], markeredgecolor="black", markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='Adults',
                   markerfacecolor=age_color["Adults"], markeredgecolor="black", markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='Seniors',
                   markerfacecolor=age_color["Seniors"], markeredgecolor="black", markersize=8),
    ]
    mode_handles = [
        plt.Line2D([0],[0], marker=mode_marker["Walkers"], color='w', label='Walkers',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["Cyclists"], color='w', label='Cyclists',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["PTP"], color='w', label='PTP',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["Drivers"], color='w', label='Drivers',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
    ]

    leg1 = ax.legend(handles=age_handles, title="Age", loc="lower left",
                     frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")
    leg1.get_frame().set_linewidth(1.2)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=mode_handles, title="Travel Mode", loc="lower right",
                     frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")
    leg2.get_frame().set_linewidth(1.2)

    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

#--------------------------------------------------------------------------------------------------------------
#---------------<< Main Simulation Loop >>--------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def run_simulation(scenario_name='scenario_name', day_type= 'day_type'):
    """
    Runs the full ABM simulation for the road closure scenario.
    Uses realised trip times with departure/arrival tracking.
    """
    print(f"\n--- Starting Simulation: {scenario_name} ---")

    agents = create_agents()
    time_of_day = 6.0  # Starting time (8:00 AM)

    agents_over_time = []
    completed_trips = []  # list of completed trips for post-analysis

    # --- Metric Lists ---
    metrics = {
        'total_agents':[],
        'agents_at_risk': [],
        'people_at_risk': [],
        'unserved_agents_share': [],
        'total_excess_dist_km': [],
        'total_excess_time_hours': [],
        'average_excess_time_mins': [],
        'traveling_agent_count': [],
        'idle_agents_share': [],
        'closed_roads_count': [],
        'closed_roads_share': [],   
        'unserved_agents_share_state': [],  
        'state_idle_count':[],
        "state_travelling_count":[],
        "state_unserved_count":[],
        "attempting_agents_t":[],
        "departures_by_step":[],
        'travelling_agents_share':[],
        'state_travelling_share':[],
        'state_sum_check':[],
    
        
        'attempting_agents_count': [],
        'unserved_agents_share_attempting': [],
        'unserved_agents_count_state': [],
        'mean_rpi': [],
        'median_rpi': [],   

    }
    
    metrics.setdefault("departures_by_step", [])

    
    # demographic exposure in flooded area (depth > CLOSURE_DEPTH)
    metrics['flooded_by_age'] = {age: [] for age in age_types}
    metrics['flooded_by_emp'] = {emp: [] for emp in employment_status_types}
    metrics['flooded_by_mode'] = {mode: [] for mode in travel_modes}

    video_file_name = f"HEC_ABM_high agent_Weekday_18-02-26_Unserved agents_NewRPI_{scenario_name}.mp4"

    # --- Setup Figure and Colorbar *before* the loop ---
    fig, ax = plt.subplots(figsize=(8, 8))
    #ax.set_title("Road Closure Scenario")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-0.36, -0.29)
    ax.set_ylim(51.37, 51.44)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.08, top=0.94)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_edgecolor('black')

    dummy_scatter = ax.scatter([], [], c=[], s=3, cmap='Blues', vmin=0, vmax=6.0, alpha=0.9, zorder=1)
    cbar = fig.colorbar(dummy_scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Water Depth (m)")
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')

    fig.tight_layout()

    frames_dir = f"frames_weekday_high agent_18-02-26_ExposureScenario_NewRPI_{scenario_name}"
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Saving all frames to '{frames_dir}/' directory...")

    writer = FFMpegWriter(fps=2, metadata={'title': f'Agent Movement_weekday_high agent-18-02-26_ExposureScenario_NewRPI_{scenario_name}'})
    
    
    
    # ---------------------------------------------------------
    # Exposure tracking settings
    # ---------------------------------------------------------
    START_HOUR = 6
    START_MIN  = 0
    STEP_MINUTES = int(STEP_HOURS * 60)  
    N_STEPS = total_time_steps  
    
    # Choose consistent category lists (important for stable heatmaps)
    AGE_GROUPS = ["Children", "Adults", "Seniors"]
    EMP_GROUPS = ["Employed", "Unemployed", "Student"]
    MODE_GROUPS = ["Walkers", "Cyclists", "PTP", "Drivers"]
    
    # Total population per group (computed once from ALL agents)
    def count_group_totals(agents, key, allowed_groups):
        counts = {g: 0 for g in allowed_groups}
        for a in agents:
            v = safe_norm(a.get(key, "Other"))
            if v in counts:
                counts[v] += 1
        return np.array([counts[g] for g in allowed_groups], dtype=float)
    
    
    # Exposure count matrices (G,T)
    E_age  = np.zeros((len(AGE_GROUPS), N_STEPS), dtype=float)
    E_emp  = np.zeros((len(EMP_GROUPS), N_STEPS), dtype=float)
    E_mode = np.zeros((len(MODE_GROUPS), N_STEPS), dtype=float)
    
    # >>> ADD THIS HERE <<<
    A_age  = np.zeros((len(AGE_GROUPS), N_STEPS), dtype=float)
    A_emp  = np.zeros((len(EMP_GROUPS), N_STEPS), dtype=float)
    A_mode = np.zeros((len(MODE_GROUPS), N_STEPS), dtype=float)
    
    # Time labels for plotting
    time_labels = [step_to_clock_str(s, START_HOUR, START_MIN, STEP_MINUTES) for s in range(N_STEPS)]


    with writer.saving(fig, video_file_name, dpi=100):
        for t in range(total_time_steps):
            # ABM clock (08:00 -> 08:00+24h)
            current_hour = int(np.floor(time_of_day)) % 24
            current_minute = int(round((time_of_day - np.floor(time_of_day)) * 60)) % 60
            time_label = f"{current_hour:02d}:{current_minute:02d}"
            
            for ag in agents:
                ag["unserved"] = False



            # exact mapping to the filtered 24h HEC window
            t_idx = hec_indices[t]


            # 1. Update RPI based on HEC-RAS depth (Individual RPI)
            num_agents_at_risk, agents = RiskPerceptionIndex_HECRAS(agents, hec, t_idx, day_type=day_type)
            metrics['agents_at_risk'].append(num_agents_at_risk)
            metrics['people_at_risk'].append(num_agents_at_risk * scaling_factor)

            # 2. Social Network RPI Update
            new_rpis = {}
            for i, agent in enumerate(agents):
                individual_rpi = agent.get('RPI', 0)
                if agent['neighbors']:
                    neighbor_rpis = [agents[j].get('RPI', 0) for j in agent['neighbors']]
                    social_rpi = np.mean(neighbor_rpis) if neighbor_rpis else 0
                    updated_rpi = (lambda_social * individual_rpi) + ((1 - lambda_social) * social_rpi)
                    new_rpis[i] = min(1.0, updated_rpi)
                else:
                    new_rpis[i] = individual_rpi
            for i, agent in enumerate(agents):
                agent['RPI'] = new_rpis.get(i)
                
            metrics.setdefault("mean_rpi", []).append(
            float(np.mean([agent.get("RPI", 0.0) for agent in agents]))
            )
            metrics.setdefault("median_rpi", []).append(
                float(np.median([agent.get("RPI", 0.0) for agent in agents]))
)
        

            force_open_now = (t < FORCE_OPEN_FIRST_N_STEPS) or (t >= total_time_steps - FORCE_OPEN_LAST_N_STEPS)
            
            if ENABLE_ROAD_CLOSURES and (not force_open_now):
                flooded_edge_indices = get_closed_edges(
                    edges_bng,
                    hec,
                    t_idx,
                    default_threshold=CLOSURE_DEPTH,
                    n_samples=5
                )
                G_t = apply_road_closures_to_graph(G_base, flooded_edge_indices)
            else:
                flooded_edge_indices = set()
                G_t = G_base  # intact network
            
            closed_count = len(flooded_edge_indices)
            metrics['closed_roads_count'].append(closed_count)
            
            total_edges = len(edges_gdf)
            metrics['closed_roads_share'].append(100.0 * closed_count / max(1, total_edges))


            if ENABLE_ROAD_CLOSURES:
                closed_main_edge_indices = [
                    eid for eid in flooded_edge_indices
                    if eid in main_edges_bng.index
                ]
            
                # (2) sample points on those closed edges
                road_pts_EN = []
                road_pts_edge_ids = []
                for eid in closed_main_edge_indices:
                    geom = main_edges_bng.loc[eid].geometry
                    if geom is None:
                        continue
                    pts = sample_line_points(geom, n_samples=8)
                    road_pts_EN.extend(pts)
                    road_pts_edge_ids.extend([eid] * len(pts))
            
                if road_pts_EN:
                    depths_pts, _ = hec.get_depth_wse_at_points(road_pts_EN, t_idx)
            
                    edge_max_depth = {}
                    for d, eid in zip(depths_pts, road_pts_edge_ids):
                        if d > edge_max_depth.get(eid, 0.0):
                            edge_max_depth[eid] = float(d)
            
                    closed_edge_depths = list(edge_max_depth.values())
                    depth_max_closed = max(closed_edge_depths) if closed_edge_depths else 0.0
                    depth_mean_closed = float(np.mean(closed_edge_depths)) if closed_edge_depths else 0.0
                else:
                    depth_max_closed = 0.0
                    depth_mean_closed = 0.0
            
                metrics.setdefault("closed_main_roads_count", []).append(len(closed_main_edge_indices))
            else:
                # No closures => define closure-depth metrics as 0
                depth_max_closed = 0.0
                depth_mean_closed = 0.0
                metrics.setdefault("closed_main_roads_count", []).append(0)
            
            metrics.setdefault("closed_roads_depth_max", []).append(depth_max_closed)
            metrics.setdefault("closed_roads_depth_mean", []).append(depth_mean_closed)

            flooded_agents = [
                ag for ag in agents
                if ag.get('water_depth', 0.0) > CLOSURE_DEPTH
            ]
            
            age_counts = {age: 0 for age in age_types}
            emp_counts = {emp: 0 for emp in employment_status_types}
            mode_counts = {mode: 0 for mode in travel_modes}
            
            for ag in flooded_agents:
                a = ag.get('age')
                e = ag.get('employment_status')
                m = ag.get('travelMode')
            
                if a in age_counts:
                    age_counts[a] += 1
                if e in emp_counts:
                    emp_counts[e] += 1
                if m in mode_counts:
                    mode_counts[m] += 1
            
            # Store time-resolved exposure counts
            for age in age_types:
                metrics['flooded_by_age'][age].append(age_counts[age]* scaling_factor)
            for emp in employment_status_types:
                metrics['flooded_by_emp'][emp].append(emp_counts[emp]* scaling_factor)
            for mode in travel_modes:
                metrics['flooded_by_mode'][mode].append(mode_counts[mode]* scaling_factor)
                
                
            step = t  
            
            
            def is_active(a):
                return (
                    a.get("trip_active", False)
                    or a.get("waiting_to_depart", False)
                    or a.get("staying", False)
                )
            
            def is_exposed(a):
                return (a.get("water_depth", 0.0) > CLOSURE_DEPTH)
            
            # Count exposures
            age_idx = {g:i for i,g in enumerate(AGE_GROUPS)}
            emp_idx = {g:i for i,g in enumerate(EMP_GROUPS)}
            mode_idx = {g:i for i,g in enumerate(MODE_GROUPS)}
            
            for a in agents:
                if not is_exposed(a):
                    continue
            
                ag = safe_norm(a.get("age", ""))
                em = safe_norm(a.get("employment_status", ""))
                md = safe_norm(a.get("travelMode", ""))
            
                if ag in age_idx:
                    E_age[age_idx[ag], step] += 1
                if em in emp_idx:
                    E_emp[emp_idx[em], step] += 1
                if md in mode_idx:
                    E_mode[mode_idx[md], step] += 1
    



            total_excess_dist_t = 0.0
            total_excess_time_t = 0.0
            traveling_agents_t = 0
            unserved_agents_t = 0
            idle_agents_t = 0
            attempting_agents_t = 0
            departures_this_step = 0
            N_total = len(agents)
            #staying_count = 0



            for i, agent in enumerate(agents):
                # reset per-step disruption (but NOT path / DistanceToTarget)
                agent['baseline_dist'] = 0.0
                agent['flooded_dist'] = 0.0
                agent['excess_dist'] = 0.0
                agent['excess_time'] = 0.0
                #agent['unserved'] = False
                
                
                # If cooldown ended previously and we stored a planned trip, restore it now
                if agent.get("resume_target") is not None and agent.get("target") is None and agent.get("activity") == "Idle":
                    agent["target"] = agent["resume_target"]
                    agent["activity"] = agent["resume_activity"] if agent.get("resume_activity") else agent["activity"]
                    agent["resume_target"] = None
                    agent["resume_activity"] = None


                # Activity-based demand using LTDS-inspired model
                update_agent_activity(agent, current_hour, current_minute, day_type)
                
                
                age_idx  = {g:i for i,g in enumerate(AGE_GROUPS)}
                emp_idx  = {g:i for i,g in enumerate(EMP_GROUPS)}
                mode_idx = {g:i for i,g in enumerate(MODE_GROUPS)}
                
                step = t
                
                for a in agents:
                    ag = safe_norm(a.get("age", ""))
                    em = safe_norm(a.get("employment_status", ""))
                    md = safe_norm(a.get("travelMode", ""))
                
                    if is_active(a) and is_exposed(a):
                        if ag in age_idx:   E_age[age_idx[ag], step]   += 1
                        if em in emp_idx:   E_emp[emp_idx[em], step]   += 1
                        if md in mode_idx:  E_mode[mode_idx[md], step] += 1
                

                has_target = agent.get('target') is not None
                is_travelling_activity = (agent['activity'] != 'Idle') and has_target
                wants_to_travel = has_target and is_travelling_activity and not agent.get('has_reached_target', False)

                if wants_to_travel:
                    # New trip starting now
                    if not agent.get('trip_active', False):
                        attempting_agents_t += 1   # only for newly starting trips
                        departures_this_step += 1
                        traffic_density = 0.6 if 8 <= current_hour < 10 or 17 <= current_hour < 19 else 0.25
                        global_traffic_factor = get_traffic_factor_from_nasch(
                            traffic_density, base_speeds['Drivers'], TOTAL_ROAD_CELLS)

                        if agent.get('travelMode') in ['Drivers', 'PTP']:
                            agent['speed'] = base_speeds[agent['travelMode']] / global_traffic_factor
                        else:
                            agent['speed'] = base_speeds[agent['travelMode']]

                        # Baseline (no flooding)
                        _, base_dist_m = get_astar_path(
                            G_base,
                            agent['x'], agent['y'],
                            agent['target']['X'], agent['target']['Y']
                        )

                        # Flooded network
                        flooded_path_coords, flooded_dist_m = get_astar_path(
                            G_t,
                            agent['x'], agent['y'],
                            agent['target']['X'], agent['target']['Y']
                        )

                        if not flooded_path_coords or base_dist_m <= 0:
                            # Trip unserved in current network (this step)
                            agent["unserved"] = True                 # per-step marker (video)
                            agent["unserved_state"] = True           # persistent state (chart)
                            agent["unserved_since_step"] = t
                            unserved_agents_t += 1
                        
                            agent["path"] = []
                            agent["DistanceToTarget"] = 0.0
                            agent["trip_active"] = False
                            agent["has_reached_target"] = False
    
            
                        else:
                            # Start trip
                            agent['baseline_dist'] = base_dist_m / 1000.0
                            agent['flooded_dist'] = flooded_dist_m / 1000.0
                            if agent['speed'] > 0:
                                agent['current_baseline_time_hr'] = agent['baseline_dist'] / agent['speed']
                                agent['current_flooded_time_hr'] = agent['flooded_dist'] / agent['speed']
                            else:
                                agent['current_baseline_time_hr'] = 0.0
                                agent['current_flooded_time_hr'] = 0.0

                            agent['excess_dist'] = agent['flooded_dist'] - agent['baseline_dist']
                            agent['excess_time'] = agent['current_flooded_time_hr'] - agent['current_baseline_time_hr']

                            agent['path'] = flooded_path_coords
                            agent['DistanceToTarget'] = agent['flooded_dist']
                            agent['trip_active'] = True
                            agent['trip_departure_step'] = t
                            agent['trip_departure_label'] = time_label
                            agent['has_reached_target'] = False

                            total_excess_dist_t += agent['excess_dist']
                            total_excess_time_t += agent['excess_time']
                            traveling_agents_t += 1
                    else:
                        # Trip already active, just count as travelling
                        traveling_agents_t += 1

                # Agent idle?
                if agent['activity'] == 'Idle' or agent['target'] is None:
                    idle_agents_t += 1

            # 5. MOVE AGENTS & CHECK ARRIVALS
            for i, agent in enumerate(agents):
                if agent.get('trip_active', False) and agent.get('path'):
                    distance_travelled_step = agent['speed'] * time_step_increment
                    remaining_distance_this_step = distance_travelled_step

                    while len(agent['path']) > 0 and remaining_distance_this_step > 0.001:
                        next_point = agent['path'][0]
                        R = 6371
                        lat1, lon1 = np.radians(agent['x']), np.radians(agent['y'])
                        lat2, lon2 = np.radians(next_point[0]), np.radians(next_point[1])
                        dlat, dlon = lat2 - lat1, lon2 - lon1
                        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                        distance_to_next = R * c

                        if distance_to_next <= 0.001:
                            agent['path'].pop(0)
                            continue

                        if distance_to_next <= remaining_distance_this_step:
                            agent['x'], agent['y'] = next_point
                            agent['path'].pop(0)
                            remaining_distance_this_step -= distance_to_next
                        else:
                            frac = remaining_distance_this_step / distance_to_next
                            new_x = agent['x'] + (next_point[0] - agent['x']) * frac
                            new_y = agent['y'] + (next_point[1] - agent['y']) * frac
                            agent['x'], agent['y'] = new_x, new_y
                            remaining_distance_this_step = 0

                    agent['DistanceToTarget'] -= distance_travelled_step
                    if agent['DistanceToTarget'] < 0:
                        agent['DistanceToTarget'] = 0

                    # Arrival condition
                    if agent['DistanceToTarget'] <= 0.001 or (not agent['path']):
                        agent['has_reached_target'] = True
                        agent['trip_active'] = False
                        agent['path'] = []

                        # realised travel time 
                        if agent.get('trip_departure_step') is not None:
                            realised_hr = (t - agent['trip_departure_step'] + 1) * time_step_increment
                            completed_trips.append({
                                'departure_step': agent['trip_departure_step'],
                                'arrival_step': t,
                                'departure_time_label': agent['trip_departure_label'],
                                'realised_time_hr': realised_hr,
                                'baseline_time_hr_est': agent.get('current_baseline_time_hr', 0.0),
                                'flooded_time_hr_est': agent.get('current_flooded_time_hr', 0.0),
                                'origin_activity': agent.get('previous_activity', 'Home'),
                                'dest_activity': agent.get('activity', None),
                        
                                # >>> LTDS-inspired attributes <<<
                                'age': agent.get('age'),
                                'gender': agent.get('gender'),
                                'employment_status': agent.get('employment_status'),
                                'travel_mode': agent.get('travelMode'),
                                'demographic_modifier': agent.get('demographic_modifier'),
                                'RPI_departure': agent.get('RPI', 0.0),
                            })


            # ===================== Smooth Unserved -> Idle after 18:00, all by 23:00 =====================
            now_h = current_hour + current_minute / 60.0
            
            if now_h >= 18.0:
                # progress from 0 at 18:00 to 1 at 23:00
                progress = min(1.0, max(0.0, (now_h - 18.0) / (23.0 - 18.0)))
            
                p_give_up = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))  # 0..1
            
                for ag in agents:
                    if ag.get("unserved_state", False):
                        # At 23:00+, force everyone to Idle
                        if now_h >= 23.0:
                            force_idle = True
                        else:
                            force_idle = (random.random() < p_give_up)
            
                        if force_idle:
                            ag["unserved_state"] = False   
                            ag["unserved"] = False         
            
                            ag["activity"] = "Idle"
                            ag["target"] = None
                            ag["trip_active"] = False
                            ag["path"] = []
                            ag["DistanceToTarget"] = 0.0
                            ag["waiting_to_depart"] = False
                            ag["staying"] = False
                            ag["stay_until"] = None



            # ===================== STATE ACCOUNTING  =====================
            
            state_idle = 0
            state_travelling = 0   # ACTIVE = travelling + waiting + staying
            state_unserved = 0
            #state_waiting = 0
            #state_staying = 0
            
            for ag in agents:
            

                if ag.get("trip_active", False) or ag.get("waiting_to_depart", False) or ag.get("staying", False):
                    state_travelling += 1    
            
                   
                elif ag.get("unserved_state", False):
                    state_unserved += 1

    
                else:
                    state_idle += 1
            
            
            # ---- Save counts (scaled) ----
            metrics["state_idle_count"].append(state_idle * scaling_factor)
            metrics["state_travelling_count"].append(state_travelling * scaling_factor)
            metrics["state_unserved_count"].append(state_unserved * scaling_factor)

            # ---- Save shares ----
            N = len(agents)
            metrics["idle_agents_share"].append(state_idle / N)
            metrics["state_travelling_share"].append(state_travelling / N)
            metrics["unserved_agents_share_state"].append(state_unserved / N)


            metrics.setdefault("state_sum_check", []).append((state_idle + state_travelling + state_unserved) / N)
        
            # Save counts-----------------------------------------------------
            metrics['total_excess_dist_km'].append(total_excess_dist_t)
            metrics['total_excess_time_hours'].append(total_excess_time_t)
            metrics["total_agents"].append(N_total* scaling_factor)
            metrics['traveling_agent_count'].append(traveling_agents_t * scaling_factor)
            metrics["attempting_agents_t"].append(attempting_agents_t* scaling_factor)
            metrics["departures_by_step"].append(departures_this_step * scaling_factor)
            
            # Save shares------------------------------------------------------
            
            agent_count = len(agents) if len(agents) > 0 else 1

            metrics['unserved_agents_share'].append(unserved_agents_t / max(1, agent_count))
            metrics['unserved_agents_share_attempting'].append(unserved_agents_t / max(1, attempting_agents_t))

            
            metrics['travelling_agents_share'].append(traveling_agents_t / agent_count)


            if traveling_agents_t > 0:
                avg_excess_time_mins = (total_excess_time_t * 60) / traveling_agents_t
            else:
                avg_excess_time_mins = 0.0
            metrics['average_excess_time_mins'].append(avg_excess_time_mins)

            # 7. VISUALISATION
            ax.clear()
            #ax.set_title("Road Closure Scenario")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(-0.36, -0.29)
            ax.set_ylim(51.37, 51.44)
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.5)
                sp.set_edgecolor('black')

            try:
                E, N, D = hec.get_depth_field(t_idx, area=HEC_FLOW_AREAS[0])

                lon, lat = to_wgs84.transform(E, N)
                flood_mask = D > 0.01
                lon_f = lon[flood_mask]
                lat_f = lat[flood_mask]
                D_f = D[flood_mask]
                sc = ax.scatter(lon_f, lat_f, c=D_f, s=3, cmap='Blues', vmin=0.1, vmax=6.0,
                                alpha=0.9, zorder=1)
                cbar.mappable.set_array(D_f)
            except Exception:
                pass

            try:
                edges_gdf.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.7, zorder=2)
            
                if ENABLE_ROAD_CLOSURES and len(flooded_edge_indices) > 0:
                    closed_mask = edges_gdf.index.isin(flooded_edge_indices)
                    edges_gdf[closed_mask].plot(ax=ax, linewidth=3, edgecolor='purple',
                                                alpha=0.8, zorder=3)
            except Exception:
                pass


            time_text = f'Time: {current_hour:02d}:{current_minute:02d}'
            ax.text(0.02, 0.98, time_text, transform=ax.transAxes, fontsize=12,
                    fontweight='bold', color='black', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=1),
                    zorder=20)

            for agent in agents:
                x_lat, y_lon = agent['x'], agent['y']
        
                if agent.get("unserved", False) or agent.get("unserved_state", False):
                    color = "purple"
                elif agent.get("trip_active", False):
                    color = "blue"
                else:
                    color = "grey"
                    
                ax.scatter(y_lon, x_lat, s=30, color=color, edgecolor='black',
                           alpha=0.9, zorder=10)

            handles = [
                mpatches.Patch(color='#08306B', label='River channel'),
                mpatches.Patch(color='#73B2D8', label='Flooded area'),
                plt.Line2D([0], [0], color='gray', lw=1, label='Main road'),
                plt.Line2D([0], [0], marker='o', color='w', label='Travelling',
                           markerfacecolor='blue', markeredgecolor='black', markersize=8),
                plt.Line2D([0], [0], marker='o', color='w', label='Idle',
                           markerfacecolor='grey', markeredgecolor='black', markersize=8),
                plt.Line2D([0], [0], marker='o', color='w', label='Failed trip',
                           markerfacecolor='purple', markeredgecolor='black', markersize=8)
            ]
            
            if ENABLE_ROAD_CLOSURES:
                handles.insert(2, plt.Line2D([0], [0], color='purple', lw=5, label='Flooded road'))

            leg = ax.legend(handles=handles, loc='lower right', fontsize='small',
                            frameon=True)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_alpha(1.0)
            leg.get_frame().set_linewidth(1.5)
            leg.set_zorder(50)

            writer.grab_frame()
            frame_filename = os.path.join(
                frames_dir,
                f"frame_{t:02d}_{current_hour:02d}{current_minute:02d}.png"
            )
            fig.savefig(frame_filename, dpi=150, bbox_inches='tight',
                        pad_inches=0, transparent=True)

            if t % 5 == 0:
                print(f"  ... saved frame {t}/{total_time_steps}")

            agents_over_time.append(copy.deepcopy(agents))

            time_of_day += time_step_increment
            if time_of_day >= 24:
                time_of_day = 0
  

    with open(f"agents_over_time_{scenario_name}.pkl", "wb") as f:
        pickle.dump(agents_over_time, f)
    print(f"[SAVE] All agent states saved to agents_over_time_{scenario_name}.pkl")

    plt.close(fig)
    print(f"--- Simulation {scenario_name} Finished. Video saved to {video_file_name} ---")
    print(f"--- All {total_time_steps} frames saved to '{frames_dir}/' directory. ---")
    
    
    # compute RRI matrices + plot heatmaps
    # ---------------------------------------------------------
    RRI_age  = compute_rri_active_matrix(E_age,  A_age,  min_active_total=10, min_active_group=3)
    RRI_emp  = compute_rri_active_matrix(E_emp,  A_emp,  min_active_total=10, min_active_group=3)
    RRI_mode = compute_rri_active_matrix(E_mode, A_mode, min_active_total=10, min_active_group=3)
    
    # Create output folder
    out_dir = os.path.join("outputs", f"{day_type}_heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving heatmaps to: {out_dir}")
    
    
    START_HOUR = 6
    
    # ---------------------------------------------------------
    # Exposure heatmaps (absolute counts) – plotted 06:00 → 06:00
    # ---------------------------------------------------------
    plot_heatmap(E_age, AGE_GROUPS, time_labels,
                 #title=f"Exposure by Age Group",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_age_{day_type}.png"),
                 start_hour=START_HOUR)
    
    plot_heatmap(E_emp, EMP_GROUPS, time_labels,
                 #title=f"Exposure by Employment Status",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_employment_{day_type}.png"),
                 start_hour=START_HOUR)
    
    plot_heatmap(E_mode, MODE_GROUPS, time_labels,
                 #title=f"Exposure by Travel Mode",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_mode_{day_type}.png"),
                 start_hour=START_HOUR)
    
    
    # ---------------------------------------------------------
    # RRI heatmaps (relative risk) – plotted 06:00 → 06:00
    # ---------------------------------------------------------
    plot_heatmap(RRI_age, AGE_GROUPS, time_labels,
                 #title=f"RRI by Age Group",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_age_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_age))),
                 start_hour=START_HOUR)
    
    plot_heatmap(RRI_emp, EMP_GROUPS, time_labels,
                 #title=f"RRI by Employment Status",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_employment_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_emp))),
                 start_hour=START_HOUR)
    
    plot_heatmap(RRI_mode, MODE_GROUPS, time_labels,
                 #title=f"RRI by Travel Mode",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_mode_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_mode))),
                 start_hour=START_HOUR)
    
    print("Heatmaps successfully saved (plotted 06:00 → 06:00).")

    return metrics, agents_over_time, completed_trips



# =============================================================================
# -----------------------------<< MAIN EXECUTION>> ----------------------------
# =============================================================================

# Weekday scenario
results_weekday, agents_weekday, completed_trips = run_simulation(
    scenario_name="Road_Closure_Weekday",
    day_type="weekday"
)

#scenario_name="Road_Closure_Weekend",
     #day_type="weekend"
 #)

# Collect scenario results for saving
all_scenario_results = {
    "Road_Closure_Weekday": results_weekday,
     #"Road_Closure_Weekend": results_weekend,   # For weekend
}


trips_df = pd.DataFrame(completed_trips)

if trips_df.empty:
    print("No completed trips to plot.")
else:
    # Convert hours -> minutes
    trips_df["realised_time_min"] = trips_df["realised_time_hr"] * 60.0
    trips_df["baseline_time_min_est"] = trips_df["baseline_time_hr_est"] * 60.0
    trips_df["flooded_time_min_est"]  = trips_df["flooded_time_hr_est"] * 60.0
    trips_df["excess_time_min"] = (trips_df["flooded_time_hr_est"] - trips_df["baseline_time_hr_est"]) * 60.0

    trips_df["dep_hour"] = trips_df["departure_time_label"].str.slice(0, 2).astype(int)

# ==========================================================================================================================================
# --------------------------------------------------<< FINAL PLOTTING & DATA SAVING >>----------------------------------------------------
# =========================================================================================================================================

print("\n--- Generating Final Plots ---")

plt.style.use('seaborn-v0_8-darkgrid')

time_steps = np.arange(total_time_steps)

time_labels = [
    (DISPLAY_START_CLOCK + datetime.timedelta(minutes=STEP_MINUTES * i)).strftime('%H:%M')
    for i in range(total_time_steps)
]

def sl(x):
    """No slicing. Keep full 24h series, just relabeled in plots."""
    return np.asarray(x)

tick_idx = np.arange(0, total_time_steps, 2)  # hourly ticks for 30-min step



# =============================
# SNAPSHOTS for ALL timesteps
# =============================
snap_dir = "snapshots_all_steps_RPI_or_depth_Weekday"
os.makedirs(snap_dir, exist_ok=True)

# Loop all steps
for t in range(total_time_steps):
    t_idx = hec_indices[t]                
    tstr = time_labels[t]                

    out_png = os.path.join(
        snap_dir,
        f"snapshot_{t:03d}_{tstr.replace(':','')}.png"
    )

    plot_agents_snapshot(
        agents_t=agents_weekday[t] if isinstance(agents_weekday, list) else agents_weekday[t],
        # agents_t=agents_over_time[t],
        hec=hec,
        t_idx=t_idx,
        title_time_str=tstr,
        edges_gdf=edges_gdf,
        out_png=out_png,
        hec_area=HEC_FLOW_AREAS[0],
        rpi_thr=0.8,
        depth_thr=0.30
    )

    if t % 5 == 0:
        print(f"[SNAPSHOT] saved {t}/{total_time_steps}: {out_png}")



# ===================== 3.1 Flood exposure by age group =======================
age_colors = {
    'Children': 'tab:blue',
    'Adults': 'tab:orange',
    'Seniors': 'tab:green'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series: number of agents in flooded area by age ---
for age in age_types:
    series = sl(results_weekday['flooded_by_age'][age])
    ax_line.plot(
        time_steps,
        series,
        label=age,
        linewidth=2,
        marker='o',
        markersize=3,
        color=age_colors.get(age, None)
    )

ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')
ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Age Group")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart: total exposure share by age (over whole simulation) ---
age_totals = [sum(sl(results_weekday['flooded_by_age'][age])) for age in age_types]
total_exposed = sum(age_totals)

ax_pie.axis('equal')
if total_exposed > 0:
    wedges, texts, autotexts = ax_pie.pie(
        age_totals,
        labels=age_types,
        autopct='%1.1f%%',
        startangle=90,
        colors=[age_colors[a] for a in age_types],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Age Group")

plt.tight_layout()
plt.savefig("plot_3_1_flood_exposure_age_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()


# ================= 3.2 Flood exposure by employment status ====================
emp_colors = {
    'Employed': 'tab:blue',
    'Unemployed': 'tab:red',
    'Student': 'tab:green'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series ---
for emp in employment_status_types:
    series = sl(results_weekday['flooded_by_emp'][emp])
    ax_line.plot(
        time_steps,
        series,
        label=emp,
        linewidth=2,
        marker='o',
        markersize=3,
        color=emp_colors.get(emp, None)
    )

ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')

ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Employment Status")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart ---
emp_totals = [sum(sl(results_weekday['flooded_by_emp'][emp])) for emp in employment_status_types]
total_exposed_emp = sum(emp_totals)

ax_pie.axis('equal')
if total_exposed_emp > 0:
    wedges, texts, autotexts = ax_pie.pie(
        emp_totals,
        labels=employment_status_types,
        autopct='%1.1f%%',
        startangle=90,
        colors=[emp_colors[e] for e in employment_status_types],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Employment Status")

plt.tight_layout()
plt.savefig("plot_3_2_flood_exposure_employment_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()


# =================== 3.3 Flood exposure by travel mode =======================
mode_colors = {
    'Walkers': 'tab:green',
    'Cyclists': 'tab:cyan',
    'PTP': 'tab:purple',
    'Drivers': 'tab:orange'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series ---
for mode in travel_modes:
    series = sl(results_weekday['flooded_by_mode'][mode])
    ax_line.plot(
        time_steps,
        series,
        label=mode.capitalize(),
        linewidth=2,
        marker='o',
        markersize=3,
        color=mode_colors.get(mode, None)
    )
ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')

ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Travel Mode")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart ---
mode_totals = [sum(sl(results_weekday['flooded_by_mode'][mode])) for mode in travel_modes]
total_exposed_mode = sum(mode_totals)

ax_pie.axis('equal')
if total_exposed_mode > 0:
    wedges, texts, autotexts = ax_pie.pie(
        mode_totals,
        labels=[m.capitalize() for m in travel_modes],
        autopct='%1.1f%%',
        startangle=90,
        colors=[mode_colors[m] for m in travel_modes],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Travel Mode")

plt.tight_layout()
plt.savefig("plot_3_3_flood_exposure_mode_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()


#--------------- Number of travelling, unserved, idle, and percentage of the closed road agent--------------


# ---------------- Prepare data ----------------
time_steps = np.arange(len(results_weekday["total_agents"]))

total_agents = np.array(results_weekday["total_agents"])
travelling = np.array(results_weekday["state_travelling_count"])
idle = np.array(results_weekday["state_idle_count"])
unserved = np.array(results_weekday["state_unserved_count"])
#waiting = np.array(results_weekday["state_waiting_count"])
#staying = np.array(results_weekday["state_staying_count"])


# Closed roads percentage (already stored as % in your code)
closed_roads_pct = np.array(results_weekday["closed_roads_share"])

# ---------------- Plot settings ----------------
bar_width = 0.13

fig, ax1 = plt.subplots(figsize=(14, 6))

# >>> Make background white
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ---------------- Bars: agent states ----------------
ax1.bar(time_steps - 2.5 * bar_width, total_agents, width=bar_width, label="Total agents", color="lightgray")
ax1.bar(time_steps - 1.5 * bar_width, travelling,   width=bar_width, label="Travelling agents", color="tab:blue")
ax1.bar(time_steps - 0.5 * bar_width, unserved,     width=bar_width, label="Failed trips", color="tab:purple")
#ax1.bar(time_steps + 0.5 * bar_width, waiting,      width=bar_width, label="Waiting agents", color="tab:green")
#ax1.bar(time_steps + 1.5 * bar_width, staying,      width=bar_width, label="Staying agents", color="tab:brown")
ax1.bar(time_steps + 2.5 * bar_width, idle,         width=bar_width, label="Idle agents", color="tab:orange")


ax1.set_xlabel("Time of day")
ax1.set_ylabel("Number of people")
ax1.set_ylim(0, total_agents.max() * 1.1)

# ---------------- Secondary axis: closed roads (%) ----------------
ax2 = ax1.twinx()

ax2.plot(time_steps, closed_roads_pct,
         color="red", marker="s", linestyle="--",
         linewidth=2, label="Closed roads")

ax2.set_ylabel("Closed roads")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0, max(5, closed_roads_pct.max() * 1.2))

# ---------------- X-axis ticks ----------------
ax1.set_xticks(tick_idx)
ax1.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha="right")

# ---------------- Grid & aesthetics ----------------
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

# ---------------- Legend ----------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend = ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    edgecolor="black"
)
legend.get_frame().set_linewidth(1.3)

# ---------------- Title ----------------
#plt.title("Agent States and Road Closures Over Time")

# ---------------- Save & show ----------------
output_file = "agent_states_vs_closed_roads_weekday_30min_18-02-26.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

#------------------------------------------------------------------------------

# --- ---------------------------Save Data ------------------------------------
print("\n--- Saving All Scenario Results ---")


base = {
    'time_step': list(time_steps),
    'time_of_day': list(time_labels),
    'people_at_risk': sl(results_weekday['people_at_risk']),
    'unserved_agents_share_state': sl(results_weekday['unserved_agents_share_state']),
    'travelling_agents_share': sl(results_weekday['travelling_agents_share']),
    'idle_agents_share': sl(results_weekday['idle_agents_share']),
    'traveling_agent_count': sl(results_weekday['traveling_agent_count']),
    'total_agents': sl(results_weekday['total_agents']),
    'state_idle_count': sl(results_weekday['state_idle_count']),
    'state_travelling_count': sl(results_weekday['state_travelling_count']),
    'state_unserved_count': sl(results_weekday['state_unserved_count']),
    'closed_roads_count': sl(results_weekday['closed_roads_count']),
    'closed_roads_share': sl(results_weekday['closed_roads_share']),
    'mean_rpi': sl(results_weekday['mean_rpi']),
    'median_rpi': sl(results_weekday['median_rpi']),
}
results_df = pd.DataFrame(base)

# --- flatten demographic dict-of-lists into columns ---
# Age
for age in age_types:
    results_df[f"flooded_by_age_{age}"] = sl(results_weekday['flooded_by_age'][age])

# Employment
for emp in employment_status_types:
    results_df[f"flooded_by_emp_{emp}"] = sl(results_weekday['flooded_by_emp'][emp])

# Mode
for mode in travel_modes:
    results_df[f"flooded_by_mode_{mode}"] = sl(results_weekday['flooded_by_mode'][mode])
    

results_df.to_csv("No_Closure_Weekday_results_highAgent_weekday_18-02-26_LTDSPop_ExposureScenario_UnserevdAgent_NewRPI.csv", index=False)

# --- Save Workspace ---
save_workspace(
    filename="ABM_HECRAS_No_Closure_Weekday_high agent_weekday_18-02-26_LTDSPop_ExposureScenario_Unserevd agents_NewRPI.pkl",
    scenario_results=all_scenario_results
)

print("\n--- Simulation Complete ---")



def RiskPerceptionIndex_HECRAS(agents, hec_reader, t_index,
                              alpha: float = ALPHA,
                              beta: float = BETA,
                              T: float = T_DEPTH,
                              k: float = K_SIG,
                              day_type: str = "weekday"):
    """
    NEW RPI:
      RPI_{i,t} = Clamp( ((alpha*f(D_{i,t}) + beta*C_t) * M_i), 0, 1 )
    where:
      f(D) is sigmoid depth perception
      C_t is official warning (0..1)
      M_i is demographic amplifier (>=1)
    """
    # --- points for HEC sampling ---
    pts_EN = []
    for ag in agents:
        lat = ag['x']  # Agent stores lat in x
        lon = ag['y']  # Agent stores lon in y
        E, N = wgs84_to_bng(lon, lat)
        pts_EN.append((E, N))

    depths, wses = hec_reader.get_depth_wse_at_points(pts_EN, t_index, area=HEC_FLOW_AREAS[0])


    # --- official communication value for this timestep ---
    C_t = get_official_warning_C_t(hec_reader, t_index)

    num_agents_at_risk = 0

    for i, agent in enumerate(agents):
        depth = max(0.0, float(depths[i]))
        agent['water_depth'] = depth

        if depth > CLOSURE_DEPTH:
            num_agents_at_risk += 1

        # 1) Nonlinear depth perception
        fD = depth_perception_sigmoid(depth, T=T, k=k)

        # 2) Demographic amplifier
        M_i = compute_M_i(agent)
        agent["M_i"] = M_i  

        # 3) Weighted hazard + warning, then amplify, then clamp
        raw = (alpha * fD + beta * C_t) * M_i
        agent['RPI_individual'] = raw  
        agent['RPI'] = clamp01(raw)

       
        agent['f_depth'] = fD
        agent['C_t'] = C_t

    return num_agents_at_risk, agents


def save_workspace(filename, scenario_results):
    """
    Save only the scenario_results dict to disk.
    This avoids iterating over globals() and associated issues.
    """
    workspace = {
        "scenario_results": scenario_results
    }

    with open(filename, "wb") as f:
        pickle.dump(workspace, f)

    print(f" Workspace saved to {filename}")


#------------------------------------------------------------------------------------------------------------
#-------------------------------<< Initialisation Phase>>----------------------------------------------------
#------------------------------------------------------------------------------------------------------------
print("Loading static data (Population, Buildings, Roads)...")
try:
    population_data = gpd.read_file('Agents_in_FigureBBox.shp')
    building_data = gpd.read_file('Buildings_in_FigureBBox.shp')
    
    # --------------------- Load density points as population seeds -----------------------------

    
    # --------------------- Prepare residential buildings (homes) -----------------------------
    # building_data already loaded from Buildings_in_FigureBBox.shp
    homes_bld = building_data[building_data["use"] == "RESIDENTIAL ONLY"].copy()
    
    # Ensure both are in the SAME projected CRS for nearest-neighbour (BNG)
    population_bng = population_data.to_crs(BNG)
    homes_bng = homes_bld.to_crs(BNG)
    
    # Keep only geometry + id fields
    population_bng = population_bng[["geometry"]].copy()
    homes_bng = homes_bng[["geometry"]].copy()
    
    # Find nearest residential building for each population point
    # (requires GeoPandas >= 0.10)
    pop_to_home = gpd.sjoin_nearest(
        population_bng,
        homes_bng,
        how="left",
        distance_col="dist_to_home_m"
    )
    
    # Convert matched home geometry back to WGS84 (lat/lon)
    # store as arrays aligned with population points
    home_geom = homes_bng.loc[pop_to_home["index_right"]].geometry.reset_index(drop=True)
    home_cent = home_geom.centroid
    
    # centroid coords in BNG -> WGS84
    home_lon, home_lat = to_wgs84.transform(home_cent.x.values, home_cent.y.values)
    
    # Fallback: if any population point had no matched home, use the point itself
    pop_cent = population_bng.geometry.centroid
    pop_lon, pop_lat = to_wgs84.transform(pop_cent.x.values, pop_cent.y.values)
    
    home_lat = np.where(np.isnan(home_lat), pop_lat, home_lat)
    home_lon = np.where(np.isnan(home_lon), pop_lon, home_lon)
    
    # Save arrays for agent creation (same order as population_data rows)
    HOME_LAT_FOR_POPPOINT = home_lat
    HOME_LON_FOR_POPPOINT = home_lon


    # --------------------- Load Road Network -----------------------------
    center_point = (51.41, -0.33)
    radius = 5000
    custom_filter = '["highway"~"motorway|trunk|primary|secondary"]'
    G_base = ox.graph_from_point(center_point, dist=radius,
                                 custom_filter=custom_filter, simplify=True)

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_base)
    edges_bng = edges_gdf.to_crs(BNG)  # For flood checks

    if edges_gdf.crs.is_geographic:
        edges_utm = edges_gdf.to_crs("EPSG:32630")
        total_road_length_meters = edges_utm.geometry.length.sum()
    else:
        total_road_length_meters = edges_gdf.geometry.length.sum()

    CELL_LENGTH_METERS = 7.5
    TOTAL_ROAD_CELLS = int(total_road_length_meters / CELL_LENGTH_METERS)
    print(f"Total road network length: {total_road_length_meters:.2f} meters")
    print(f"Equivalent NaSch cells: {TOTAL_ROAD_CELLS}")
    
    

    edges_bng2 = edges_bng.copy()
    
    def is_main_highway(hw):
        # hw can be string or list in OSMnx
        if isinstance(hw, (list, tuple, set)):
            hw = hw[0] if len(hw) else None
        return hw in {"motorway", "trunk", "primary", "secondary"}
    
    main_edges_bng = edges_bng2[edges_bng2["highway"].apply(is_main_highway)].copy()
    
    def sample_line_points(line, n_samples=10):
        ds = np.linspace(0, line.length, n_samples)
        return [(line.interpolate(d).x, line.interpolate(d).y) for d in ds]
    
    # This will be filled EACH timestep from the currently closed edges
    road_pts_EN = []                 # sampled points on closed roads (BNG)
    road_pts_edge_ids = []           # mapping each point -> edge id (u,v,key)
    


##-----------------------------------------------------------------------------

    # Initialize categories
    categories = {
        'Home': {'X': [], 'Y': []},
        'School': {'X': [], 'Y': []},
        'Work': {'X': [], 'Y': []},
        'Recreation': {'X': [], 'Y': []}
    }

    for _, row in building_data.iterrows():
        building_type = row['use']
        xx = row['X_Coordina']
        yy = row['Y_Coordina']
        lat, lon = transform_coordinates(xx, yy)
        if lat is None:
            continue

        if building_type == 'RESIDENTIAL ONLY':
            categories['Home']['X'].append(lat)
            categories['Home']['Y'].append(lon)
        elif building_type == 'COMMUNITY - EDUCATIONAL':
            categories['School']['X'].append(lat)
            categories['School']['Y'].append(lon)
        elif building_type in [
            'RETAIL WITH OFFICE/RESIDENTIAL ABOVE', 'OFFICE ONLY',
            'COMMUNITY - GOVERNMENTAL (CENTRAL AND LOCAL)',
            'GENERAL COMMERCIAL - MIXED USE', 'RETAIL ONLY',
            'RETAIL - WITH MORE RECENT EXTENSIONS OF DIFFERENT TYPE CONSTRUCTION/AGE'
        ]:
            categories['Work']['X'].append(lat)
            categories['Work']['Y'].append(lon)
        elif building_type in ['RECREATION AND LEISURE', 'COMMUNITY - RELIGIOUS']:
            categories['Recreation']['X'].append(lat)
            categories['Recreation']['Y'].append(lon)

    # Entry points for external agents
    entry_points = [
        {"name": "North West Entrance", "x": 51.4480, "y": -0.3935},
        {"name": "North Entrance", "x": 51.4505, "y": -0.3400},
        {"name": "North East Entrance", "x": 51.4485, "y": -0.2780},
        {"name": "East Entrance", "x": 51.4250, "y": -0.2650},
        {"name": "South East Entrance", "x": 51.3800, "y": -0.2700},
        {"name": "South Entrance", "x": 51.3700, "y": -0.3100},
        {"name": "South West Entrance", "x": 51.3700, "y": -0.3700},
        {"name": "West Entrance", "x": 51.4000, "y": -0.3950},
        {"name": "A3 Entrance", "x": 51.4150, "y": -0.3850},
        {"name": "M4 Junction", "x": 51.4300, "y": -0.3600},
        {"name": "Twickenham Entrance", "x": 51.4400, "y": -0.3450},
        {"name": "Wandsworth Entrance", "x": 51.4450, "y": -0.3200},
        {"name": "Battersea Entrance", "x": 51.4700, "y": -0.2900},
        {"name": "Richmond Entrance", "x": 51.4600, "y": -0.3050},
        {"name": "Tooting Entrance", "x": 51.4200, "y": -0.3900},
        {"name": "Sutton Entrance", "x": 51.3600, "y": -0.3200},
        {"name": "Croydon Entrance", "x": 51.3700, "y": -0.2800},
        {"name": "Lewisham Entrance", "x": 51.3850, "y": -0.2600},
        {"name": "Fulham Entrance", "x": 51.4700, "y": -0.2200},
        {"name": "Epsom Entrance", "x": 51.3400, "y": -0.3200}
    ]

    hec = HECRAS2DReader(
        HEC_HDF_PATH,
        flow_areas=HEC_FLOW_AREAS,
        prefer_native_depth=PREFER_NATIVE_DEPTH
    )
    print(f"[HEC] Loaded {len(hec.areas)} 2D area(s). HDF timesteps = {len(hec.times)}")
    
    # ------------------ filter HEC timesteps to a 24h window ------------------
    end_clock = START_CLOCK + pd.Timedelta(hours=WINDOW_HOURS)
    mask = (hec.times >= START_CLOCK) & (hec.times < end_clock)
    hec_indices = np.where(mask)[0].tolist()
    
    if len(hec_indices) == 0:
        raise RuntimeError(f"No HEC timesteps found in window {START_CLOCK} to {end_clock}")
    
    used_times = hec.times[hec_indices]
    total_time_steps = len(hec_indices)     # <-- IMPORTANT: ABM loop length becomes HEC-driven
    
    print(f"[HEC] Window steps found: {total_time_steps}")
    print(f"[HEC] Window range: {used_times.min()} -> {used_times.max()}")
    
    if len(used_times) >= 2:
        dt_min = np.median(np.diff(used_times.values).astype("timedelta64[m]").astype(int))
        print(f"[HEC] Median timestep in window: {dt_min} minutes")


except Exception as e:
    print(f"!!! CRITICAL ERROR during Initialization: {e} !!!")
    print("Check all input file paths (HDF, SHP) and network connectivity.")
    raise


#------------------------------------------------------------------------------------------------------------
#-------------------------------<< Agent Creation >>---------------------------------------------------------
#------------------------------------------------------------------------------------------------------------


def create_agents():
    """Creates a new list of agents with initial properties + daily-trip flags."""
    agents = []

    # Local residents from population_data
    for idx, row in population_data.iterrows():
        # Use assigned home building lat/lon (not the random density point)
        x = float(HOME_LAT_FOR_POPPOINT[idx])   # agent lat
        y = float(HOME_LON_FOR_POPPOINT[idx])   # agent lon
        
        age = sample_age()
        
        agents.append(
            {
                "x": x,
                "y": y,
                "home_lat": x,
                "home_lon": y,
                "activity": "Home",
                "target": None,
                "speed": None,
                "age": age,
                "travelMode": sample_travel_mode(day_type="weekday", age=age),
                "path": None,
                "stay_until": None,
                "traffic_factor": None,
                "agent_type": "Local",
                "movement_history": None,
                "DistanceToTarget": 0.0,
                "has_reached_target": False,
                "flood_risk": None,
                "gender": sample_gender(),
                "employment_status": sample_employment(day_type="weekday"),
                "I_media": random.uniform(0.1, 0.3),
                "demographic_modifier": None,
                "location_risk_factor": None,
                "RPI": None,
                "trust_official": random.uniform(0.5, 1.0),
                "cancelled_trip": False,
                "baseline_dist": 0.0,
                "flooded_dist": 0.0,
                "excess_dist": 0.0,
                "excess_time": 0.0,
                "unserved": False,
                "neighbor_mean_rpi_t_minus_1": 0.0,
                # LTDS trip planning
                "trips_generated": False,
                "daily_trips": [],
                "trip_active": False,
                "trip_departure_step": None,
                "trip_departure_label": None,
                "current_baseline_time_hr": 0.0,
                "current_flooded_time_hr": 0.0,
                "previous_activity": "Home",
                "waiting_to_depart": False,
                "staying": False,
                "cooldown_steps": 0,              # how many future steps to force idle
                "resume_activity": None,          # activity to restore after cooldown
                "resume_target": None,            # target to restore after cooldown
                "unserved_state": False,       # persistent (used for graphs)
                "unserved_since_step": None,



            }
        )


    # External agents from entry_points
    for entry in entry_points:
        for _ in range(2):
            age = sample_age()
            agents.append(
                {
                    "x": entry["x"],
                    "y": entry["y"],
                    "home_lat": entry["x"],
                    "home_lon": entry["y"],

                    "activity": "Home",
                    "target": None,
                    "speed": None,
                    "age": age,
                    "travelMode": sample_travel_mode(day_type="weekday", age=age),
                    "path": None,
                    "movement_history": [],
                    "stay_until": None,
                    "DistanceToTarget": 0.0,
                    "has_reached_target": False,
                    "origin_entry": entry["name"],
                    "agent_type": random.choice(["visitor", "commuter"]),
                    "flood_risk": None,
                    "gender": sample_gender(),
                    "employment_status": sample_employment(day_type="weekday"),
                    "I_media": random.uniform(0.1, 0.3),
                    "demographic_modifier": None,
                    "location_risk_factor": None,
                    "RPI": None,
                    "trust_official": random.uniform(0.5, 1.0),
                    "cancelled_trip": False,
                    "baseline_dist": 0.0,
                    "flooded_dist": 0.0,
                    "excess_dist": 0.0,
                    "excess_time": 0.0,
                    "unserved": False,
                    "neighbor_mean_rpi_t_minus_1": 0.0,
                    "trips_generated": False,
                    "daily_trips": [],
                    "trip_active": False,
                    "trip_departure_step": None,
                    "trip_departure_label": None,
                    "current_baseline_time_hr": 0.0,
                    "current_flooded_time_hr": 0.0,
                    "previous_activity": "Home",
                    "waiting_to_depart": False,
                    "staying": False,
                    "cooldown_steps": 0,              # how many future steps to force idle
                    "resume_activity": None,          # activity to restore after cooldown
                    "resume_target": None,            # target to restore after cooldown
                    "unserved_state": False,          # persistent (used for graphs)
                    "unserved_since_step": None,     



                }
            )


        for agent in agents:
            # If Child: force Student + forbid driving
            if agent["age"] == "Children":
                agent["employment_status"] = "Student"
        
                # forbid driving for children
                if agent["travelMode"] == "Drivers":
                    agent["travelMode"] = np.random.choice(
                        ["Walkers", "Cyclists", "PTP"],
                        p=[0.55, 0.10, 0.35]   
                    )
        
            # speed after fixing mode
            agent["speed"] = base_speeds[agent["travelMode"]]
        
            agent["demographic_modifier"] = compute_M_i(agent)



    # Social network
    N = len(agents)
    G_social = nx.watts_strogatz_graph(N, k_neighbors, p_rewire)
    for i, agent in enumerate(agents):
        nbrs = list(G_social.neighbors(i))
        agent["neighbors"] = nbrs
        agent["social_weights"] = {j: 1 / len(nbrs) for j in nbrs} if nbrs else {}

    # Initial RPI
    _, agents = RiskPerceptionIndex_HECRAS(agents, hec, 0)

    print(f"Created {len(agents)} total agents.")
    return agents


#--------------------------------------------------------------------------------------------------------------
#---------------<< Activity generation & update (LTDS-based) >>-----------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def update_agent_activity(agent, current_hour, current_minute, day_type: str):
    """
    Convert planned trips into actual activities (Home -> Work, etc.)
    and also handle return trips when stay_until is reached.
    """
    # ensure the agent has a daily plan
    generate_daily_trips_for_agent(agent, day_type)

    now_h = current_hour + current_minute / 60.0
    current_step = int(round(now_h / STEP_HOURS)) 
    
    agent["waiting_to_depart"] = False
    agent["staying"] = False
    
    # If agent is currently in an activity (has stay_until), they are "staying"
    if agent.get("stay_until") is not None:
        sh, sm = agent["stay_until"]
        if not time_reached(current_hour, current_minute, sh, sm):
            agent["staying"] = True
            return  # nothing else should start while staying


    # 1) Check if any *new* trip should start now
    for trip in agent["daily_trips"]:
        if trip["assigned"] or trip["completed"]:
            continue
        
        # Find first upcoming trip that isn't assigned/completed
        if current_step < trip["dep_step"] and current_step >= (trip["dep_step"] - 1):
            agent["waiting_to_depart"] = True


        # >>> exact step trigger
        if current_step == trip["dep_step"]:
            purpose = trip["purpose"]
            rpi = agent.get("RPI", 0.0)
            
            if purpose == "Home" and trip.get("direction") == "return":
                agent["activity"] = "Home"
                
                agent["target"] = {"X": agent["home_lat"], "Y": agent["home_lon"]}  # stored at creation

                agent["has_reached_target"] = False
                agent["cancelled_trip"] = False
                agent["baseline_dist"] = 0.0
                agent["flooded_dist"] = 0.0
                agent["excess_dist"] = 0.0
                agent["excess_time"] = 0.0
                agent["trip_active"] = False
                agent["trip_departure_step"] = None
                agent["trip_departure_label"] = None
                
                trip["assigned"] = True
                break
                
            # RPI thresholds: cancel trip if risk too high
            if purpose in ["Work", "Education"] and rpi > 0.9:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue
            if purpose in ["Shopping", "Leisure", "Other"] and rpi > 0.8:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue

            # choose destination with realistic trip length
            dest_info = choose_destination_with_length_bin(agent, purpose)
            if dest_info is None:
                agent["activity"] = "Idle"
                agent["target"] = None
                agent["cancelled_trip"] = True
                trip["assigned"] = True
                trip["completed"] = True
                continue

            target, dist_km = dest_info
            agent["target"] = target

            # set activity label based on purpose
            if purpose == "Work":
                agent["activity"] = "Work"
            elif purpose == "Education":
                agent["activity"] = "School"
            elif purpose == "Shopping":
                agent["activity"] = "Shop"
            elif purpose == "Leisure":
                agent["activity"] = "Recreation"
            else:
                agent["activity"] = random.choice(["Recreation", "Shop"])

            # store origin activity for this trip
            agent["previous_activity"] = "Home" if agent["activity"] != "Home" else "Home"

            # length of stay: rough rule-of-thumb by purpose
            if agent["activity"] in ["Work", "School"]:
                stay_hours = random.uniform(3.0, 5.0)
            elif agent["activity"] == "Shop":
                stay_hours = random.uniform(0.5, 1.0)
            else:
                stay_hours = random.uniform(1.0, 2.0)

            end_time = now_h + stay_hours
            end_time = min(end_time, 23.9)
            stay_h = int(end_time)
            stay_m = int(round((end_time - stay_h) * 60))
            agent["stay_until"] = [stay_h, stay_m]

            # reset movement flags for new trip
            agent["has_reached_target"] = False
            agent["cancelled_trip"] = False
            agent["baseline_dist"] = 0.0
            agent["flooded_dist"] = 0.0
            agent["excess_dist"] = 0.0
            agent["excess_time"] = 0.0
            agent["trip_active"] = False   # will be set when path is found
            agent["trip_departure_step"] = None
            agent["trip_departure_label"] = None

            trip["assigned"] = True
            break  # only start one new trip per timestep

    # 2) Return trips: when stay_until reached at destination -> go Home (or Exit for visitors)
    if agent.get("stay_until") is not None:
        sh, sm = agent["stay_until"]
        if time_reached(current_hour, current_minute, sh, sm):
            agent["stay_until"] = None



def plot_agents_snapshot(
    agents_t,
    hec,
    t_idx,
    title_time_str,
    edges_gdf=None,
    out_png=None,
    hec_area=None,
    rpi_thr=0.8,
    depth_thr=0.30,
    lonlim=(-0.36, -0.29),
    latlim=(51.37, 51.44),
):
    """
    One snapshot map:
      - Flood depth scatter (HEC 2D)
      - Roads
      - Agents meeting: (RPI >= rpi_thr) OR (water_depth >= depth_thr)
      - Marker shape = travel mode
      - Color = age group
    """

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    #ax.set_title(f"Agents with RPI>{rpi_thr} or Water depth>{depth_thr} m", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(*lonlim)
    ax.set_ylim(*latlim)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_edgecolor('black')

    # ---------- Flood map ----------
    try:
        E, N, D = hec.get_depth_field(t_idx, area=hec_area)
        lon, lat = to_wgs84.transform(E, N)

        flood_mask = D > 0.01
        lon_f, lat_f, D_f = lon[flood_mask], lat[flood_mask], D[flood_mask]

        sc = ax.scatter(
            lon_f, lat_f, c=D_f, s=3,
            cmap="Blues", vmin=0.1, vmax=6.0,
            alpha=0.9, zorder=1
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Water Depth (m)")
        cbar.outline.set_linewidth(1.2)
        cbar.outline.set_edgecolor("black")
    except Exception as e:
        sc = None

    # ---------- Roads ----------
    if edges_gdf is not None:
        try:
            edges_gdf.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.7, zorder=2)
        except Exception:
            pass


    try:
        pts_EN = []
        for ag in agents_t:
            E_ag, N_ag = wgs84_to_bng(ag["y"], ag["x"])
            pts_EN.append((E_ag, N_ag))
        depths, _ = hec.get_depth_wse_at_points(pts_EN, t_idx, area=hec_area)

        for ag, d in zip(agents_t, depths):
            ag["water_depth"] = float(max(0.0, d))
    except Exception:
        # If fails, keep existing agent["water_depth"] if present
        pass

    # ---------- Filter agents of interest ----------
    selected = [
        ag for ag in agents_t
        if (ag.get("RPI", 0.0) >= rpi_thr) or (ag.get("water_depth", 0.0) >= depth_thr)
    ]

    # ---------- Styles ----------
    age_color = {
        "Children":  "tab:blue",
        "Adults":  "tab:orange",
        "Seniors": "tab:green"
    }
    mode_marker = {
        "Walkers": "o",
        "Cyclists": "^",
        "PTP": "s",
        "Drivers": "D"
    }

    # ---------- Plot agents (by mode to get correct legend) ----------
    for mode, mk in mode_marker.items():
        subset_mode = [ag for ag in selected if ag.get("travelMode") == mode]
        if not subset_mode:
            continue

        xs = [ag["y"] for ag in subset_mode]  # lon
        ys = [ag["x"] for ag in subset_mode]  # lat
        cs = [age_color.get(ag.get("age", "Adults"), "tab:orange") for ag in subset_mode]

        ax.scatter(
            xs, ys,
            s=55,
            marker=mk,
            c=cs,
            edgecolor="black",
            linewidths=0.8,
            alpha=0.95,
            zorder=10
        )

    # ---------- Time label ----------
    ax.text(
        0.02, 0.98, f"Time: {title_time_str}",
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=1.0),
        zorder=50
    )

    # ---------- Legends (Age + Travel Mode) ----------
    age_handles = [
        plt.Line2D([0],[0], marker='o', color='w', label='Children',
                   markerfacecolor=age_color["Children"], markeredgecolor="black", markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='Adults',
                   markerfacecolor=age_color["Adults"], markeredgecolor="black", markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='Seniors',
                   markerfacecolor=age_color["Seniors"], markeredgecolor="black", markersize=8),
    ]
    mode_handles = [
        plt.Line2D([0],[0], marker=mode_marker["Walkers"], color='w', label='Walkers',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["Cyclists"], color='w', label='Cyclists',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["PTP"], color='w', label='PTP',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
        plt.Line2D([0],[0], marker=mode_marker["Drivers"], color='w', label='Drivers',
                   markerfacecolor='black', markeredgecolor="black", markersize=7),
    ]

    leg1 = ax.legend(handles=age_handles, title="Age", loc="lower left",
                     frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")
    leg1.get_frame().set_linewidth(1.2)
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=mode_handles, title="Travel Mode", loc="lower right",
                     frameon=True, framealpha=1.0, edgecolor="black", facecolor="white")
    leg2.get_frame().set_linewidth(1.2)

    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

#--------------------------------------------------------------------------------------------------------------
#---------------<< Main Simulation Loop >>--------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

def run_simulation(scenario_name='scenario_name', day_type= 'day_type'):
    """
    Runs the full ABM simulation for the road closure scenario.
    Uses realised trip times with departure/arrival tracking.
    """
    print(f"\n--- Starting Simulation: {scenario_name} ---")

    agents = create_agents()
    time_of_day = 6.0  # Starting time (8:00 AM)

    agents_over_time = []
    completed_trips = []  # list of completed trips for post-analysis

    # --- Metric Lists ---
    metrics = {
        'total_agents':[],
        'agents_at_risk': [],
        'people_at_risk': [],
        'unserved_agents_share': [],
        'total_excess_dist_km': [],
        'total_excess_time_hours': [],
        'average_excess_time_mins': [],
        'traveling_agent_count': [],
        'idle_agents_share': [],
        'closed_roads_count': [],
        'closed_roads_share': [],   
        'unserved_agents_share_state': [],  
        'state_idle_count':[],
        "state_travelling_count":[],
        "state_unserved_count":[],
        "attempting_agents_t":[],
        "departures_by_step":[],
        'travelling_agents_share':[],
        'state_travelling_share':[],
        'state_sum_check':[],
    
        
        'attempting_agents_count': [],
        'unserved_agents_share_attempting': [],
        'unserved_agents_count_state': [],
        'mean_rpi': [],
        'median_rpi': [],   

    }
    
    metrics.setdefault("departures_by_step", [])

    
    # New: demographic exposure in flooded area (depth > CLOSURE_DEPTH)
    metrics['flooded_by_age'] = {age: [] for age in age_types}
    metrics['flooded_by_emp'] = {emp: [] for emp in employment_status_types}
    metrics['flooded_by_mode'] = {mode: [] for mode in travel_modes}

    video_file_name = f"HEC_ABM_high agent_Weekday_18-02-26_Unserved agents_NewRPI_{scenario_name}.mp4"

    # --- Setup Figure and Colorbar *before* the loop ---
    fig, ax = plt.subplots(figsize=(8, 8))
    #ax.set_title("Road Closure Scenario")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-0.36, -0.29)
    ax.set_ylim(51.37, 51.44)
    ax.set_axisbelow(True)
    ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.08, top=0.94)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.5)
        sp.set_edgecolor('black')

    dummy_scatter = ax.scatter([], [], c=[], s=3, cmap='Blues', vmin=0, vmax=6.0, alpha=0.9, zorder=1)
    cbar = fig.colorbar(dummy_scatter, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Water Depth (m)")
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')

    fig.tight_layout()

    frames_dir = f"frames_weekday_high agent_18-02-26_ExposureScenario_NewRPI_{scenario_name}"
    os.makedirs(frames_dir, exist_ok=True)
    print(f"Saving all frames to '{frames_dir}/' directory...")

    writer = FFMpegWriter(fps=2, metadata={'title': f'Agent Movement_weekday_high agent-18-02-26_ExposureScenario_NewRPI_{scenario_name}'})
    
    
    
    # ---------------------------------------------------------
    # Exposure tracking settings
    # ---------------------------------------------------------
    START_HOUR = 6
    START_MIN  = 0
    STEP_MINUTES = int(STEP_HOURS * 60)  
    N_STEPS = total_time_steps  
    
    AGE_GROUPS = ["Children", "Adults", "Seniors"]
    EMP_GROUPS = ["Employed", "Unemployed", "Student"]
    MODE_GROUPS = ["Walkers", "Cyclists", "PTP", "Drivers"]
    
    # Total population per group (computed once from ALL agents)
    def count_group_totals(agents, key, allowed_groups):
        counts = {g: 0 for g in allowed_groups}
        for a in agents:
            v = safe_norm(a.get(key, "Other"))
            if v in counts:
                counts[v] += 1
        return np.array([counts[g] for g in allowed_groups], dtype=float)
    
    N_age = count_group_totals(agents, "age", AGE_GROUPS)
    N_emp = count_group_totals(agents, "employment_status", EMP_GROUPS)
    N_mode = count_group_totals(agents, "travelMode", MODE_GROUPS)
    
    # Exposure count matrices (G,T)
    E_age  = np.zeros((len(AGE_GROUPS), N_STEPS), dtype=float)
    E_emp  = np.zeros((len(EMP_GROUPS), N_STEPS), dtype=float)
    E_mode = np.zeros((len(MODE_GROUPS), N_STEPS), dtype=float)
    
    # >>> ADD THIS HERE <<<
    A_age  = np.zeros((len(AGE_GROUPS), N_STEPS), dtype=float)
    A_emp  = np.zeros((len(EMP_GROUPS), N_STEPS), dtype=float)
    A_mode = np.zeros((len(MODE_GROUPS), N_STEPS), dtype=float)
    
    # Time labels for plotting
    time_labels = [step_to_clock_str(s, START_HOUR, START_MIN, STEP_MINUTES) for s in range(N_STEPS)]


    with writer.saving(fig, video_file_name, dpi=100):
        for t in range(total_time_steps):
            # ABM clock (08:00 -> 08:00+24h)
            current_hour = int(np.floor(time_of_day)) % 24
            current_minute = int(round((time_of_day - np.floor(time_of_day)) * 60)) % 60
            time_label = f"{current_hour:02d}:{current_minute:02d}"
            
            for ag in agents:
                ag["unserved"] = False



            # NEW: exact mapping to the filtered 24h HEC window
            t_idx = hec_indices[t]


            # 1. Update RPI based on HEC-RAS depth (Individual RPI)
            num_agents_at_risk, agents = RiskPerceptionIndex_HECRAS(agents, hec, t_idx, day_type=day_type)
            metrics['agents_at_risk'].append(num_agents_at_risk)
            metrics['people_at_risk'].append(num_agents_at_risk * scaling_factor)

            # 2. Social Network RPI Update
            new_rpis = {}
            for i, agent in enumerate(agents):
                individual_rpi = agent.get('RPI', 0)
                if agent['neighbors']:
                    neighbor_rpis = [agents[j].get('RPI', 0) for j in agent['neighbors']]
                    social_rpi = np.mean(neighbor_rpis) if neighbor_rpis else 0
                    updated_rpi = (lambda_social * individual_rpi) + ((1 - lambda_social) * social_rpi)
                    new_rpis[i] = min(1.0, updated_rpi)
                else:
                    new_rpis[i] = individual_rpi
            for i, agent in enumerate(agents):
                agent['RPI'] = new_rpis.get(i)
                
            metrics.setdefault("mean_rpi", []).append(
            float(np.mean([agent.get("RPI", 0.0) for agent in agents]))
            )
            metrics.setdefault("median_rpi", []).append(
                float(np.median([agent.get("RPI", 0.0) for agent in agents]))
)
        

            force_open_now = (t < FORCE_OPEN_FIRST_N_STEPS) or (t >= total_time_steps - FORCE_OPEN_LAST_N_STEPS)
            
            if ENABLE_ROAD_CLOSURES and (not force_open_now):
                flooded_edge_indices = get_closed_edges(
                    edges_bng,
                    hec,
                    t_idx,
                    default_threshold=CLOSURE_DEPTH,
                    n_samples=5
                )
                G_t = apply_road_closures_to_graph(G_base, flooded_edge_indices)
            else:
                flooded_edge_indices = set()
                G_t = G_base  # intact network
            
            closed_count = len(flooded_edge_indices)
            metrics['closed_roads_count'].append(closed_count)
            
            total_edges = len(edges_gdf)
            metrics['closed_roads_share'].append(100.0 * closed_count / max(1, total_edges))

            
    


            if ENABLE_ROAD_CLOSURES:
                
                closed_main_edge_indices = [
                    eid for eid in flooded_edge_indices
                    if eid in main_edges_bng.index
                ]
            
                # (2) sample points on those closed edges
                road_pts_EN = []
                road_pts_edge_ids = []
                for eid in closed_main_edge_indices:
                    geom = main_edges_bng.loc[eid].geometry
                    if geom is None:
                        continue
                    pts = sample_line_points(geom, n_samples=8)
                    road_pts_EN.extend(pts)
                    road_pts_edge_ids.extend([eid] * len(pts))
            
                if road_pts_EN:
                    depths_pts, _ = hec.get_depth_wse_at_points(road_pts_EN, t_idx)
            
                    edge_max_depth = {}
                    for d, eid in zip(depths_pts, road_pts_edge_ids):
                        if d > edge_max_depth.get(eid, 0.0):
                            edge_max_depth[eid] = float(d)
            
                    closed_edge_depths = list(edge_max_depth.values())
                    depth_max_closed = max(closed_edge_depths) if closed_edge_depths else 0.0
                    depth_mean_closed = float(np.mean(closed_edge_depths)) if closed_edge_depths else 0.0
                else:
                    depth_max_closed = 0.0
                    depth_mean_closed = 0.0
            
                metrics.setdefault("closed_main_roads_count", []).append(len(closed_main_edge_indices))
            else:
                # No closures => define closure-depth metrics as 0
                depth_max_closed = 0.0
                depth_mean_closed = 0.0
                metrics.setdefault("closed_main_roads_count", []).append(0)
            
            metrics.setdefault("closed_roads_depth_max", []).append(depth_max_closed)
            metrics.setdefault("closed_roads_depth_mean", []).append(depth_mean_closed)


         
            
            # 3b. Demographic exposure: agents physically in flooded area
            flooded_agents = [
                ag for ag in agents
                if ag.get('water_depth', 0.0) > CLOSURE_DEPTH
            ]
            
            age_counts = {age: 0 for age in age_types}
            emp_counts = {emp: 0 for emp in employment_status_types}
            mode_counts = {mode: 0 for mode in travel_modes}
            
            for ag in flooded_agents:
                a = ag.get('age')
                e = ag.get('employment_status')
                m = ag.get('travelMode')
            
                if a in age_counts:
                    age_counts[a] += 1
                if e in emp_counts:
                    emp_counts[e] += 1
                if m in mode_counts:
                    mode_counts[m] += 1
            
            # Store time-resolved exposure counts
            for age in age_types:
                metrics['flooded_by_age'][age].append(age_counts[age]* scaling_factor)
            for emp in employment_status_types:
                metrics['flooded_by_emp'][emp].append(emp_counts[emp]* scaling_factor)
            for mode in travel_modes:
                metrics['flooded_by_mode'][mode].append(mode_counts[mode]* scaling_factor)
                
                

            step = t 
            

            
            def is_active(a):
                return (
                    a.get("trip_active", False)
                    or a.get("waiting_to_depart", False)
                    or a.get("staying", False)
                )
            
            def is_exposed(a):
                return (a.get("water_depth", 0.0) > CLOSURE_DEPTH)
            
            # Count exposures
            age_idx = {g:i for i,g in enumerate(AGE_GROUPS)}
            emp_idx = {g:i for i,g in enumerate(EMP_GROUPS)}
            mode_idx = {g:i for i,g in enumerate(MODE_GROUPS)}
            
            for a in agents:
                if not is_exposed(a):
                    continue
            
                ag = safe_norm(a.get("age", ""))
                em = safe_norm(a.get("employment_status", ""))
                md = safe_norm(a.get("travelMode", ""))
            
                if ag in age_idx:
                    E_age[age_idx[ag], step] += 1
                if em in emp_idx:
                    E_emp[emp_idx[em], step] += 1
                if md in mode_idx:
                    E_mode[mode_idx[md], step] += 1
    


            # 4. Update Agent Activity & Decide Trips
            total_excess_dist_t = 0.0
            total_excess_time_t = 0.0
            traveling_agents_t = 0
            unserved_agents_t = 0
            idle_agents_t = 0
            attempting_agents_t = 0
            departures_this_step = 0
            N_total = len(agents)
            #staying_count = 0



            for i, agent in enumerate(agents):
                # reset per-step disruption (but NOT path / DistanceToTarget)
                agent['baseline_dist'] = 0.0
                agent['flooded_dist'] = 0.0
                agent['excess_dist'] = 0.0
                agent['excess_time'] = 0.0
                #agent['unserved'] = False
                
                # keep unserved status unless a NEW target is assigned this step
                #old_target = agent.get('target', None)
                
                
                # If cooldown ended previously and we stored a planned trip, restore it now
                if agent.get("resume_target") is not None and agent.get("target") is None and agent.get("activity") == "Idle":
                    agent["target"] = agent["resume_target"]
                    agent["activity"] = agent["resume_activity"] if agent.get("resume_activity") else agent["activity"]
                    agent["resume_target"] = None
                    agent["resume_activity"] = None


                # Activity-based demand using LTDS-inspired model
                update_agent_activity(agent, current_hour, current_minute, day_type)
                
                
                age_idx  = {g:i for i,g in enumerate(AGE_GROUPS)}
                emp_idx  = {g:i for i,g in enumerate(EMP_GROUPS)}
                mode_idx = {g:i for i,g in enumerate(MODE_GROUPS)}
                
                step = t
                
                for a in agents:
                    ag = safe_norm(a.get("age", ""))
                    em = safe_norm(a.get("employment_status", ""))
                    md = safe_norm(a.get("travelMode", ""))
                
                    # ---- ACTIVE counts ----
                    if is_active(a):
                        if ag in age_idx:   A_age[age_idx[ag], step]   += 1
                        if em in emp_idx:   A_emp[emp_idx[em], step]   += 1
                        if md in mode_idx:  A_mode[mode_idx[md], step] += 1
                
                    # ---- EXPOSED among ACTIVE (recommended) ----
                    if is_active(a) and is_exposed(a):
                        if ag in age_idx:   E_age[age_idx[ag], step]   += 1
                        if em in emp_idx:   E_emp[emp_idx[em], step]   += 1
                        if md in mode_idx:  E_mode[mode_idx[md], step] += 1
                

                # ==== START / CONTINUE TRIP LOGIC ====
                has_target = agent.get('target') is not None
                is_travelling_activity = (agent['activity'] != 'Idle') and has_target
                wants_to_travel = has_target and is_travelling_activity and not agent.get('has_reached_target', False)

                if wants_to_travel:
                    # New trip starting now
                    if not agent.get('trip_active', False):
                        attempting_agents_t += 1   # NEW: only for newly starting trips
                        departures_this_step += 1
                        traffic_density = 0.6 if 8 <= current_hour < 10 or 17 <= current_hour < 19 else 0.25
                        global_traffic_factor = get_traffic_factor_from_nasch(
                            traffic_density, base_speeds['Drivers'], TOTAL_ROAD_CELLS)

                        if agent.get('travelMode') in ['Drivers', 'PTP']:
                            agent['speed'] = base_speeds[agent['travelMode']] / global_traffic_factor
                        else:
                            agent['speed'] = base_speeds[agent['travelMode']]

                        # Baseline (no flooding)
                        _, base_dist_m = get_astar_path(
                            G_base,
                            agent['x'], agent['y'],
                            agent['target']['X'], agent['target']['Y']
                        )

                        # Flooded network
                        flooded_path_coords, flooded_dist_m = get_astar_path(
                            G_t,
                            agent['x'], agent['y'],
                            agent['target']['X'], agent['target']['Y']
                        )

                        if not flooded_path_coords or base_dist_m <= 0:
                            # Trip unserved in current network (this step)
                            agent["unserved"] = True                 # per-step marker (video)
                            agent["unserved_state"] = True           # persistent state (chart)
                            agent["unserved_since_step"] = t
                            unserved_agents_t += 1
                        
                            agent["path"] = []
                            agent["DistanceToTarget"] = 0.0
                            agent["trip_active"] = False
                            agent["has_reached_target"] = False
    
            
                        else:
                            # Start trip
                            agent['baseline_dist'] = base_dist_m / 1000.0
                            agent['flooded_dist'] = flooded_dist_m / 1000.0
                            if agent['speed'] > 0:
                                agent['current_baseline_time_hr'] = agent['baseline_dist'] / agent['speed']
                                agent['current_flooded_time_hr'] = agent['flooded_dist'] / agent['speed']
                            else:
                                agent['current_baseline_time_hr'] = 0.0
                                agent['current_flooded_time_hr'] = 0.0

                            agent['excess_dist'] = agent['flooded_dist'] - agent['baseline_dist']
                            agent['excess_time'] = agent['current_flooded_time_hr'] - agent['current_baseline_time_hr']

                            agent['path'] = flooded_path_coords
                            agent['DistanceToTarget'] = agent['flooded_dist']
                            agent['trip_active'] = True
                            agent['trip_departure_step'] = t
                            agent['trip_departure_label'] = time_label
                            agent['has_reached_target'] = False

                            total_excess_dist_t += agent['excess_dist']
                            total_excess_time_t += agent['excess_time']
                            traveling_agents_t += 1
                    else:
                        # Trip already active, just count as travelling
                        traveling_agents_t += 1

                # Agent idle?
                if agent['activity'] == 'Idle' or agent['target'] is None:
                    idle_agents_t += 1

            # 5. MOVE AGENTS & CHECK ARRIVALS
            for i, agent in enumerate(agents):
                if agent.get('trip_active', False) and agent.get('path'):
                    destination = agent['path'][-1]

                    distance_travelled_step = agent['speed'] * time_step_increment
                    remaining_distance_this_step = distance_travelled_step

                    while len(agent['path']) > 0 and remaining_distance_this_step > 0.001:
                        next_point = agent['path'][0]
                        R = 6371
                        lat1, lon1 = np.radians(agent['x']), np.radians(agent['y'])
                        lat2, lon2 = np.radians(next_point[0]), np.radians(next_point[1])
                        dlat, dlon = lat2 - lat1, lon2 - lon1
                        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                        distance_to_next = R * c

                        if distance_to_next <= 0.001:
                            agent['path'].pop(0)
                            continue

                        if distance_to_next <= remaining_distance_this_step:
                            agent['x'], agent['y'] = next_point
                            agent['path'].pop(0)
                            remaining_distance_this_step -= distance_to_next
                        else:
                            frac = remaining_distance_this_step / distance_to_next
                            new_x = agent['x'] + (next_point[0] - agent['x']) * frac
                            new_y = agent['y'] + (next_point[1] - agent['y']) * frac
                            agent['x'], agent['y'] = new_x, new_y
                            remaining_distance_this_step = 0

                    agent['DistanceToTarget'] -= distance_travelled_step
                    if agent['DistanceToTarget'] < 0:
                        agent['DistanceToTarget'] = 0

                    # Arrival condition
                    if agent['DistanceToTarget'] <= 0.001 or (not agent['path']):
                        agent['has_reached_target'] = True
                        agent['trip_active'] = False
                        agent['path'] = []

                        if agent.get('trip_departure_step') is not None:
                            realised_hr = (t - agent['trip_departure_step'] + 1) * time_step_increment
                            completed_trips.append({
                                'departure_step': agent['trip_departure_step'],
                                'arrival_step': t,
                                'departure_time_label': agent['trip_departure_label'],
                                'realised_time_hr': realised_hr,
                                'baseline_time_hr_est': agent.get('current_baseline_time_hr', 0.0),
                                'flooded_time_hr_est': agent.get('current_flooded_time_hr', 0.0),
                                'origin_activity': agent.get('previous_activity', 'Home'),
                                'dest_activity': agent.get('activity', None),
                        
                                # >>> LTDS-inspired attributes <<<
                                'age': agent.get('age'),
                                'gender': agent.get('gender'),
                                'employment_status': agent.get('employment_status'),
                                'travel_mode': agent.get('travelMode'),
                                'demographic_modifier': agent.get('demographic_modifier'),
                                'RPI_departure': agent.get('RPI', 0.0),
                            })


            # 6. Collect metrics per timestep

            # ===================== Smooth Unserved -> Idle after 18:00, all by 23:00 =====================
            now_h = current_hour + current_minute / 60.0
            
            if now_h >= 18.0:
                # progress from 0 at 18:00 to 1 at 23:00
                progress = min(1.0, max(0.0, (now_h - 18.0) / (23.0 - 18.0)))
            
                # This makes the decline more gradual early, faster near the middle, then flatten.
                p_give_up = 1.0 / (1.0 + np.exp(-8.0 * (progress - 0.5)))  # 0..1
            
                for ag in agents:
                    if ag.get("unserved_state", False):
                        # At 23:00+, force everyone to Idle
                        if now_h >= 23.0:
                            force_idle = True
                        else:
                            force_idle = (random.random() < p_give_up)
            
                        if force_idle:
                            ag["unserved_state"] = False   # remove from Unserved state
                            ag["unserved"] = False         
            
                            # become Idle and stop trying (for the rest of the night)
                            ag["activity"] = "Idle"
                            ag["target"] = None
                            ag["trip_active"] = False
                            ag["path"] = []
                            ag["DistanceToTarget"] = 0.0
                            ag["waiting_to_depart"] = False
                            ag["staying"] = False
                            ag["stay_until"] = None



            # ===================== STATE ACCOUNTING  =====================
            
            state_idle = 0
            state_travelling = 0   # ACTIVE = travelling + waiting + staying
            state_unserved = 0
            #state_waiting = 0
            #state_staying = 0
            
            for ag in agents:
            
                # 1. Travelling: currently moving on a path
                #if ag.get("trip_active", False):
                    #state_travelling += 1
                    
                 #2) Travelling (ACTIVE): moving OR waiting to depart OR staying at destination
                if ag.get("trip_active", False) or ag.get("waiting_to_depart", False) or ag.get("staying", False):
                    state_travelling += 1    
            
                # 2. Unserved: failed trip, not travelling, still wants to go
                #elif (
                    #ag.get("unserved", False)
                    #and ag.get("target") is not None
                    #and not ag.get("has_reached_target", False)
                #):
                   # state_unserved += 1
                   
                elif ag.get("unserved_state", False):
                    state_unserved += 1

                    
                #elif ag.get("staying", False):
                    #state_staying += 1

                #elif ag.get("waiting_to_depart", False):
                    #state_waiting += 1    
            
                # 3. Idle: everything else
                else:
                    state_idle += 1
            
            
            # ---- Save counts (scaled) ----
            metrics["state_idle_count"].append(state_idle * scaling_factor)
            metrics["state_travelling_count"].append(state_travelling * scaling_factor)
            metrics["state_unserved_count"].append(state_unserved * scaling_factor)
            #metrics.setdefault("state_waiting_count", []).append(state_waiting * scaling_factor)
            #metrics.setdefault("state_staying_count", []).append(state_staying * scaling_factor)
            
            # ---- Save shares ----
            N = len(agents)
            metrics["idle_agents_share"].append(state_idle / N)
            metrics["state_travelling_share"].append(state_travelling / N)
            metrics["unserved_agents_share_state"].append(state_unserved / N)
            #metrics.setdefault("state_waiting_share", []).append(state_waiting / N)
            #metrics.setdefault("state_staying_share", []).append(state_staying / N)
            
            # ---- Sanity check ----
            # metrics.setdefault("state_sum_check", []).append(
                #(state_idle + state_travelling + state_unserved + state_waiting + state_staying) / N
            #)
            
            #---- Sanity check ----
            metrics.setdefault("state_sum_check", []).append((state_idle + state_travelling + state_unserved) / N)
        
            # Save counts-----------------------------------------------------
            metrics['total_excess_dist_km'].append(total_excess_dist_t)
            metrics['total_excess_time_hours'].append(total_excess_time_t)
            metrics["total_agents"].append(N_total* scaling_factor)
            #metrics["state_idle_count"].append(idle_agents_t* scaling_factor)
           # metrics["state_travelling_count"].append(traveling_agents_t* scaling_factor)
            metrics['traveling_agent_count'].append(traveling_agents_t * scaling_factor)
            #metrics["state_unserved_count"].append(unserved_agents_t* scaling_factor)
            metrics["attempting_agents_t"].append(attempting_agents_t* scaling_factor)
            metrics["departures_by_step"].append(departures_this_step * scaling_factor)
            
            # Save shares------------------------------------------------------
            
            agent_count = len(agents) if len(agents) > 0 else 1
            #unserved_state_count = sum(1 for ag in agents if ag.get('unserved', False))
            #metrics['idle_agents_share'].append(idle_agents_t / agent_count)
            
            metrics['unserved_agents_share'].append(unserved_agents_t / max(1, agent_count))
            metrics['unserved_agents_share_attempting'].append(unserved_agents_t / max(1, attempting_agents_t))
            
            #unserved_state_count = sum(1 for ag in agents if ag.get('unserved', False))
            #metrics['unserved_agents_share_state'].append(unserved_agents_t / max(1, agent_count))
            
            metrics['travelling_agents_share'].append(traveling_agents_t / agent_count)
            #metrics['state_travelling_share'].append(traveling_agents_t / agent_count)

            


            # Sanity: these 3 states should sum to ~1.0
            #metrics["state_sum_check"].append(
               #metrics["idle_agents_share"][-1]
                #+ metrics["state_travelling_share"][-1]
                #+ metrics["unserved_agents_share"][-1]
#)


            if traveling_agents_t > 0:
                avg_excess_time_mins = (total_excess_time_t * 60) / traveling_agents_t
            else:
                avg_excess_time_mins = 0.0
            metrics['average_excess_time_mins'].append(avg_excess_time_mins)

            # 7. VISUALISATION
            ax.clear()
            #ax.set_title("Road Closure Scenario")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim(-0.36, -0.29)
            ax.set_ylim(51.37, 51.44)
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=0.5, linestyle='--', color='0.8')

            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.5)
                sp.set_edgecolor('black')

            try:
                E, N, D = hec.get_depth_field(t_idx, area=HEC_FLOW_AREAS[0])

                lon, lat = to_wgs84.transform(E, N)
                flood_mask = D > 0.01
                lon_f = lon[flood_mask]
                lat_f = lat[flood_mask]
                D_f = D[flood_mask]
                sc = ax.scatter(lon_f, lat_f, c=D_f, s=3, cmap='Blues', vmin=0.1, vmax=6.0,
                                alpha=0.9, zorder=1)
                cbar.mappable.set_array(D_f)
            except Exception:
                pass

            try:
                edges_gdf.plot(ax=ax, linewidth=0.5, edgecolor='gray', alpha=0.7, zorder=2)
            
                if ENABLE_ROAD_CLOSURES and len(flooded_edge_indices) > 0:
                    closed_mask = edges_gdf.index.isin(flooded_edge_indices)
                    edges_gdf[closed_mask].plot(ax=ax, linewidth=3, edgecolor='purple',
                                                alpha=0.8, zorder=3)
            except Exception:
                pass


            time_text = f'Time: {current_hour:02d}:{current_minute:02d}'
            ax.text(0.02, 0.98, time_text, transform=ax.transAxes, fontsize=12,
                    fontweight='bold', color='black', va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=1),
                    zorder=20)

            for agent in agents:
                x_lat, y_lon = agent['x'], agent['y']
        
                if agent.get("unserved", False) or agent.get("unserved_state", False):
                    color = "purple"
                elif agent.get("trip_active", False):
                    color = "blue"
                else:
                    color = "grey"
                    
                ax.scatter(y_lon, x_lat, s=30, color=color, edgecolor='black',
                           alpha=0.9, zorder=10)

            handles = [
                mpatches.Patch(color='#08306B', label='River channel'),
                mpatches.Patch(color='#73B2D8', label='Flooded area'),
                plt.Line2D([0], [0], color='gray', lw=1, label='Main road'),
                plt.Line2D([0], [0], marker='o', color='w', label='Travelling',
                           markerfacecolor='blue', markeredgecolor='black', markersize=8),
                plt.Line2D([0], [0], marker='o', color='w', label='Idle',
                           markerfacecolor='grey', markeredgecolor='black', markersize=8),
                plt.Line2D([0], [0], marker='o', color='w', label='Failed trip',
                           markerfacecolor='purple', markeredgecolor='black', markersize=8)
            ]
            
            if ENABLE_ROAD_CLOSURES:
                handles.insert(2, plt.Line2D([0], [0], color='purple', lw=5, label='Flooded road'))

            leg = ax.legend(handles=handles, loc='lower right', fontsize='small',
                            frameon=True)
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_alpha(1.0)
            leg.get_frame().set_linewidth(1.5)
            leg.set_zorder(50)

            writer.grab_frame()
            frame_filename = os.path.join(
                frames_dir,
                f"frame_{t:02d}_{current_hour:02d}{current_minute:02d}.png"
            )
            fig.savefig(frame_filename, dpi=150, bbox_inches='tight',
                        pad_inches=0, transparent=True)

            if t % 5 == 0:
                print(f"  ... saved frame {t}/{total_time_steps}")

            agents_over_time.append(copy.deepcopy(agents))

            time_of_day += time_step_increment
            if time_of_day >= 24:
                time_of_day = 0

    with open(f"agents_over_time_{scenario_name}.pkl", "wb") as f:
        pickle.dump(agents_over_time, f)
    print(f"[SAVE] All agent states saved to agents_over_time_{scenario_name}.pkl")

    plt.close(fig)
    print(f"--- Simulation {scenario_name} Finished. Video saved to {video_file_name} ---")
    print(f"--- All {total_time_steps} frames saved to '{frames_dir}/' directory. ---")
    
    
    # compute RRI matrices + plot heatmaps
    # ---------------------------------------------------------
    RRI_age  = compute_rri_active_matrix(E_age,  A_age,  min_active_total=10, min_active_group=3)
    RRI_emp  = compute_rri_active_matrix(E_emp,  A_emp,  min_active_total=10, min_active_group=3)
    RRI_mode = compute_rri_active_matrix(E_mode, A_mode, min_active_total=10, min_active_group=3)
    
    # Create output folder
    out_dir = os.path.join("outputs", f"{day_type}_heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving heatmaps to: {out_dir}")
    
    
    # Define simulation start hour for 06:00–06:00 heatmaps
    START_HOUR = 6
    
    # ---------------------------------------------------------
    # Exposure heatmaps (absolute counts) – plotted 06:00 → 06:00
    # ---------------------------------------------------------
    plot_heatmap(E_age, AGE_GROUPS, time_labels,
                 #title=f"Exposure by Age Group",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_age_{day_type}.png"),
                 start_hour=START_HOUR)
    
    plot_heatmap(E_emp, EMP_GROUPS, time_labels,
                 #title=f"Exposure by Employment Status",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_employment_{day_type}.png"),
                 start_hour=START_HOUR)
    
    plot_heatmap(E_mode, MODE_GROUPS, time_labels,
                 #title=f"Exposure by Travel Mode",
                 cbar_label="Exposed agents (count)",
                 out_png=os.path.join(out_dir, f"exposure_heatmap_mode_{day_type}.png"),
                 start_hour=START_HOUR)
    
    
    # ---------------------------------------------------------
    # RRI heatmaps (relative risk) – plotted 06:00 → 06:00
    # ---------------------------------------------------------
    plot_heatmap(RRI_age, AGE_GROUPS, time_labels,
                 #title=f"RRI by Age Group",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_age_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_age))),
                 start_hour=START_HOUR)
    
    plot_heatmap(RRI_emp, EMP_GROUPS, time_labels,
                 #title=f"RRI by Employment Status",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_employment_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_emp))),
                 start_hour=START_HOUR)
    
    plot_heatmap(RRI_mode, MODE_GROUPS, time_labels,
                 #title=f"RRI by Travel Mode",
                 cbar_label="Relative Risk Index (RRI)",
                 out_png=os.path.join(out_dir, f"rri_heatmap_mode_{day_type}.png"),
                 vmin=0.0,
                 vmax=max(2.0, float(np.nanmax(RRI_mode))),
                 start_hour=START_HOUR)
    
    print("Heatmaps successfully saved (plotted 06:00 → 06:00).")

    return metrics, agents_over_time, completed_trips




# =============================================================================
# -----------------------------<< MAIN EXECUTION>> ----------------------------
# =============================================================================

# Weekday scenario
results_weekday, agents_weekday, completed_trips = run_simulation(
    scenario_name="Road_Closure_Weekday",
    day_type="weekday"
)

#scenario_name="Road_Closure_Weekend",
     #day_type="weekend"
 #)

# Collect scenario results for saving
all_scenario_results = {
    "Road_Closure_Weekday": results_weekday,
     #"Road_Closure_Weekend": results_weekend,   # For weekend
}


trips_df = pd.DataFrame(completed_trips)

if trips_df.empty:
    print("No completed trips to plot.")
else:
    # Convert hours -> minutes
    trips_df["realised_time_min"] = trips_df["realised_time_hr"] * 60.0
    trips_df["baseline_time_min_est"] = trips_df["baseline_time_hr_est"] * 60.0
    trips_df["flooded_time_min_est"]  = trips_df["flooded_time_hr_est"] * 60.0
    trips_df["excess_time_min"] = (trips_df["flooded_time_hr_est"] - trips_df["baseline_time_hr_est"]) * 60.0


    trips_df["dep_hour"] = trips_df["departure_time_label"].str.slice(0, 2).astype(int)

# ==========================================================================================================================================
# --------------------------------------------------<< FINAL PLOTTING & DATA SAVING >>----------------------------------------------------
# =========================================================================================================================================

print("\n--- Generating Final Plots ---")

plt.style.use('seaborn-v0_8-darkgrid')


time_steps = np.arange(total_time_steps)

time_labels = [
    (DISPLAY_START_CLOCK + datetime.timedelta(minutes=STEP_MINUTES * i)).strftime('%H:%M')
    for i in range(total_time_steps)
]

def sl(x):
    """No slicing. Keep full 24h series, just relabeled in plots."""
    return np.asarray(x)

tick_idx = np.arange(0, total_time_steps, 2)  # hourly ticks for 30-min step



# =============================
# SNAPSHOTS for ALL timesteps
# =============================
snap_dir = "snapshots_all_steps_RPI_or_depth_Weekday"
os.makedirs(snap_dir, exist_ok=True)


# Loop all steps
for t in range(total_time_steps):
    t_idx = hec_indices[t]                 # HEC timestep aligned to ABM step
    tstr = time_labels[t]                  # "06:00", "06:30", ...

    out_png = os.path.join(
        snap_dir,
        f"snapshot_{t:03d}_{tstr.replace(':','')}.png"
    )

    plot_agents_snapshot(
        agents_t=agents_weekday[t] if isinstance(agents_weekday, list) else agents_weekday[t],

        hec=hec,
        t_idx=t_idx,
        title_time_str=tstr,
        edges_gdf=edges_gdf,
        out_png=out_png,
        hec_area=HEC_FLOW_AREAS[0],
        rpi_thr=0.8,
        depth_thr=0.30
    )

    if t % 5 == 0:
        print(f"[SNAPSHOT] saved {t}/{total_time_steps}: {out_png}")



# ===================== 3.1 Flood exposure by age group =======================
age_colors = {
    'Children': 'tab:blue',
    'Adults': 'tab:orange',
    'Seniors': 'tab:green'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series: number of agents in flooded area by age ---
for age in age_types:
    series = sl(results_weekday['flooded_by_age'][age])
    ax_line.plot(
        time_steps,
        series,
        label=age,
        linewidth=2,
        marker='o',
        markersize=3,
        color=age_colors.get(age, None)
    )

ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')
ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Age Group")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart: total exposure share by age (over whole simulation) ---
age_totals = [sum(sl(results_weekday['flooded_by_age'][age])) for age in age_types]
total_exposed = sum(age_totals)

ax_pie.axis('equal')
if total_exposed > 0:
    wedges, texts, autotexts = ax_pie.pie(
        age_totals,
        labels=age_types,
        autopct='%1.1f%%',
        startangle=90,
        colors=[age_colors[a] for a in age_types],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Age Group")

plt.tight_layout()
plt.savefig("plot_3_1_flood_exposure_age_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()


# ================= 3.2 Flood exposure by employment status ====================
emp_colors = {
    'Employed': 'tab:blue',
    'Unemployed': 'tab:red',
    'Student': 'tab:green'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series ---
for emp in employment_status_types:
    series = sl(results_weekday['flooded_by_emp'][emp])
    ax_line.plot(
        time_steps,
        series,
        label=emp,
        linewidth=2,
        marker='o',
        markersize=3,
        color=emp_colors.get(emp, None)
    )

ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')

ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Employment Status")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart ---
emp_totals = [sum(sl(results_weekday['flooded_by_emp'][emp])) for emp in employment_status_types]
total_exposed_emp = sum(emp_totals)

ax_pie.axis('equal')
if total_exposed_emp > 0:
    wedges, texts, autotexts = ax_pie.pie(
        emp_totals,
        labels=employment_status_types,
        autopct='%1.1f%%',
        startangle=90,
        colors=[emp_colors[e] for e in employment_status_types],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Employment Status")

plt.tight_layout()
plt.savefig("plot_3_2_flood_exposure_employment_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()


# =================== 3.3 Flood exposure by travel mode =======================
mode_colors = {
    'Walkers': 'tab:green',
    'Cyclists': 'tab:cyan',
    'PTP': 'tab:purple',
    'Drivers': 'tab:orange'
}

fig, (ax_line, ax_pie) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.2, 1]}
)

# --- Time series ---
for mode in travel_modes:
    series = sl(results_weekday['flooded_by_mode'][mode])
    ax_line.plot(
        time_steps,
        series,
        label=mode.capitalize(),
        linewidth=2,
        marker='o',
        markersize=3,
        color=mode_colors.get(mode, None)
    )
ax_line.set_xticks(tick_idx)
ax_line.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha='right')

ax_line.set_xlabel("Time of day")
ax_line.set_ylabel("Number of people in flooded area")
ax_line.set_title("Agents Exposed to Flooding by Travel Mode")
ax_line.grid(True, linestyle='--', alpha=0.5)

for spine in ax_line.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.3)

legend = ax_line.legend(
    loc='upper right',
    frameon=True,
    framealpha=1.0,
    edgecolor='black',
    facecolor='white'
)
legend.get_frame().set_linewidth(1.2)

# --- Pie chart ---
mode_totals = [sum(sl(results_weekday['flooded_by_mode'][mode])) for mode in travel_modes]
total_exposed_mode = sum(mode_totals)

ax_pie.axis('equal')
if total_exposed_mode > 0:
    wedges, texts, autotexts = ax_pie.pie(
        mode_totals,
        labels=[m.capitalize() for m in travel_modes],
        autopct='%1.1f%%',
        startangle=90,
        colors=[mode_colors[m] for m in travel_modes],
        pctdistance=0.8
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax_pie.add_artist(centre_circle)
else:
    ax_pie.text(0.5, 0.5, "No exposure", ha='center', va='center')

ax_pie.set_title("Total Flood Exposure Share by Travel Mode")

plt.tight_layout()
plt.savefig("plot_3_3_flood_exposure_mode_timeseries_pie_weekday_18-02-26_NewRPI.png",
            dpi=300, bbox_inches='tight')
plt.show()




#--------------- Number of travelling, unserved, idle, and percentage of unserved agent--------------


# ---- Prepare data ----
time_steps = np.arange(len(results_weekday["total_agents"]))

total_agents = np.array(results_weekday["total_agents"])
travelling = np.array(results_weekday["state_travelling_count"])
idle = np.array(results_weekday["state_idle_count"])
unserved = np.array(results_weekday["state_unserved_count"])
unserved_pct = np.array(results_weekday["unserved_agents_share_state"]) * 100

# ---- Bar settings ----
bar_width = 0.18

# ---- Create figure ----
fig, ax1 = plt.subplots(figsize=(14, 6))

# ---- Bars: agent states (counts) ----
ax1.bar(time_steps - 1.5*bar_width, total_agents,
        width=bar_width, label="Total Agents", color="lightgray")

ax1.bar(time_steps - 0.5*bar_width, travelling,
        width=bar_width, label="Travelling", color="tab:blue")

ax1.bar(time_steps + 0.5*bar_width, idle,
        width=bar_width, label="Idle", color="tab:orange")

ax1.bar(time_steps + 1.5*bar_width, unserved,
        width=bar_width, label="Failed trips", color="tab:purple")

ax1.set_ylabel("Number of people")
ax1.set_xlabel("Time of day")
ax1.set_ylim(0, total_agents.max() * 1.1)

# ---- Secondary axis: unserved percentage ----
ax2 = ax1.twinx()
ax2.plot(time_steps, unserved_pct,
         color="black", marker="o", linewidth=2,
         label="Unserved (%)")

ax2.set_ylabel("Unserved agents (%)")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0, max(5, unserved_pct.max() * 1.2))

# ---- X-axis formatting ----
ax1.set_xticks(tick_idx)
ax1.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha="right")

# ---- Grid & aesthetics ----
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for spine in ax1.spines.values():
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

for spine in ax2.spines.values():
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

# ---- Combined legend ----
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend = ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    edgecolor="black"
)
legend.get_frame().set_linewidth(1.3)

# ---- Title ----
#plt.title("Agent States and Unserved Demand Over Time")

# ---------------- Save & show ----------------
output_file = "agent_states_vs_unserved agents_weekday_30min_18-02-26.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

#--------------- Number of travelling, unserved, idle, and percentage of the closed road agent--------------


# ---------------- Prepare data ----------------
time_steps = np.arange(len(results_weekday["total_agents"]))

total_agents = np.array(results_weekday["total_agents"])
travelling = np.array(results_weekday["state_travelling_count"])
idle = np.array(results_weekday["state_idle_count"])
unserved = np.array(results_weekday["state_unserved_count"])
#waiting = np.array(results_weekday["state_waiting_count"])
#staying = np.array(results_weekday["state_staying_count"])


closed_roads_pct = np.array(results_weekday["closed_roads_share"])

# ---------------- Plot settings ----------------
bar_width = 0.13

fig, ax1 = plt.subplots(figsize=(14, 6))

# >>> Make background white
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# ---------------- Bars: agent states ----------------
ax1.bar(time_steps - 2.5 * bar_width, total_agents, width=bar_width, label="Total agents", color="lightgray")
ax1.bar(time_steps - 1.5 * bar_width, travelling,   width=bar_width, label="Travelling agents", color="tab:blue")
ax1.bar(time_steps - 0.5 * bar_width, unserved,     width=bar_width, label="Failed trips", color="tab:purple")
#ax1.bar(time_steps + 0.5 * bar_width, waiting,      width=bar_width, label="Waiting agents", color="tab:green")
#ax1.bar(time_steps + 1.5 * bar_width, staying,      width=bar_width, label="Staying agents", color="tab:brown")
ax1.bar(time_steps + 2.5 * bar_width, idle,         width=bar_width, label="Idle agents", color="tab:orange")


ax1.set_xlabel("Time of day")
ax1.set_ylabel("Number of people")
ax1.set_ylim(0, total_agents.max() * 1.1)

# ---------------- Secondary axis: closed roads (%) ----------------
ax2 = ax1.twinx()

ax2.plot(time_steps, closed_roads_pct,
         color="red", marker="s", linestyle="--",
         linewidth=2, label="Closed roads")

ax2.set_ylabel("Closed roads")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0, max(5, closed_roads_pct.max() * 1.2))

# ---------------- X-axis ticks ----------------
ax1.set_xticks(tick_idx)
ax1.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha="right")

# ---------------- Grid & aesthetics ----------------
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

# ---------------- Legend ----------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend = ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    edgecolor="black"
)
legend.get_frame().set_linewidth(1.3)

# ---------------- Title ----------------
#plt.title("Agent States and Road Closures Over Time")

# ---------------- Save & show ----------------
output_file = "agent_states_vs_closed_roads_weekday_30min_18-02-26.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
plt.show()

#-------------------Number of travelling, unserved, idle, and percentage of the closed road agent--------------

# ---------------- Prepare data ----------------
time_steps = np.arange(len(results_weekday["total_agents"]))

total_agents = sl(results_weekday["total_agents"])
travelling = sl(results_weekday["state_travelling_count"])
idle = sl(results_weekday["state_idle_count"])
unserved = sl(results_weekday["state_unserved_count"])
#waiting = sl(results_weekday["state_waiting_count"])
#staying = sl(results_weekday["state_staying_count"])

# NEW: RPI time series (stored as 0..1)
mean_rpi = np.array(sl(results_weekday["mean_rpi"]))
mean_rpi_pct = mean_rpi * 100.0   # plot as %

# ---------------- Plot settings ----------------
bar_width = 0.13
fig, ax1 = plt.subplots(figsize=(14, 6))

# ---------------- Bars: agent states ----------------
ax1.bar(time_steps - 2.5 * bar_width, total_agents, width=bar_width,
        label="Total agents", color="lightgray")
ax1.bar(time_steps - 1.5 * bar_width, travelling, width=bar_width,
        label="Travelling agents", color="tab:blue")
ax1.bar(time_steps - 0.5 * bar_width, unserved, width=bar_width,
        label="Failed Trips", color="tab:purple")
#ax1.bar(time_steps + 0.5 * bar_width, waiting, width=bar_width,
       #label="Waiting agents", color="tab:green")
#ax1.bar(time_steps + 1.5 * bar_width, staying, width=bar_width,
        #label="Staying agents", color="tab:brown")
ax1.bar(time_steps + 2.5 * bar_width, idle, width=bar_width,
        label="Idle agents", color="tab:orange")

ax1.set_xlabel("Time of day")
ax1.set_ylabel("Number of people")
ax1.set_ylim(0, total_agents.max() * 1.1)

# ---------------- Secondary axis: Mean RPI (%) ----------------
ax2 = ax1.twinx()

ax2.plot(time_steps, mean_rpi_pct,
         color="black", marker="o", linestyle="--",
         linewidth=2, label="Mean RPI (%)")

ax2.set_ylabel("Mean RPI (%)")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_ylim(0, max(5, mean_rpi_pct.max() * 1.2))

# ---------------- X-axis ticks ----------------

ax1.set_xticks(tick_idx)
ax1.set_xticklabels([time_labels[i] for i in tick_idx], rotation=45, ha="right")

# ---------------- Grid & aesthetics ----------------
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.3)
    spine.set_edgecolor("black")

# ---------------- Legend ----------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend = ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    edgecolor="black"
)
legend.get_frame().set_linewidth(1.3)

# ---------------- Title ----------------
plt.title("Agent States and Mean RPI Over Time")

# ---------------- Save & show ----------------
output_file = "agent_states_vs_mean_rpi_weekday_30min_18-02-26.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------
# Plot
# ----------------------------


# ----------------------------
# Group by departure hour
# ----------------------------
by_hour = (trips_df
           .groupby("dep_hour", as_index=False)
           .agg(
               baseline_mean_min=("baseline_time_min_est", "mean"),
               flooded_mean_min=("flooded_time_min_est", "mean"),
               n_trips=("realised_time_min", "size"),
           ))

def sort_key(h):
    return h if h >= 6 else h + 24

by_hour["sort_h"] = by_hour["dep_hour"].apply(sort_key)
by_hour = by_hour.sort_values("sort_h")

# Make a continuous x-axis like 6..29 (where 24..29 represents 0..5)
x = by_hour["sort_h"].to_numpy()
xticklabels = [f"{(int(h) % 24):02d}:00" for h in x]
    
plt.figure(figsize=(12, 5))

plt.plot(
    x, by_hour["baseline_mean_min"],
    marker="o", linewidth=2, label="Baseline (No Flooding)"
)
plt.plot(
    x, by_hour["flooded_mean_min"],
    marker="s", linewidth=2, linestyle="--", label="Flooded Network"
)

plt.title("Average Travel Time by Departure Hour")
plt.xlabel("Departure Time (Hour of Day)")
plt.ylabel("Average Travel Time per Completed Trip (Minutes)")

plt.xticks(x, xticklabels, rotation=0)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(frameon=True)

plt.tight_layout()
plt.savefig("avg_travel_time_by_departure_hour_18-02-26.png", dpi=300, bbox_inches="tight")
plt.show()


#------------------------------------------------------------------------------

# --- ---------------------------Save Data ------------------------------------
print("\n--- Saving All Scenario Results ---")


base = {
    'time_step': list(time_steps),
    'time_of_day': list(time_labels),
    'people_at_risk': sl(results_weekday['people_at_risk']),
    'unserved_agents_share_state': sl(results_weekday['unserved_agents_share_state']),
    'travelling_agents_share': sl(results_weekday['travelling_agents_share']),
    'idle_agents_share': sl(results_weekday['idle_agents_share']),
    'traveling_agent_count': sl(results_weekday['traveling_agent_count']),
    'total_agents': sl(results_weekday['total_agents']),
    'state_idle_count': sl(results_weekday['state_idle_count']),
    'state_travelling_count': sl(results_weekday['state_travelling_count']),
    'state_unserved_count': sl(results_weekday['state_unserved_count']),
    'closed_roads_count': sl(results_weekday['closed_roads_count']),
    'closed_roads_share': sl(results_weekday['closed_roads_share']),
    'mean_rpi': sl(results_weekday['mean_rpi']),
    'median_rpi': sl(results_weekday['median_rpi']),
}
results_df = pd.DataFrame(base)

# --- flatten demographic dict-of-lists into columns ---
# Age
for age in age_types:
    results_df[f"flooded_by_age_{age}"] = sl(results_weekday['flooded_by_age'][age])

# Employment
for emp in employment_status_types:
    results_df[f"flooded_by_emp_{emp}"] = sl(results_weekday['flooded_by_emp'][emp])

# Mode
for mode in travel_modes:
    results_df[f"flooded_by_mode_{mode}"] = sl(results_weekday['flooded_by_mode'][mode])
    

results_df.to_csv("No_Closure_Weekday_results_highAgent_weekday_18-02-26_LTDSPop_ExposureScenario_UnserevdAgent_NewRPI.csv", index=False)

# --- Save Workspace ---
save_workspace(
    filename="ABM_HECRAS_No_Closure_Weekday_high agent_weekday_18-02-26_LTDSPop_ExposureScenario_Unserevd agents_NewRPI.pkl",
    scenario_results=all_scenario_results
)

print("\n--- Simulation Complete ---")
