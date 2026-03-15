# config.py

import os

ROOT_DIR = r"C:\Users\LukeSkywalker\Documents\Schule\Studium\SE5\Praxissemester in SA\Richtiger Code + Richies Paper\Code Versuch"
SOLAR_DATA_DIR = os.path.join(ROOT_DIR, "solar_data")
USER_DATA_DIR  = os.path.join(ROOT_DIR, "user_data", "Classified_Profiles")

# Simulation knobs
SELECTED_PROFILES = ['Light', 'Medium', 'Heavy'] 
SELECTED_LOCATIONS  = ["CapeTown", "Johannesburg", "Lusaka", "Luanda", "Kinshasa", "Nairobi", "Lagos"]          
NUM_PANELS          = 4                     # PV-panels per string 
TILT                = 30                    # deg
AZIMUTH             = 30                    # deg (0 = North in PVLib)
EQUATOR_LATITUDE_THRESHOLD = 15.0           # Cities located between -15°C and +15°C are considered tropical.
TROPICAL_SUBSTITUTE_SEASON = "Spring"       # Season replacement instead of winter

# Locations & site metadata
LOCATIONS = ["CapeTown","Johannesburg","Lusaka","Luanda","Kinshasa","Nairobi","Lagos"]

LOCATION_PARAMS = {
    "CapeTown":     {"latitude": -33.9249, "longitude":  18.4241, "timezone": "Africa/Johannesburg"},
    "Johannesburg": {"latitude": -26.2041, "longitude":  28.0473, "timezone": "Africa/Johannesburg"},
    "Lusaka":       {"latitude": -15.3875, "longitude":  28.3228, "timezone": "Africa/Lusaka"},
    "Luanda":       {"latitude":  -8.8390, "longitude":  13.2894, "timezone": "Africa/Luanda"},
    "Kinshasa":     {"latitude":  -4.4419, "longitude":  15.2663, "timezone": "Africa/Kinshasa"},
    "Nairobi":      {"latitude":  -1.2921, "longitude":  36.8219, "timezone": "Africa/Nairobi"},
    "Lagos":        {"latitude":   6.5244, "longitude":   3.3792, "timezone": "Africa/Lagos"},
}

# PV orientation Per-city in degrees
# surface_tilt: tilt from horizontal (0 = flat, 90 = vertical)
# azimuth: pvlib convention (typically 180 = south in N hemisphere, 0/360 = north)
CITY_ORIENTATIONS = {
    "CapeTown":     {"surface_tilt": 45.0, "azimuth": 45.0},
    "Johannesburg": {"surface_tilt": 40.0, "azimuth": 30.0},
    "Lusaka":       {"surface_tilt": 35.0, "azimuth": 45.0},
    "Luanda":       {"surface_tilt": 25.0,  "azimuth": 60.0},
    "Kinshasa":     {"surface_tilt": 15.0,  "azimuth": 90.0},
    "Nairobi":      {"surface_tilt": 20.0, "azimuth": 90.0},
    "Lagos":        {"surface_tilt": 20.0, "azimuth": 90.0},
}

def get_irradiance_path(city: str) -> str:

    """
    Strict resolver:
      <SOLAR_DATA_DIR>/<city>/<city>.csv
      <SOLAR_DATA_DIR>/<city>/<city>_meteo_5min.csv
    """
    
    base = os.path.join(SOLAR_DATA_DIR, city)

    candidates = [
        os.path.join(base, f"{city}.csv"),
        os.path.join(base, f"{city}_meteo_5min.csv"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p
    
def get_system_params(city: str) -> dict:

    """Return a location PV system parameter dictionary."""

    if city not in LOCATION_PARAMS:
        raise ValueError(f"Unknown city '{city}'. Valid: {LOCATIONS}")
    loc = LOCATION_PARAMS[city]
    lat = loc["latitude"]

    tilt = float(TILT)
    azimuth = float(AZIMUTH)
    
    default_azimuth = 180.0 if lat > 0 else 0.0
    
    if city in CITY_ORIENTATIONS:
        tilt = float(CITY_ORIENTATIONS[city].get("surface_tilt", tilt))
        azimuth = float(CITY_ORIENTATIONS[city].get("azimuth", default_azimuth))

    return {
        "tilt": tilt,
        "azimuth": azimuth,
        "inverter": {"pdc0": 3000, "eta_inv_nom": 0.96},
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "timezone": loc["timezone"],
        "racking_model": "open_rack_glass_polymer",
        "num_panels": NUM_PANELS,
    }

# Tank & demand parameters
TANK_PARAMS = {
    "setpoint": 65.0,             # °C
    "deadband": 3.0,              # °C
    "volume_l": 150,              # L
    "c": 4184,                    # J/(kg·K)
    "rho": 1000,                  # kg/m³
    "R_th": 2.5,                  # K/W
    "element_rating_kw": 3.0,     # kW
    "element_voltage_v": 230.0,   # V
    "dt_s": 300,                  # 5-min step
    "cold_event_temperature": 40,    # °C (threshold for a cold event)
    "cold_event_min_volume_L": 2.0,  # L (minimum total volume per event)
    "pv_enable_min_kw": 0.2,     # minimum PV power before we switch to PV mode
}

SIM_PARAMS = {
    "mains_inlet_temperature_C": 15.0,  # °C
    "min_draw_l_per_event": 2.0,     # L
}


# Diagnostics
PERMANENT_LOAD_TEST = False   # set True to force element ON every step
DIAG_PRINTS = False            # set False to silence the prints

# Runsweep
RUN_SWEEP = False #True  # set to False/True to skip/run

# Demand synthesis from seasonal month blocks
DEMAND_REPEAT_PER_SEASON = 3      # always 3 months per season
REVERSE_SEASONS_FOR = {"Lagos"}   # cities that use Northern Hemisphere seasons

# PV module choice
MODULE_NAME = "United_Renewable_Energy_Co_Ltd_D7K420H8A"