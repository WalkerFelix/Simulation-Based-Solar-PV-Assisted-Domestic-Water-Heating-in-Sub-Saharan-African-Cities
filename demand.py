"""
demand.py

Load and process hot-water draw profiles. Aggregates 1-minute profiles to 5-minute steps.
"""

import pandas as pd
import numpy as np

class DemandProfile:

    """
    Convert volume profiles to 5-min energy draws.
    """

    def __init__(self, profile_path: str, tank_setpoint: float, temp_in: float):
        # Detect separator
        df = pd.read_csv(profile_path, sep=None, engine="python")

        season_cols = [c for c in df.columns if c.lower().endswith("water_consumption")]
        if not season_cols:
            raise ValueError(
                f"No '*Water_Consumption' columns found in {profile_path}. "
                f"Got columns: {list(df.columns)[:8]}..."
            )

        for c in season_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        self.tank_setpoint = float(tank_setpoint)
        self.temp_in = float(temp_in)

    def get_draw_energy(self) -> pd.Series:
        mass_kg = self.df["volume_l"]
        c = 4184.0
        dT = self.tank_setpoint - self.temp_in
        energy_J = mass_kg * c * dT
        return energy_J / (3600.0 * 1000.0)  # kWh
