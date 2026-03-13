# simulate.py

import os
from typing import Dict
from datetime import time
import numpy as np
import pandas as pd
import pvlib
from tank import StratifiedTank
from demand import DemandProfile
from pv_module import PVModule
from utils import CHUNK_SIZE
from config import (
    TANK_PARAMS, SIM_PARAMS, MODULE_NAME,
    PERMANENT_LOAD_TEST, DEMAND_REPEAT_PER_SEASON, REVERSE_SEASONS_FOR,DIAG_PRINTS,
)

# Southern Hemisphere mapping
_SEASON_BY_MONTH_SOUTH = {
    12: "Summer", 1: "Summer", 2: "Summer",
     3: "Autumn", 4: "Autumn", 5: "Autumn",
     6: "Winter", 7: "Winter", 8: "Winter",
     9: "Spring",10: "Spring",11: "Spring",
}

# Northern Hemisphere mapping
_SEASON_BY_MONTH_NORTH = {
    12: "Winter", 1: "Winter", 2: "Winter",
     3: "Spring", 4: "Spring", 5: "Spring",
     6: "Summer", 7: "Summer", 8: "Summer",
     9: "Autumn",10: "Autumn",11: "Autumn",
}

def _read_seasonal_blocks(profile_path: str) -> dict[str, pd.Series]:
    """
    Read one CSV with columns like:
      <Season>_Timestamps, <Season>_Water_Consumption, <Season>_Ambient_Temperature, <Season>_Power
    for each of: Summer, Autumn, Winter, Spring.

    Returns { "Summer": Series[L/min @ 1min], ... } indexed by 1-minute timestamps.
    Clips negatives to zero. Missing minutes become 0 L/min.
    """
    df = pd.read_csv(profile_path, low_memory=False)
    out: dict[str, pd.Series] = {}

    for season in ("Summer", "Autumn", "Winter", "Spring"):
        ts_col = f"{season}_Timestamps"
        wc_col = f"{season}_Water_Consumption"
        if ts_col not in df.columns or wc_col not in df.columns:
            raise ValueError(f"{profile_path}: missing columns for season '{season}'")

        ts = pd.to_datetime(df[ts_col], errors="coerce")
        vals = pd.to_numeric(df[wc_col], errors="coerce").fillna(0.0).clip(lower=0.0)

        s = pd.Series(vals.to_numpy(float), index=ts)
        s = s.loc[s.index.notna()].sort_index()

        # ensure 1-min cadence
        s = s.resample("1min").sum().fillna(0.0)
        out[season] = s
    return out

def _seasonal_energy_and_litres(
    season_L_per_min: pd.Series, setpoint_C: float, cold_C: float
) -> tuple[pd.Series, pd.Series]:
    
    """
    Convert 1-minute L/min to two aligned 5-min series:
    - litres_per_step (L/5min)  = sum over each 5-minute bin
    - energy_kwh (kWh/5min)     = litres * 4184 * dT / 3.6e6
    """

    dT = float(setpoint_C) - float(cold_C)
    L_5 = season_L_per_min.resample("5min", label="right", closed="right").sum()
    kWh = L_5 * 4184.0 * dT / 3_600_000.0
    return kWh, L_5

def _stitch_full_year_from_seasons(
    energy_5min: dict[str, pd.Series],
    litres_5min: dict[str, pd.Series],
    irr_index_local_5min: pd.DatetimeIndex,
    city: str,
    repeats_per_season: int,
) -> tuple[pd.Series, float]:
    
    season_by_month = _SEASON_BY_MONTH_NORTH if city in REVERSE_SEASONS_FOR else _SEASON_BY_MONTH_SOUTH

    # Pre-tile each season’s arrays
    tiled_energy: dict[str, np.ndarray] = {}
    tiled_litres: dict[str, np.ndarray] = {}
    for season in ("Summer", "Autumn", "Winter", "Spring"):
        base_e = energy_5min[season].to_numpy(float)
        base_L = litres_5min[season].to_numpy(float)
        if base_e.size == 0:
            base_e = np.zeros(1, float)
            base_L = np.zeros(1, float)
        tiled_energy[season] = np.tile(base_e, max(1, repeats_per_season))
        tiled_litres[season] = np.tile(base_L, max(1, repeats_per_season))

    # Grouped by year and month
    month_slots: dict[tuple[int, int], np.ndarray] = {}
    for i, ts in enumerate(irr_index_local_5min):
        month_slots.setdefault((ts.year, ts.month), []).append(i)

    # Each month of the season picks up where the previous left off
    curs_e = {s: 0 for s in tiled_energy}
    curs_L = {s: 0 for s in tiled_litres}

    out_energy = np.zeros(len(irr_index_local_5min), float)
    total_L = 0.0

    for (y, m), idxs in month_slots.items():
        season = season_by_month[m]
        seq_e = tiled_energy[season]
        seq_L = tiled_litres[season]
        cur_e = curs_e[season]
        cur_L = curs_L[season]
        need = len(idxs)

        # energy slice
        if cur_e + need <= seq_e.size:
            take_e = seq_e[cur_e:cur_e+need];  cur_e += need
        else:
            r = (cur_e + need) - seq_e.size
            take_e = np.concatenate([seq_e[cur_e:], seq_e[:r]]);  cur_e = r

        # litres slice
        if cur_L + need <= seq_L.size:
            take_L = seq_L[cur_L:cur_L+need];  cur_L += need
        else:
            r = (cur_L + need) - seq_L.size
            take_L = np.concatenate([seq_L[cur_L:], seq_L[:r]]);  cur_L = r

        out_energy[np.array(idxs, dtype=int)] = take_e
        total_L += float(take_L.sum())

        curs_e[season] = cur_e
        curs_L[season] = cur_L

    return pd.Series(out_energy, index=irr_index_local_5min), float(total_L)


def get_nrel_mains_temperature(doy_arr: np.ndarray, t_amb_series: pd.Series, lat: float) -> np.ndarray:
    
    # Calculates the inlet water temperature according to the NREL / Energy Star algorithm.
    
    # Annual averages and monthly extremes from local environmental data
    t_avg_c = t_amb_series.mean()
    monthly_avg_temps = t_amb_series.resample('ME').mean()
    t_max_c = monthly_avg_temps.max()
    t_min_c = monthly_avg_temps.min()
    
    # Conversion to Fahrenheit
    t_avg_f = t_avg_c * 1.8 + 32.0
    t_max_f = t_max_c * 1.8 + 32.0
    t_min_f = t_min_c * 1.8 + 32.0
    
    # Calculate NREL parameters
    ratio = 0.4 + 0.01 * (t_avg_f - 44.0)
    lag = 35.0 - 1.0 * (t_avg_f - 44.0)
    
    # Hemisphere adjustment (NREL assumes Northern Hemisphere)
    day_calc = doy_arr.astype(float)
    if lat < 0:
        # Shift in the Southern Hemisphere
        day_calc = (day_calc + 182.5) % 365
        
    angle_rad = (2 * np.pi / 365.0) * (day_calc - 15.0 - lag) - (np.pi / 2.0)
    
    # Calculate tap water temperature
    amp_f = (t_max_f - t_min_f) / 2.0
    t_mains_f = (t_avg_f + 6.0) + ratio * amp_f * np.sin(angle_rad)
    
    # Convert to Celsius
    t_mains_c = (t_mains_f - 32.0) / 1.8
    return np.clip(t_mains_c, 0.0, None)

def simulate_household_efficient(
    profile_path: str,
    irr_df: pd.DataFrame,
    flat_rate_usd_per_kwh: float,
    system_params: Dict,
    city: str,
) -> Dict:
    
    """
        Simulates one household EWH with PV-assisted controller.

        Control and dispatch:
        - When the Thermostat is ON, the heater requests up to element_rating_kw.
        - Source selection: controller selects PV mode or grid mode per timestep.
        - PV mode: heating power is from PV only and limited by available PV power
        - Grid mode: heating power is supplied from the grid
        - PV and grid power are not combined simultaneously.

        Comfort metric:
        - Cold draws are counted as draw events with total volume >= cold_event_min_volume_L.
        - An event is cold if outlet proxy temperature is below cold_event_temperature.
    """
    
    try:
        household_id = os.path.basename(profile_path)

        # Build full-year demand
        def _read_seasonal_blocks(csv_path: str) -> dict[str, pd.Series]:
            df = pd.read_csv(csv_path, low_memory=False)
            out = {}
            for season in ("Summer", "Autumn", "Winter", "Spring"):
                ts_col = f"{season}_Timestamps"
                wc_col = f"{season}_Water_Consumption"
                if ts_col not in df.columns or wc_col not in df.columns:
                    raise ValueError(f"{csv_path}: missing columns for {season}")
                ts   = pd.to_datetime(df[ts_col], errors="coerce")
                vals = pd.to_numeric(df[wc_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                s = pd.Series(vals.to_numpy(float), index=ts)
                s = s.loc[s.index.notna()].sort_index()
                s = s.resample("1min").sum().fillna(0.0)
                out[season] = s
            return out

        def _to_5min_energy_and_litres(season_Lpm: pd.Series, setpoint_C: float, cold_C: float) -> tuple[pd.Series, pd.Series]:
            dT = float(setpoint_C) - float(cold_C)
            L_5 = season_Lpm.resample("5min", label="right", closed="right").sum()
            kWh = L_5 * 4184.0 * dT / 3_600_000.0
            return kWh, L_5

        SEASON_BY_MONTH_SOUTH = {
            12: "Summer", 1: "Summer", 2: "Summer",
             3: "Autumn", 4: "Autumn", 5: "Autumn",
             6: "Winter", 7: "Winter", 8: "Winter",
             9: "Spring",10: "Spring",11: "Spring",
        }
        SEASON_BY_MONTH_NORTH = {
            12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Autumn",10: "Autumn",11: "Autumn",
        }

        if irr_df is None or len(irr_df) == 0:
            return _fail(household_id, "Empty irradiance DataFrame")

        idx = irr_df.index
        setpoint = float(TANK_PARAMS['setpoint'])
        inlet_temp = float(SIM_PARAMS["mains_inlet_temperature_C"])
        seasonal_Lpm = _read_seasonal_blocks(profile_path)
        seasonal_kwh: dict[str, pd.Series] = {}
        seasonal_L5:  dict[str, pd.Series] = {}
        for season, ser in seasonal_Lpm.items():
            k5, L5 = _to_5min_energy_and_litres(ser, setpoint, inlet_temp)
            seasonal_kwh[season] = k5
            seasonal_L5[season]  = L5

        city_is_north = any(city == c for c in REVERSE_SEASONS_FOR)
        season_by_month = SEASON_BY_MONTH_NORTH if city_is_north else SEASON_BY_MONTH_SOUTH

        lat_abs = abs(float(system_params.get("latitude", 0)))
        season_by_month = season_by_month.copy()

        from config import EQUATOR_LATITUDE_THRESHOLD, TROPICAL_SUBSTITUTE_SEASON
        
        if lat_abs < EQUATOR_LATITUDE_THRESHOLD:
            if DIAG_PRINTS:
                print(f"[{household_id}] Tropical location detected (|lat|={lat_abs:.2f}). "
                      f"Replacing 'Winter' profiles with '{TROPICAL_SUBSTITUTE_SEASON}'.")
            for month, season in season_by_month.items():
                if season == "Winter":
                    season_by_month[month] = TROPICAL_SUBSTITUTE_SEASON

        import numpy as _np
        reps = int(DEMAND_REPEAT_PER_SEASON)
        tiled_energy: dict[str, _np.ndarray] = {}
        tiled_litres: dict[str, _np.ndarray] = {}
        for season in ("Summer", "Autumn", "Winter", "Spring"):
            base_e = seasonal_kwh[season].reindex(seasonal_kwh[season].index, fill_value=0.0).to_numpy(float)
            base_L = seasonal_L5[season].reindex(seasonal_L5[season].index,   fill_value=0.0).to_numpy(float)
            if base_e.size == 0:
                base_e = _np.zeros(1, float); base_L = _np.zeros(1, float)
            tiled_energy[season] = _np.tile(base_e, max(1, reps))
            tiled_litres[season] = _np.tile(base_L, max(1, reps))

        month_slots: dict[tuple[int, int], list[int]] = {}
        for i, ts in enumerate(idx):
            month_slots.setdefault((ts.year, ts.month), []).append(i)

        curs_e = {s: 0 for s in tiled_energy}
        curs_L = {s: 0 for s in tiled_litres}
        demand_energy = _np.zeros(len(idx), float)
        
        for (y, m), pos in month_slots.items():
            season = season_by_month[m]
            seq_e = tiled_energy[season]; cur_e = curs_e[season]
            need = len(pos)
            if cur_e + need <= seq_e.size:
                take_e = seq_e[cur_e:cur_e+need]; cur_e += need
            else:
                r = (cur_e + need) - seq_e.size
                take_e = _np.concatenate([seq_e[cur_e:], seq_e[:r]]); cur_e = r
            demand_energy[_np.array(pos, int)] = take_e
            curs_e[season] = cur_e

        draw_kwh = pd.Series(demand_energy, index=idx, dtype=float)
        demand_kwh_sum = float(draw_kwh.sum())


        # PV system init
        try:
            cec = pvlib.pvsystem.retrieve_sam('CECMod')
            if MODULE_NAME not in cec:
                return _fail(household_id, f"CEC module not found: {MODULE_NAME}")
            module_params = cec[MODULE_NAME]
            pv = PVModule(module_params, system_params)
        except Exception as e:
            return _fail(household_id, f"Error initializing PV module: {e}")

        for col, default in (('temp_air', 20.0), ('wind_speed', 1.0)):
            if col not in irr_df.columns:
                irr_df[col] = default
        for col in ('dni', 'ghi', 'dhi'):
            if col not in irr_df.columns:
                return _fail(household_id, f"Missing irradiance column: {col}")

        irr_aligned = irr_df
        pv_kw = pv.get_power(irr_aligned).clip(lower=0.0).astype(float).to_numpy()
        on_steps_total = 0


        # Tank init & constants
        PV_START = time(5, 30)   # 05:30
        PV_END   = time(14, 30)  # 14:30
        
        # Main Tank (PV-assisted)
        tank = StratifiedTank(**TANK_PARAMS)
        tank.initialize(setpoint)
        if not hasattr(tank, 'top_temp'):    tank.top_temp = setpoint
        if not hasattr(tank, 'bottom_temp'): tank.bottom_temp = setpoint
        
        # Baseline Tank (Grid only) (for unavoidable cold draws)
        tank_baseline = StratifiedTank(**TANK_PARAMS)
        tank_baseline.initialize(setpoint)
        if not hasattr(tank_baseline, 'top_temp'):    tank_baseline.top_temp = setpoint
        if not hasattr(tank_baseline, 'bottom_temp'): tank_baseline.bottom_temp = setpoint

        dt_s = int(TANK_PARAMS['dt_s'])
        dt_h = dt_s / 3600.0
        p_target = float(TANK_PARAMS['element_rating_kw'])
        deadband = float(TANK_PARAMS.get('deadband', 3.0))
        on_thr   = setpoint - deadband
        off_thr  = setpoint
        ambient_override = TANK_PARAMS.get('ambient_override_C', None)

        heater_on = False
        heater_on_baseline = False

        # Accumulators
        total_cost_usd = 0.0
        total_solar_savings_usd = 0.0
        grid_kwh_sum = 0.0
        pv_used_kwh_sum = 0.0
        heating_event_count = 0
        solar_event_count   = 0
        solar_used_when_needed_kwh = 0.0
        grid_used_when_needed_kwh  = 0.0

        # Comfort metrics (Main Tank)
        cold_event_count = 0
        draw_event_count = 0
        in_draw_event = False
        event_volume_L = 0.0
        event_was_cold = False

        # Comfort metrics (Baseline Tank)
        cold_event_count_baseline = 0
        draw_event_count_baseline = 0
        in_draw_event_baseline = False
        event_volume_L_baseline = 0.0
        event_was_cold_baseline = False

        total_points = 0
        temp_sum     = 0.0
        pv_generated_kwh_sum = 0.0

        # simulation loop
        draw_vals = draw_kwh.to_numpy(float)

        if 'temp_air' in irr_aligned.columns:
            amb_vals = irr_aligned['temp_air'].to_numpy(float)
        else:
            amb_vals = _np.full(len(idx), 20.0, float)
            
        if ambient_override is None:
            alpha_damping = 0.05  
            amb_vals = pd.Series(amb_vals).ewm(alpha=alpha_damping, adjust=False).mean().to_numpy()

        step_heater_kw  = np.zeros(len(idx), float)
        step_heater_kwh = np.zeros(len(idx), float)
        actual_volumes_L = np.zeros(len(idx), float) 

        lat = float(system_params.get("latitude", -30.0))
        doy_array = getattr(idx, 'day_of_year', idx.dayofyear).to_numpy()
        
        if 'temp_air' in irr_aligned.columns:
            mains_temp_vals = get_nrel_mains_temperature(doy_array, irr_aligned['temp_air'], lat)
        else:
            mains_temp_vals = np.full(len(idx), float(SIM_PARAMS.get("mains_inlet_temperature_C", 15.0)))

        for i, ts in enumerate(idx):
            p_pv_kw   = float(pv_kw[i])
            demand_kwh = float(draw_vals[i])
            t_amb = float(ambient_override) if ambient_override is not None else float(amb_vals[i])
            current_inlet_temp = float(mains_temp_vals[i])

            # Main Tank Control (PV-Assisted)
            if PERMANENT_LOAD_TEST:
                heater_on = True
            else:
                if not heater_on:
                    if (getattr(tank, 'bottom_temp', None) is not None and tank.bottom_temp <= on_thr) or (tank.top_temp <= on_thr):
                        heater_on = True
                if heater_on and tank.top_temp >= off_thr:
                    heater_on = False

            pv_enable_min_kw = float(TANK_PARAMS.get("pv_enable_min_kw", 0.0))

            if heater_on:
                heating_event_count += 1
                on_steps_total += 1
                t_local = ts.time()
                pv_in_window = (PV_START <= t_local < PV_END)
                pv_available = (p_pv_kw >= pv_enable_min_kw)
                pv_allowed = pv_in_window and pv_available

                if pv_allowed:
                    p_pv_used = float(min(p_pv_kw, p_target))
                    p_grid = 0.0
                    if p_pv_used > 0:
                        solar_event_count += 1
                else:
                    p_pv_used = 0.0
                    p_grid = float(p_target)
            else:
                p_pv_used = 0.0
                p_grid = 0.0

            T_out_before = tank.top_temp
            top_T, bot_T = tank.step(p_grid + p_pv_used, demand_kwh, t_amb, current_inlet_temp, setpoint)
            tank.top_temp = float(top_T)
            tank.bottom_temp = float(bot_T)

            # Baseline Tank Control (Grid Only)
            if PERMANENT_LOAD_TEST:
                heater_on_baseline = True
            else:
                if not heater_on_baseline:
                    if (getattr(tank_baseline, 'bottom_temp', None) is not None and tank_baseline.bottom_temp <= on_thr) or (tank_baseline.top_temp <= on_thr):
                        heater_on_baseline = True
                if heater_on_baseline and tank_baseline.top_temp >= off_thr:
                    heater_on_baseline = False
            
            p_grid_baseline = float(p_target) if heater_on_baseline else 0.0
            T_out_before_baseline = tank_baseline.top_temp
            
            top_T_b, bot_T_b = tank_baseline.step(p_grid_baseline, demand_kwh, t_amb, current_inlet_temp, setpoint)
            tank_baseline.top_temp = float(top_T_b)
            tank_baseline.bottom_temp = float(bot_T_b)


            # Energy & money
            step_grid_kwh    = p_grid * dt_h
            step_pv_used_kwh = p_pv_used * dt_h
            step_heater_kw[i]  = p_grid + p_pv_used
            step_heater_kwh[i] = step_grid_kwh + step_pv_used_kwh
            grid_kwh_sum    += step_grid_kwh
            pv_used_kwh_sum += step_pv_used_kwh
            total_cost_usd          += step_grid_kwh * flat_rate_usd_per_kwh
            total_solar_savings_usd += step_pv_used_kwh * flat_rate_usd_per_kwh
            solar_used_when_needed_kwh += step_pv_used_kwh
            grid_used_when_needed_kwh  += step_grid_kwh
            pv_generated_kwh_sum += p_pv_kw * dt_h

            # Volume calculation
            T_use = 40.0 
            dT_ref = max(1e-6, setpoint - current_inlet_temp)
            vol_ref_L = demand_kwh * 3_600_000.0 / (float(TANK_PARAMS["c"]) * dT_ref)

            if T_out_before > current_inlet_temp:
                if T_out_before > T_use:
                    actual_step_vol = vol_ref_L * (T_use - current_inlet_temp) / (T_out_before - current_inlet_temp)
                else:
                    actual_step_vol = vol_ref_L
            else:
                actual_step_vol = 0.0
            
            actual_volumes_L[i] = actual_step_vol

            # Comfort (for Both tnaks)
            dT = max(1e-6, setpoint - inlet_temp) 
            step_volume_L = demand_kwh * 3_600_000.0 / (float(TANK_PARAMS["c"]) * dT)

            cold_thr_C = float(TANK_PARAMS.get("cold_event_temperature", 40.0))
            min_event_vol_L = float(TANK_PARAMS.get("cold_event_min_volume_L", 2.0))

            draw_active = step_volume_L > 0.0

            if draw_active:
                # Main Tank Update
                if not in_draw_event:
                    in_draw_event = True
                    event_volume_L = 0.0
                    event_was_cold = False
                event_volume_L += step_volume_L
                if T_out_before < cold_thr_C:
                    event_was_cold = True

                # Baseline Tank Update
                if not in_draw_event_baseline:
                    in_draw_event_baseline = True
                    event_volume_L_baseline = 0.0
                    event_was_cold_baseline = False
                event_volume_L_baseline += step_volume_L
                if T_out_before_baseline < cold_thr_C:
                    event_was_cold_baseline = True

            else:
                # Finish Main Event
                if in_draw_event:
                    if event_volume_L >= min_event_vol_L:
                        draw_event_count += 1
                        if event_was_cold:
                            cold_event_count += 1
                    in_draw_event = False
                    event_volume_L = 0.0
                    event_was_cold = False

                # Finish Baseline Event
                if in_draw_event_baseline:
                    if event_volume_L_baseline >= min_event_vol_L:
                        draw_event_count_baseline += 1
                        if event_was_cold_baseline:
                            cold_event_count_baseline += 1
                    in_draw_event_baseline = False
                    event_volume_L_baseline = 0.0
                    event_was_cold_baseline = False

            temp_sum += tank.top_temp
            total_points += 1
            
        if in_draw_event:
            min_event_vol_L = float(TANK_PARAMS.get("cold_event_min_volume_L", 2.0))
            if event_volume_L >= min_event_vol_L:
                draw_event_count += 1
                if event_was_cold:
                    cold_event_count += 1
                    
        if in_draw_event_baseline:
            min_event_vol_L = float(TANK_PARAMS.get("cold_event_min_volume_L", 2.0))
            if event_volume_L_baseline >= min_event_vol_L:
                draw_event_count_baseline += 1
                if event_was_cold_baseline:
                    cold_event_count_baseline += 1

        if DIAG_PRINTS:
            sim_steps_total = len(idx)
            dt_h = float(TANK_PARAMS['dt_s']) / 3600.0
            sim_hours_total = sim_steps_total * dt_h
            sim_range_min, sim_range_max = idx[0], idx[-1]
            pv_kwh_possible_total = float((pv_kw * dt_h).sum())
            setpoint = float(TANK_PARAMS['setpoint'])
            inlet_temp = float(SIM_PARAMS["mains_inlet_temperature_C"])
            dT = max(1e-6, setpoint - inlet_temp)
            total_litres_est = float(demand_kwh_sum * 3_600_000.0 / (4184.0 * dT))
            sim_days = max(1, (sim_range_max.normalize() - sim_range_min.normalize()).days + 1)

            print(f"[{household_id}] SIM RANGE: {sim_range_min} → {sim_range_max}")
            print(f"[{household_id}] SIM STEPS={sim_steps_total}, HOURS={sim_hours_total:.1f}")
            print(f"[{household_id}] Heater ON steps: {on_steps_total}/{sim_steps_total} → hours_on={on_steps_total * dt_h:.1f}")
            print(f"[{household_id}] PV energy available (kWh): {pv_kwh_possible_total:.1f}")
            print(f"[{household_id}] PV used (kWh): {pv_used_kwh_sum:.1f} | Grid (kWh): {grid_kwh_sum:.1f} | Total heating (kWh): {pv_used_kwh_sum + grid_kwh_sum:.1f}")
            print(f"[{household_id}] Demand (kWh): {demand_kwh_sum:.1f} | Total litres (est): {total_litres_est:,.0f} L (≈ {total_litres_est/sim_days:.1f} L/day)")

            if PERMANENT_LOAD_TEST:
                expected_kwh = float(p_target * sim_hours_total)
                actual_kwh   = float(pv_used_kwh_sum + grid_kwh_sum)
                diff_pct = (abs(actual_kwh - expected_kwh) / expected_kwh * 100.0) if expected_kwh > 0 else 0.0
                print(f"[{household_id}] PERM-LOAD: expected {expected_kwh:.1f} kWh, actual {actual_kwh:.1f} kWh (Δ={diff_pct:.2f}%)")

        # KPIs
        total_heating_kwh = pv_used_kwh_sum + grid_kwh_sum
        solar_heating_energy_fraction = (pv_used_kwh_sum / total_heating_kwh) if total_heating_kwh > 0 else 0.0
        solar_heating_event_fraction  = (solar_event_count / heating_event_count) if heating_event_count > 0 else 0.0

        cost_without_solar_usd = total_cost_usd + total_solar_savings_usd
        savings_pct = (total_solar_savings_usd / cost_without_solar_usd * 100.0) if cost_without_solar_usd > 0 else 0.0

        total_heating_kwh_when_needed = solar_used_when_needed_kwh + grid_used_when_needed_kwh
        solar_fraction_heating_pct = (solar_used_when_needed_kwh / total_heating_kwh_when_needed * 100) if total_heating_kwh_when_needed > 0 else 0.0
        
        pv_utilization_pct = (solar_used_when_needed_kwh / pv_generated_kwh_sum * 100) if pv_generated_kwh_sum > 0 else 0

        total_L = sum(actual_volumes_L)
        num_days = len(idx) / (24 * (60/5))
        avg_daily_L = total_L / num_days if num_days > 0 else 0

        tank_vol = float(TANK_PARAMS.get('volume_l', 150))
        usage_pct_of_tank_vol = (avg_daily_L / tank_vol) * 100 if tank_vol > 0 else 0

        # Calculation Cold Draw
        solar_induced_cold_events = max(0, cold_event_count - cold_event_count_baseline)
        
        # Calculate percentage values
        pct_total = (cold_event_count / draw_event_count * 100.0) if draw_event_count > 0 else 0.0
        pct_unavoidable = (cold_event_count_baseline / draw_event_count * 100.0) if draw_event_count > 0 else 0.0
        pct_solar_induced = (solar_induced_cold_events / draw_event_count * 100.0) if draw_event_count > 0 else 0.0

        kpis = {
            'annual_grid_kwh': grid_kwh_sum,
            'annual_solar_kwh': pv_used_kwh_sum,
            'annual_demand_kwh': demand_kwh_sum,
            'solar_fraction': (pv_used_kwh_sum / total_heating_kwh) if total_heating_kwh > 0 else 0.0,
            'solar_heating_energy_fraction': solar_heating_energy_fraction,
            'solar_heating_event_fraction':  solar_heating_event_fraction,
            
            # Events (value)
            'draw_event_count': draw_event_count,
            'cold_event_count_TOTAL': cold_event_count,
            'cold_event_count_UNAVOIDABLE': cold_event_count_baseline,
            'cold_event_count_SOLAR_INDUCED': solar_induced_cold_events,
            
            # Events (pct)
            'cold_draw_pct': pct_total, 
            'cold_draw_pct_total': pct_total,
            'cold_draw_pct_unavoidable': pct_unavoidable,
            'cold_draw_pct_solar_induced': pct_solar_induced,
            
            'pv_utilization_pct_for_the_hours_were_the_sun_is_shining': pv_utilization_pct,
            'solar_fraction_heating_pct': solar_fraction_heating_pct,
            'avg_temp': (temp_sum / total_points) if total_points > 0 else np.nan,                 
            'solar_used_when_needed_kwh': solar_used_when_needed_kwh,
            'grid_used_when_needed_kwh':  grid_used_when_needed_kwh,
            'annual_solar_savings_USD': total_solar_savings_usd,
            'cost_without_solar_USD':   cost_without_solar_usd,
            'savings_percentage':       savings_pct,
            'pv_generated_kwh': pv_generated_kwh_sum,
            'avg_daily_litres': avg_daily_L,
        }

        return {
            'profile': household_id,
            'kpis': kpis,
            'cost_USD': total_cost_usd,
            'timeseries': {
            'heater_kw':      pd.Series(step_heater_kw,  index=idx),
            'heater_kwh_5min':pd.Series(step_heater_kwh, index=idx),
            'draw_kwh_5min':  draw_kwh, 
            'water_L_5min':   pd.Series(actual_volumes_L, index=idx),
        },
            'success': True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _fail(os.path.basename(profile_path), str(e))

                    
def _fail(household: str, msg: str):
    return {
        'profile': household,
        'kpis': {},
        'cost_USD': 0.0,
        'success': False,
        'error': msg,
    }
