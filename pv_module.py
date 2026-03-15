import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, calcparams_cec
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

class PVModule:
    """
    PV wrapper using pvlib mit Unterstützung für resistive Lastanpassung.
    """
    def __init__(self, module_params: dict, system_params: dict):
        self.module = module_params

        tilt = system_params['tilt']
        azimuth = system_params['azimuth']
        lat = system_params['latitude']
        lon = system_params['longitude']
        tz = system_params['timezone']
        racking_model = system_params.get('racking_model', 'open_rack_glass_glass')

        self.n_series = int(system_params['num_panels'])
        if self.n_series < 1:
            raise ValueError("num_panels must be >= 1")
        strings = 1  

        # Temperaturmodell-Parameter (SAPM)
        if racking_model not in TEMPERATURE_MODEL_PARAMETERS['sapm']:
            print(f"Warning: Invalid racking model '{racking_model}', using 'open_rack_glass_glass'.")
            racking_model = 'open_rack_glass_glass'
        temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm'][racking_model]

        # Helfer zum Suchen von Parametern (da Namen variieren können)
        def _get_param(keys):
            for k in keys:
                if k in self.module and pd.notnull(self.module[k]):
                    return float(self.module[k])
            return None

        vmp = _get_param(['V_mp_ref', 'Vmp', 'V_mp'])
        imp = _get_param(['I_mp_ref', 'Imp', 'I_mp'])
        
        if vmp is None or imp is None:
            raise ValueError("CEC module params missing Vmp/Imp (need V_mp_ref/I_mp_ref or Vmp/Imp).")

        # Wir speichern die CEC-spezifischen Keys für später (get_power_resistive)
        self.cec_params = {
            'alpha_sc': _get_param(['alpha_sc', 'alpha_Isc']),
            'a_ref':    _get_param(['a_ref']),
            'I_L_ref':  _get_param(['I_L_ref', 'I_l_ref']),
            'I_o_ref':  _get_param(['I_o_ref']),
            'R_sh_ref': _get_param(['R_sh_ref', 'R_sh', 'Rsh']),
            'R_s_ref':  _get_param(['R_s_ref', 'R_s', 'Rs']),
            'Adjust':   _get_param(['Adjust', 'adjust'])
        }

        # Check ob alle kritischen CEC Parameter da sind
        missing = [k for k, v in self.cec_params.items() if v is None]
        if missing:
            raise KeyError(f"Missing critical CEC parameters in module data: {missing}")

        pmp_module_w = vmp * imp                      
        pdc0 = pmp_module_w * self.n_series * strings      
        inverter_params = {
            'pdc0': float(pdc0),
            'eta_inv_nom': 1.0,
            'eta_inv_ref': 1.0,
        }
        print(f"Initializing PVSystem with tilt={tilt}°, azimuth={azimuth}°, location=({lat}, {lon}), pdc0={pdc0:.1f} W")

        pv_sys = PVSystem(
            module_parameters=self.module,
            inverter_parameters=inverter_params,
            surface_tilt=tilt,
            surface_azimuth=azimuth,
            modules_per_string=self.n_series,
            strings_per_inverter=strings,
            temperature_model_parameters=temp_params,
        )

        self.location = Location(latitude=lat, longitude=lon, tz=tz)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mc = ModelChain(
                pv_sys,
                self.location,
                aoi_model='no_loss',
                spectral_model='no_loss',
                temperature_model='sapm',
                dc_model='cec',
                ac_model='pvwatts',   
            )
        print("ModelChain initialized successfully")

    def get_power_resistive(self, meteo: pd.DataFrame, R_total: float) -> pd.Series:
        orig_idx = meteo.index
        weather = meteo[['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']].copy()
        for c in weather.columns:
            weather[c] = pd.to_numeric(weather[c], errors='coerce').fillna(0)
        if weather.index.tz is None:
            weather.index = weather.index.tz_localize(self.location.tz)
        
        self.mc.run_model(weather)
        
        effective_irrad = self.mc.results.effective_irradiance
        if hasattr(self.mc.results, 'cell_temperature'):
            cell_temp = self.mc.results.cell_temperature
        else:
            cell_temp = weather['temp_air'] 

        # Nutze die sicher extrahierten cec_params
        p = self.cec_params
        res = calcparams_cec(
            effective_irrad, cell_temp,
            p['alpha_sc'], p['a_ref'], p['I_L_ref'], p['I_o_ref'],
            p['R_sh_ref'], p['R_s_ref'], p['Adjust']
        )
        IL, Io, Rs, Rsh, nNsVth = res

        R_per_module = R_total / self.n_series

        def solve_v(il, io, rs, rsh, nnsvth, r_load):
            if il <= 0.001: return 0.0
            f = lambda v: il - io * (np.exp((v + (v/r_load)*rs) / nnsvth) - 1) - (v + (v/r_load)*rs) / rsh - (v/r_load)
            try:
                # brentq braucht Vorzeichenwechsel. Falls f(0) und f(150) gleiches Vorzeichen:
                if f(0) * f(150) > 0: return 0.0
                return brentq(f, 0, 150)
            except:
                return 0.0

        v_results = []
        IL_vals, Io_vals, Rs_vals, Rsh_vals, nNsVth_vals = [x.to_numpy() for x in [IL, Io, Rs, Rsh, nNsVth]]
        irr_vals = effective_irrad.to_numpy()

        for i in range(len(IL_vals)):
            if irr_vals[i] < 1:
                v_results.append(0.0)
            else:
                v_op = solve_v(IL_vals[i], Io_vals[i], Rs_vals[i], Rsh_vals[i], nNsVth_vals[i], R_per_module)
                v_results.append(v_op)

        v_modul = np.array(v_results)
        i_modul = np.where(v_modul > 0, v_modul / R_per_module, 0.0)
        p_total_w = v_modul * self.n_series * i_modul
        
        out = pd.Series(p_total_w / 1000.0, index=orig_idx)
        return out.clip(lower=0.0).fillna(0.0)

    def get_power(self, meteo: pd.DataFrame) -> pd.Series:
        try:
            required = ['dni', 'ghi', 'dhi', 'temp_air', 'wind_speed']
            orig_idx = meteo.index  
            weather = meteo[required].copy()
            for c in required:
                weather[c] = pd.to_numeric(weather[c], errors='coerce').fillna(0)
            if weather.index.tz is None:
                weather.index = weather.index.tz_localize(self.location.tz)
            self.mc.run_model(weather)
            dc = getattr(self.mc.results, 'dc', None)
            if isinstance(dc, pd.DataFrame) and 'p_mp' in dc:
                pmp = dc['p_mp']
            else:
                pmp = getattr(dc, 'p_mp', pd.Series(0.0, index=weather.index))
            if pmp.index.tz is not None and orig_idx.tz is None:
                pmp.index = pmp.index.tz_localize(None)
            return (pmp / 1000.0).reindex(orig_idx).clip(lower=0.0).fillna(0.0)
        except Exception as e:
            print(f"Error computing MPPT power: {e}")
            return pd.Series(0.0, index=meteo.index)
