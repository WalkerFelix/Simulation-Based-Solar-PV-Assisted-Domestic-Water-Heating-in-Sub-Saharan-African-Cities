"""
tank.py

Two-node stratified electric water heater tank.
"""
import numpy as np

class StratifiedTank:
    """
    Two-node stratified tank.

    Attributes
    ----------
    top_temp : float
    bottom_temp : float
    volume : float
    c : float
        Specific heat capacity of water (J/kg.K).
    rho : float
        Water density (kg/m3).
    R_th : float
        Thermal resistance to ambient (K/W).
    element_rating : float
        Heating element power (kW).
    dt : float
        Timestep (s).
    """
    def __init__(self, volume_l: float, c: float, rho: float,
                 R_th: float, element_rating_kw: float, dt_s: float, **kwargs):
        self.volume = float(volume_l) / 1000  
        self.c = float(c)
        self.rho = float(rho)
        self.R_th = float(R_th)               
        self.R_th_ref   = self.R_th           
        self.deltaT_ref = 40.0                
        self.alpha_UA   = 0.00              
        self.element_rating = float(element_rating_kw)
        self.dt = float(dt_s)
        
        # Split volume equally between top and bottom nodes
        self.v_top = self.volume / 2
        self.v_bot = self.volume / 2
        
        # Mass of water in each node
        self.mass_top = self.rho * self.v_top
        self.mass_bot = self.rho * self.v_bot
        
        # Initialize temperatures
        self.top_temp = None
        self.bottom_temp = None

    def initialize(self, T0: float):
        """Initialize both nodes to T0 (°C)."""
        self.top_temp = float(T0)
        self.bottom_temp = float(T0)

    def step(self, power_kw: float, draw_kwh: float, T_amb: float, T_inlet: float, T_setpoint: float):
        """
        Advance the tank state by one timestep with mixing valve logic.

        Args:
            power_kw: Heating power applied to the element (kW).
            draw_kwh: Energy demand if delivered at T_setpoint (reference from profile).
            T_amb: Ambient temperature (degC).
            T_inlet: Mains inlet water temperature (degC).
            T_setpoint: Original thermostat setpoint temperature (degC).
        """
        power_kw = float(power_kw)
        draw_kwh = float(draw_kwh)
        T_amb = float(T_amb)
        
        # 1) Calculate internal energy changes (Heating and Losses)
        Q_in = power_kw * 1000 * self.dt  # J
        
        # Calculate UA (Conductance) per node.
        # R_th is the total resistance of the whole tank. UA_total = 1 / R_th.
        # Assuming the tank is split 50/50, each node has roughly half the surface area.
        # UA_node approx UA_total / 2.
        UA_total_ref = 1.0 / self.R_th_ref
        UA_node_ref  = UA_total_ref / 2.0

        # Heat transfer of the Top Node 
        deltaT_top = self.top_temp - T_amb
        UA_top_dyn = UA_node_ref * (1 + self.alpha_UA * (abs(deltaT_top) - self.deltaT_ref) / self.deltaT_ref)
        Q_loss_top = UA_top_dyn * deltaT_top * self.dt

        # Heat transfer of the Bottom Node
        deltaT_bot = self.bottom_temp - T_amb
        UA_bot_dyn = UA_node_ref * (1 + self.alpha_UA * (abs(deltaT_bot) - self.deltaT_ref) / self.deltaT_ref)
        Q_loss_bot = UA_bot_dyn * deltaT_bot * self.dt
        
        # Element Heat Distribution (30% convection to top, 70% direct to bottom)
        Q_heat_top = Q_in * 0.3 
        Q_heat_bot = Q_in * 0.7 
        
        # Update Energies (Subtracting Q_loss: if Q_loss is negative, we add energy)
        E_top = self.mass_top * self.c * self.top_temp + Q_heat_top - Q_loss_top
        E_bot = self.mass_bot * self.c * self.bottom_temp + Q_heat_bot - Q_loss_bot
        
        # 2) Mixing Valve/Draw Logic 
        # T_use is the target temperature at the tap 40°C
        T_use = 40.0 
        T_out = float(self.top_temp)
        E_draw_ref = float(draw_kwh) * 3_600_000.0  

        # Calculate actual energy to be removed from the tank
        if T_out > T_inlet:
            if T_out > T_use:
                # Mixing logic: energy removal is scaled to T_use instead of T_out
                mixing_factor = (T_use - T_inlet) / (T_out - T_inlet)
                E_remove = E_draw_ref * mixing_factor
            else:
                # Tank is cooler than T_use: scale energy removal by available T_out
                efficiency_factor = (T_out - T_inlet) / (float(T_setpoint) - T_inlet)
                E_remove = E_draw_ref * efficiency_factor
        else:
            E_remove = 0.0

        if E_remove > 0.0:
            if E_remove <= E_top:
                E_top -= E_remove
            else:
                rem = E_remove - E_top
                E_top = 0.0
                E_bot = max(0.0, E_bot - rem)
        
        # 3) Update temperatures
        if self.mass_top * self.c > 0:
            self.top_temp = max(0, E_top / (self.mass_top * self.c))
        else:
            self.top_temp = T_amb
            
        if self.mass_bot * self.c > 0:
            self.bottom_temp = max(0, E_bot / (self.mass_bot * self.c))
        else:
            self.bottom_temp = T_amb
        
        self.top_temp = min(float(self.top_temp), 90.0)
        self.bottom_temp = min(float(self.bottom_temp), 90.0)
        
        return float(self.top_temp), float(self.bottom_temp)