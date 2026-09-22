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
        self.element_rating = float(element_rating_kw)
        self.dt = float(dt_s)
        
        # UA per node: total tank conductance split equally over two nodes
        UA_total = 1.0 / self.R_th
        self.UA_node = UA_total / 2.0

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

    def step(self, power_kw: float, draw_volume_l: float, T_amb: float, T_inlet: float, T_setpoint: float):
        """
        Advance the tank state by one timestep with dynamic mixing valve logic.

        Args:
            power_kw: Heating power applied to the element (kW).
            draw_volume_l: Volume of water demanded at the tap at T_use (Liters).
            T_amb: Ambient temperature (degC).
            T_inlet: Dynamic mains inlet water temperature (degC).
            T_setpoint: Original thermostat setpoint temperature (degC).
        """
        power_kw = float(power_kw)
        draw_volume_l = float(draw_volume_l)
        T_amb = float(T_amb)
        T_in = float(T_inlet)
        
        # 1) Calculate internal energy changes (Heating and Losses)
        Q_in = power_kw * 1000 * self.dt  # J
        
        # Heat losses per node: Q_loss = UA_node * (T_node - T_amb) * dt
        Q_loss_top = self.UA_node * (self.top_temp - T_amb) * self.dt
        Q_loss_bot = self.UA_node * (self.bottom_temp - T_amb) * self.dt
        
        # Element Heat Distribution (30% convection to top, 70% direct to bottom)
        Q_heat_top = Q_in * 0.3 
        Q_heat_bot = Q_in * 0.7 
        
        # Update Energies (Subtracting Q_loss: if Q_loss is negative, we add energy)
        E_top = self.mass_top * self.c * self.top_temp + Q_heat_top - Q_loss_top
        E_bot = self.mass_bot * self.c * self.bottom_temp + Q_heat_bot - Q_loss_bot
        
        # 2) Mixing Valve/Draw Logic 
        T_use = 40.0 
        T_out = float(self.top_temp)

        # Mass of the drawn water in kg (assuming 1L = approx 1kg based on rho)
        m_draw = draw_volume_l * (self.rho / 1000.0)

        # Calculate actual energy to be removed from the tank based on dynamic T_inlet
        if T_out > T_in:
            if T_out >= T_use:
                # Mixing mode: We only remove the energy needed to heat m_draw from T_in to T_use.
                # The physical volume pulled from the tank will be less than draw_volume_l
                # due to mixing with cold water at T_in at the valve.
                E_remove = m_draw * self.c * (T_use - T_in)
            else:
                # Tank is cooler than T_use: Valve is fully open to hot side.
                # User gets T_out instead of T_use.
                E_remove = m_draw * self.c * (T_out - T_in)
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

        # 4) Inversion protection:
        # Hot water rises - bottom_temp can never exceed top_temp in a stable stratified tank.
        # If the bottom node is hotter than the top (e.g. due to direct element heating),
        # natural convection immediately mixes the two nodes to a uniform temperature.
        if self.bottom_temp > self.top_temp:
            T_mix = (
                (self.mass_top * self.top_temp + self.mass_bot * self.bottom_temp)
                / (self.mass_top + self.mass_bot)
            )
            self.top_temp    = T_mix
            self.bottom_temp = T_mix
        
        return float(self.top_temp), float(self.bottom_temp)
