#!/usr/bin/env python
# coding: utf-8

# # HEMS Simulation Implementation
# This notebook implements the Home Energy Management System (HEMS) simulation architecture described in `agents.md`.
# 
# ## Components:
# 1.  **The Plant**: Simulation Environment (Physics, State, Economics).
# 2.  **Optimizers**: Hourly Agents (Residual, Rule-Based, MPC).
# 3.  **Controllers**: Real-Time Agents (Pass-Through, Interpolating, S2).
# 4.  **Orchestration**: Simulation Loop.
# 

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')


# In[64]:


import subprocess
import time

def check_docker():
    # Auto-launch Docker Compose for URBS Solver
    print("Checking Docker Compose status...")
    cmd_status = ["docker", "compose", "ps", "-q"]
    try:
        if not subprocess.run(cmd_status, capture_output=True, text=True).stdout.strip():
            print("Starting Docker Compose...")
            subprocess.run(["docker", "compose", "up", "-d"], check=True)
            print("Docker Compose started. Waiting for services to initialize...")
            time.sleep(5) # Give it a moment
        else:
            print("Docker Compose already running.")
    except Exception as e:
        print(f"Error managing Docker: {e}")
        print("Please ensure Docker is running and docker-compose.yaml is present.")


# In[65]:


@dataclass
class HEMSConfig:
    # Time settings
    days: int = 3
    minutes_per_hour: int = 60

    # Battery specs
    battery_capacity_kwh: float = 13.5
    battery_max_power_kw: float = 5.0
    battery_efficiency: float = 0.95
    initial_soc: float = 0.5

    # Heat Pump / Storage specs
    heat_storage_capacity_kwh: float = 15.0  # Thermal capacity
    heat_pump_max_power_kw: float = 4.0      # Electrical input
    cop: float = 3.5                         # Coefficient of Performance
    heat_loss_per_hour_kwh: float = 0.2
    initial_heat_soc: float = 0.5

    # Grid
    grid_import_cost: float = 0.30  # $/kWh
    grid_export_revenue: float = 0.08 # $/kWh

    @property
    def total_minutes(self):
        return self.days * 24 * 60


# In[66]:


class HEMSPlant:
    def __init__(self, config: HEMSConfig):
        self.config = config
        self.time_step = 0

        # State
        self.battery_soc_kwh = config.battery_capacity_kwh * config.initial_soc
        self.heat_storage_kwh = config.heat_storage_capacity_kwh * config.initial_heat_soc

        # Generate Scenarios
        self._generate_scenarios()

        # History
        self.history = {
            'battery_soc': [],
            'heat_storage_soc': [],
            'grid_power': [],
            'cost': [],
            'solar_generated': [],
            'load_consumed': [],
            'heat_demand_met': [],
            'battery_power': [],
            'heat_pump_power': [],
            'import_price': [],
            'export_price': []
        }

    def _generate_scenarios(self):
        # Time index
        t = np.linspace(0, self.config.days * 24, self.config.total_minutes)

        # Solar: Peak at noon + random clouds
        self.solar_profile = np.maximum(0, 5 * np.sin(2 * np.pi * (t - 6) / 24))
        self.solar_profile = np.maximum(0, self.solar_profile - 0.3 * np.random.weibull(0.5, size=len(t)))

        # Electrical Load: Morning/Evening peaks
        load = 2 + np.cos(4 * np.pi * (t - 18) / 24) + \
                0.8 * np.cos(2 * np.pi * (t - 14) / 24)
        self.load_profile = np.maximum(0.5, load)

        # Heat Demand: Base + Rectangular Morning/Evening
        base_heat = 0.2
        self.heat_demand_profile = np.ones_like(t) * base_heat

        # Morning Rectangular (e.g., 6:00 - 9:00)
        mask_morning = ((t % 24) >= 6) & ((t % 24) < 9)
        self.heat_demand_profile[mask_morning] += 2.0 # Add 2kW

        # Evening Rectangular (e.g., 18:00 - 22:00)
        mask_evening = ((t % 24) >= 18) & ((t % 24) < 22)
        self.heat_demand_profile[mask_evening] += 2.0 # Add 2kW

        # Prices (Simple TOU)
        # High price 16:00-21:00
        self.import_price_profile = np.ones_like(t) * self.config.grid_import_cost
        mask_peak = ((t % 24) >= 16) & ((t % 24) < 21)
        self.import_price_profile[mask_peak] = 0.45 # Peak price

        self.export_price_profile = np.ones_like(t) * self.config.grid_export_revenue

    def get_observation(self):
        """Returns current state and forecasts"""
        return {
            'time_step': self.time_step,
            'battery_soc_kwh': self.battery_soc_kwh,
            'heat_storage_kwh': self.heat_storage_kwh,
            'current_solar': self.solar_profile[self.time_step],
            'current_load': self.load_profile[self.time_step],
            'current_heat_demand': self.heat_demand_profile[self.time_step]
        }

    def get_forecast(self, horizon_hours: int):
        """Returns perfect foresight for the next horizon_hours"""
        start = self.time_step
        end = min(start + horizon_hours * 60, self.config.total_minutes)

        return {
            'solar': self.solar_profile[start:end],
            'load': self.load_profile[start:end],
            'heat_demand': self.heat_demand_profile[start:end],
            'import_price': self.import_price_profile[start:end],
            'export_price': self.export_price_profile[start:end]
        }

    def step(self, action: Dict):
        """
        Executes one minute of simulation.
        Action keys:
        - battery_power_kw: (+) Charge, (-) Discharge
        - heat_pump_power_kw: (+) Consumption
        - pv_curtailment_limit_kw: Max allowed PV
        """
        if self.time_step >= self.config.total_minutes:
            return None

        # 1. Unpack Action & Apply Limits
        # Battery
        bat_power = np.clip(action.get('battery_power_kw', 0), 
                            -self.config.battery_max_power_kw, 
                            self.config.battery_max_power_kw)

        # Heat Pump
        hp_power = np.clip(action.get('heat_pump_power_kw', 0), 
                           0, 
                           self.config.heat_pump_max_power_kw)

        # PV Limit
        pv_limit = action.get('pv_curtailment_limit_kw', 999.0)

        # 2. Get Environment Data
        potential_solar = self.solar_profile[self.time_step]
        load = self.load_profile[self.time_step]
        heat_demand = self.heat_demand_profile[self.time_step]

        # 3. Physics & State Updates (Energy = Power * Time)
        dt_hours = 1.0 / 60.0

        # --- Electrical Balance ---
        # PV Generation
        actual_solar = min(potential_solar, pv_limit)

        # Battery Dynamics
        # Check SoC limits
        if bat_power > 0: # Charging
            max_charge = (self.config.battery_capacity_kwh - self.battery_soc_kwh) / dt_hours / self.config.battery_efficiency
            bat_power = min(bat_power, max_charge)
            energy_to_battery = bat_power * dt_hours * self.config.battery_efficiency
            self.battery_soc_kwh += energy_to_battery
        else: # Discharging
            max_discharge = self.battery_soc_kwh / dt_hours * self.config.battery_efficiency # Approximation for discharge eff
            # Usually discharge efficiency is applied to output: Output = Stored * Eff? Or Stored = Output / Eff?
            # Let's assume: Energy extracted from battery = Power * dt / Eff
            max_discharge_power = (self.battery_soc_kwh / dt_hours) * self.config.battery_efficiency
            bat_power = max(bat_power, -max_discharge_power)
            energy_from_battery = (-bat_power) * dt_hours / self.config.battery_efficiency
            self.battery_soc_kwh -= energy_from_battery

        # Heat Pump Dynamics
        # Check Heat Storage limits
        # Thermal Power Produced
        thermal_power = hp_power * self.config.cop

        # Check if storage is full
        max_thermal_input = (self.config.heat_storage_capacity_kwh - self.heat_storage_kwh) / dt_hours
        if thermal_power > max_thermal_input:
            thermal_power = max_thermal_input
            hp_power = thermal_power / self.config.cop # Reduce electrical input

        # Update Heat Storage
        heat_added = thermal_power * dt_hours
        heat_removed = heat_demand * dt_hours
        heat_loss = self.config.heat_loss_per_hour_kwh * dt_hours

        self.heat_storage_kwh += (heat_added - heat_removed - heat_loss)
        self.heat_storage_kwh = np.clip(self.heat_storage_kwh, 0, self.config.heat_storage_capacity_kwh)

        # Grid Balance
        # Net Load = Load + HP + Battery_Charge - Battery_Discharge - Solar
        # Positive = Import, Negative = Export
        net_load = load + hp_power + bat_power - actual_solar

        # 4. Economics
        import_price = self.import_price_profile[self.time_step]
        export_price = self.export_price_profile[self.time_step]

        cost = 0.0
        if net_load > 0:
            cost = net_load * dt_hours * import_price
        else:
            cost = net_load * dt_hours * export_price # Revenue (negative cost)

        # 5. Logging
        self.history['battery_soc'].append(self.battery_soc_kwh)
        self.history['heat_storage_soc'].append(self.heat_storage_kwh)
        self.history['grid_power'].append(net_load)
        self.history['cost'].append(cost)
        self.history['solar_generated'].append(actual_solar)
        self.history['load_consumed'].append(load)
        self.history['heat_demand_met'].append(heat_demand)
        self.history['battery_power'].append(bat_power)
        self.history['heat_pump_power'].append(hp_power)
        self.history['import_price'].append(import_price)
        self.history['export_price'].append(export_price)

        self.time_step += 1

        return {
            'cost': cost,
            'net_load': net_load,
            'battery_soc': self.battery_soc_kwh,
            'heat_storage_soc': self.heat_storage_kwh
        }


# In[67]:


# --- Real-Time Controllers (Minute Level) ---

class BaseDeviceController:
    def __init__(self, initial_setpoint: float = 0.0):
        self.last_setpoint = initial_setpoint

    def get_action(self, setpoint: Optional[float], observation: Dict, config: HEMSConfig) -> float:
        if setpoint is not None:
            self.last_setpoint = setpoint
        return self.last_setpoint

class PassThroughDeviceController(BaseDeviceController):
    """Simply applies the setpoint constant for every minute."""
    pass

class InterpolatingDeviceController(BaseDeviceController):
    """
    Interpolates between the previous setpoint and the current one over the course of the hour.
    """
    def __init__(self, initial_setpoint: float = 0.0):
        super().__init__(initial_setpoint)
        self.target_setpoint = initial_setpoint
        self.steps_since_change = 0
        self.interpolation_steps = 60 # Interpolate over 1 hour

    def get_action(self, setpoint: Optional[float], observation: Dict, config: HEMSConfig) -> float:
        # Check if setpoint has changed
        if setpoint is not None and setpoint != self.target_setpoint:
            if observation['time_step'] == 0:
                self.last_setpoint = setpoint
            else:
                self.last_setpoint = self.target_setpoint
            
            self.target_setpoint = setpoint
            self.steps_since_change = 0
        
        # Calculate interpolation
        if self.steps_since_change < self.interpolation_steps:
            alpha = self.steps_since_change / self.interpolation_steps
            action = self.last_setpoint + alpha * (self.target_setpoint - self.last_setpoint)
            self.steps_since_change += 1
            return action
        else:
            return self.target_setpoint

class S2EnvelopeController(BaseDeviceController):
    """
    S2 Inspired Envelope Controller.
    - Maintains a 10% safety buffer (10% - 90% SoC).
    - Applies a 10% tolerance band (absolute % of max power) around the setpoint.
    """
    def __init__(self, storage_key: str, capacity_attr: str, max_power_attr: str, initial_setpoint: float = 0.0):
        super().__init__(initial_setpoint)
        self.storage_key = storage_key
        self.capacity_attr = capacity_attr
        self.max_power_attr = max_power_attr

    def get_action(self, setpoint: Optional[float], observation: Dict, config: HEMSConfig) -> float:
        action = super().get_action(setpoint, observation, config)
        
        # 1. Apply 10% Tolerance Band (Absolute of Total Power)
        max_power = getattr(config, self.max_power_attr)
        deviation = max_power * 0.10
        noise = np.random.uniform(-deviation, deviation)
        action = action + noise

        # 2. Check Safety Limits (10% Buffer)
        stored = observation[self.storage_key]
        capacity = getattr(config, self.capacity_attr)
        
        # Prevent charging/heating if > 90%
        if stored >= capacity * 0.90:
            if action > 0:
                action = 0.0
        
        # Prevent discharging if < 10%
        if stored <= capacity * 0.10:
            if action < 0:
                action = 0.0
        
        return action

# In[68]:


# --- Optimizers (Hourly Level) ---

class BaseOptimizer:
    def get_setpoints(self, observation: Dict, forecast: Dict, config: HEMSConfig) -> Dict:
        """
        Returns:
            {
                'battery_power_kw': float,
                'heat_pump_power_kw': float,
                'pv_curtailment_limit_kw': float
            }
        """
        raise NotImplementedError

class ResidualOptimizer(BaseOptimizer):
    """
    Reactive: 
    - Solar > Load: Charge Battery
    - Solar < Load: Discharge Battery
    - Heat Pump: Run if heat needed (simple logic)
    """
    def get_setpoints(self, observation: Dict, forecast: Dict, config: HEMSConfig) -> Dict:
        # Average forecast for the next hour
        avg_solar = np.mean(forecast['solar'][:60])
        avg_load = np.mean(forecast['load'][:60])

        net_load = avg_load - avg_solar

        bat_power = 0
        if net_load < 0:
            # Excess solar -> Charge
            bat_power = min(-net_load, config.battery_max_power_kw)
        else:
            # Deficit -> Discharge
            bat_power = max(-net_load, -config.battery_max_power_kw)

        # Simple Heat Logic: Keep it half full
        hp_power = 0
        if observation['heat_storage_kwh'] < config.heat_storage_capacity_kwh * 0.5:
            hp_power = config.heat_pump_max_power_kw

        return {
            'battery_power_kw': bat_power,
            'heat_pump_power_kw': hp_power,
            'pv_curtailment_limit_kw': 999.0
        }

class RuleBasedOptimizer(BaseOptimizer):
    """
    Prioritized Decision Tree:
    1. Critical Heat
    2. Excess Solar -> Heat
    3. Excess Solar -> Battery
    """
    def get_setpoints(self, observation: Dict, forecast: Dict, config: HEMSConfig) -> Dict:
        avg_solar = np.mean(forecast['solar'][:60])
        avg_load = np.mean(forecast['load'][:60])

        excess_solar = max(0, avg_solar - avg_load)

        hp_power = 0
        bat_power = 0

        # 1. Critical Heat
        if observation['heat_storage_kwh'] < config.heat_storage_capacity_kwh * 0.2:
            hp_power = config.heat_pump_max_power_kw
            # Consume solar for this first
            excess_solar = max(0, excess_solar - hp_power)

        # 2. Excess Solar -> Heat
        elif excess_solar > 0 and observation['heat_storage_kwh'] < config.heat_storage_capacity_kwh * 0.9:
            # Use excess solar to run HP
            possible_hp = min(excess_solar, config.heat_pump_max_power_kw)
            hp_power = possible_hp
            excess_solar -= hp_power

        # 3. Excess Solar -> Battery
        if excess_solar > 0:
            bat_power = min(excess_solar, config.battery_max_power_kw)

        # 4. If no excess solar, but battery has charge and load exists, discharge
        if avg_solar < avg_load:
             deficit = avg_load - avg_solar
             # If we are running HP from grid, that adds to deficit
             deficit += hp_power 
             bat_power = max(-deficit, -config.battery_max_power_kw)

        return {
            'battery_power_kw': bat_power,
            'heat_pump_power_kw': hp_power,
            'pv_curtailment_limit_kw': 999.0
        }

class MPCOptimizer(BaseOptimizer):
    """
    Uses external solver (URBS) to optimize.
    """
    def __init__(self, url="http://localhost:5000/simulate"):
        self.url = url

    def get_setpoints(self, observation: Dict, forecast: Dict, config: HEMSConfig) -> Dict:
        # 1. Prepare Data
        # Forecast comes in minutes, we need to aggregate to hours for URBS (or use minutes if URBS supports it, but usually it's hourly)
        # For simplicity and speed, let's aggregate to hourly mean

        # Helper to aggregate minute array to hourly array
        def to_hourly(arr):
            return np.mean(arr.reshape(-1, 60), axis=1)

        # Check if we have enough data for at least 2 hours (URBS needs >1 timestep to avoid division by zero)
        if len(forecast['solar']) < 120:
             # Fallback to RuleBased if horizon is too short (end of simulation)
             rb = RuleBasedOptimizer()
             return rb.get_setpoints(observation, forecast, config)

        solar_hourly = to_hourly(forecast['solar'])
        load_hourly = to_hourly(forecast['load'])
        heat_demand_hourly = to_hourly(forecast['heat_demand'])
        price_hourly = to_hourly(forecast['import_price'])
        sell_price_hourly = to_hourly(forecast['export_price'])

        timesteps = len(solar_hourly)

        # Normalize Solar for "SupIm" (Supply Intermittent)
        max_solar = np.max(solar_hourly)
        if max_solar == 0: max_solar = 1.0
        norm_solar = (solar_hourly / max_solar).tolist()

        # Clip Initial SoCs to [0, 1] to avoid infeasibility
        init_bat = np.clip(observation['battery_soc_kwh'] / config.battery_capacity_kwh, 0.0, 1.0)
        init_heat = np.clip(observation['heat_storage_kwh'] / config.heat_storage_capacity_kwh, 0.0, 1.0)

        # 2. Construct Payload
        payload = {
            "site": {
                "Main": {
                    "area": 100,
                    "process": {
                        "Purchase": {
                            "wacc": 0, "cap-lo": 0, "cap-up": 1000, "fix-cost": 0, "inst-cap": 1000, "inv-cost": 0, "max-grad": "inf", "var-cost": 0,
                            "commodity": {"Elec": {"ratio": 1, "Direction": "Out", "ratio-min": 1}, "Elec buy": {"ratio": 1, "Direction": "In", "ratio-min": 1}},
                            "description": "Buy electricity", "depreciation": 50, "min-fraction": 0
                        },
                        "Feed-in": {
                            "wacc": 0, "cap-lo": 0, "cap-up": 1000, "fix-cost": 0, "inst-cap": 1000, "inv-cost": 0, "max-grad": "inf", "var-cost": 0,
                            "commodity": {"Elec": {"ratio": 1, "Direction": "In", "ratio-min": 1}, "Elec sell": {"ratio": 1, "Direction": "Out", "ratio-min": 1}},
                            "description": "Sell electricity", "depreciation": 50, "min-fraction": 0
                        },
                        "Photovoltaics": {
                            "wacc": 0.07, "cap-lo": max_solar, "cap-up": max_solar, "fix-cost": 0, "inst-cap": max_solar, "inv-cost": 0, "max-grad": "inf", "var-cost": 0,
                            "commodity": {"Elec": {"ratio": 1, "Direction": "Out", "ratio-min": 1}, "Solar": {"ratio": 1, "Direction": "In", "ratio-min": 1}},
                            "description": "PV", "area-per-cap": 5, "depreciation": 25, "min-fraction": 0
                        },
                        "HeatPump": {
                             "wacc": 0.07,
                             "cap-lo": 0,
                             "cap-up": config.heat_pump_max_power_kw * config.cop, # Capacity on Output (Heat)
                             "inst-cap": config.heat_pump_max_power_kw * config.cop,
                             "fix-cost": 0, "inv-cost": 0, "max-grad": "inf", "var-cost": 0,
                             "commodity": {
                                 "Elec": {"ratio": 1/config.cop, "Direction": "In"},
                                 "Heat": {"ratio": 1, "Direction": "Out"}
                             },
                             "description": "Heat Pump", "depreciation": 20, "min-fraction": 0
                        }
                    },
                    "commodity": {
                        "Elec": {
                            "Type": "Demand", "unitC": "kWh", "unitR": "kW", "demand": load_hourly.tolist(),
                            "storage": {
                                "Lead-Acid Battery": {
                                    "init": init_bat,
                                    "wacc": 0.007, "eff-in": np.sqrt(config.battery_efficiency), "eff-out": np.sqrt(config.battery_efficiency), # Split round trip eff
                                    "cap-lo-c": config.battery_capacity_kwh, "cap-lo-p": config.battery_max_power_kw,
                                    "cap-up-c": config.battery_capacity_kwh, "cap-up-p": config.battery_max_power_kw,
                                    "discharge": 0, "fix-cost-c": 0, "fix-cost-p": 0,
                                    "inst-cap-c": config.battery_capacity_kwh, "inst-cap-p": config.battery_max_power_kw,
                                    "inv-cost-c": 0, "inv-cost-p": 0, "var-cost-c": 0, "var-cost-p": 0,
                                    "description": "Battery", "depreciation": 5
                                }
                            }
                        },
                        "Heat": {
                            "Type": "Demand", "unitC": "kWh", "unitR": "kW", "demand": heat_demand_hourly.tolist(),
                            "storage": {
                                "WaterTank": {
                                    "init": init_heat,
                                    "wacc": 0.007, "eff-in": 1, "eff-out": 1,
                                    "cap-lo-c": config.heat_storage_capacity_kwh, "cap-lo-p": 1000,
                                    "cap-up-c": config.heat_storage_capacity_kwh, "cap-up-p": 1000,
                                    "discharge": 0, "fix-cost-c": 0, "fix-cost-p": 0,
                                    "inst-cap-c": config.heat_storage_capacity_kwh, "inst-cap-p": 1000,
                                    "inv-cost-c": 0, "inv-cost-p": 0, "var-cost-c": 0, "var-cost-p": 0,
                                    "description": "Heat Storage", "depreciation": 20
                                }
                            }
                        },
                        "Solar": {"Type": "SupIm", "supim": norm_solar, "unitC": "kWh", "unitR": "kW"},
                        "Elec buy": {"max": "inf", "Type": "Buy", "price": 0.1, "unitC": "kWh", "unitR": "kW", "maxperhour": "inf"},
                        "Elec sell": {"max": "inf", "Type": "Sell", "price": 0.0, "unitC": "kWh", "unitR": "kW", "maxperhour": "inf"}
                    }
                }
            },
            "global": {"CO2 limit": 150000000, "Cost limit": 35000000000},
            "c_timesteps": timesteps,
            "buysellprice": {"Elec buy": price_hourly.tolist(), "Elec sell": sell_price_hourly.tolist()}
        }

        # 3. Send Request
        try:
            # Save payload for debugging
            with open('adg_payload.urbs', 'w') as f:
                json.dump(payload, f, indent=2)

            response = requests.post(self.url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            # Check for solver errors
            if 'results' not in result.get('data', {}):
                raise ValueError(f"Solver failed. Response: {result}")

            # 4. Parse Result (First Hour)
            # Battery
            storage_res = result['data']['results']['Main']['Elec']['storage']
            # Handle potential nesting or direct access depending on URBS version/behavior
            if 'Lead-Acid Battery' in storage_res:
                storage_res = storage_res['Lead-Acid Battery']

            charge = storage_res['Stored'][0]
            discharge = storage_res['Retrieved'][0]
            bat_power = charge - discharge

            # Heat Pump
            hp_heat_out = result['data']['results']['Main']['HeatPump']['commodity']['Heat']['Out'][0]
            hp_power = hp_heat_out / config.cop

            return {
                'battery_power_kw': float(bat_power),
                'heat_pump_power_kw': float(hp_power),
                'pv_curtailment_limit_kw': 999.0
            }

        except Exception as e:
            print(f"MPC Error: {e}. Falling back to RuleBased.")
            # Fallback
            rb = RuleBasedOptimizer()
            return rb.get_setpoints(observation, forecast, config)


# In[69]:


def run_simulation(optimizer: BaseOptimizer, controllers: Dict[str, BaseDeviceController], config: HEMSConfig):
    plant = HEMSPlant(config)
    
    # Simulation Loop
    for minute in range(config.total_minutes):

        # 1. Hourly Trigger: Get Strategic Setpoint
        if minute % 60 == 0:
            obs = plant.get_observation()
            # Get forecast for next 24 hours
            forecast = plant.get_forecast(horizon_hours=24)
            hourly_setpoint = optimizer.get_setpoints(obs, forecast, config)

        # 2. Minute Trigger: Get Real-Time Action
        obs = plant.get_observation()
        
        action = {}
        for key, ctrl in controllers.items():
            sp = hourly_setpoint.get(key)
            action[key] = ctrl.get_action(sp, obs, config)

        # 3. Physics Step
        plant.step(action)

    return plant.history


# In[70]:


def plot_results(results_dict):
    """
    Plots comparison of multiple strategies using Plotly for interactivity.
    results_dict: { "Strategy Name": history_dict, ... }
    """
    # Get time axis from first result
    first_key = list(results_dict.keys())[0]
    t = np.arange(len(results_dict[first_key]['solar_generated'])) / 60.0

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("Environment Variables", "Battery System", "Heat System"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]]
    )

    # --- Subplot 1: Environment Variables ---
    # Assuming environment is consistent, take from first
    hist = results_dict[first_key]

    # Solar
    fig.add_trace(go.Scatter(x=t, y=hist['solar_generated'], name='Solar (kW)', 
                             line=dict(color='orange'), opacity=0.8), row=1, col=1, secondary_y=False)
    # Load
    fig.add_trace(go.Scatter(x=t, y=hist['load_consumed'], name='El. Load (kW)', 
                             line=dict(color='cyan'), opacity=0.6), row=1, col=1, secondary_y=False)
    # Heat Demand
    fig.add_trace(go.Scatter(x=t, y=hist['heat_demand_met'], name='Heat Demand (kW)', 
                             line=dict(color='red', dash='dot'), opacity=0.6), row=1, col=1, secondary_y=False)

    # Price (Right Axis)
    fig.add_trace(go.Scatter(x=t, y=hist['import_price'], name='Import Price ($/kWh)', 
                             line=dict(color='yellow', dash='dash'), opacity=0.5), row=1, col=1, secondary_y=True)

    # --- Subplot 2 & 3: Strategies ---
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b'] # Plotly default-ish colors

    max_bat_p = 0.1
    max_bat_s = 0.1

    for i, (name, h) in enumerate(results_dict.items()):
        c = colors[i % len(colors)]

        # --- Subplot 2: Battery ---
        # Power (Left)
        fig.add_trace(go.Scatter(x=t, y=h['battery_power'], name=f'{name} Bat Power', 
                                 line=dict(color=c), opacity=0.6, legendgroup=name), row=2, col=1, secondary_y=False)
        # SoC (Right)
        fig.add_trace(go.Scatter(x=t, y=h['battery_soc'], name=f'{name} Bat SoC', 
                                 line=dict(color=c, dash='dash', width=2), legendgroup=name), row=2, col=1, secondary_y=True)

        max_bat_p = max(max_bat_p, np.max(np.abs(h['battery_power'])))
        max_bat_s = max(max_bat_s, np.max(h['battery_soc']))

        # --- Subplot 3: Heat ---
        # Power (Left)
        fig.add_trace(go.Scatter(x=t, y=h['heat_pump_power'], name=f'{name} HP Power', 
                                 line=dict(color=c), opacity=0.6, legendgroup=name, showlegend=False), row=3, col=1, secondary_y=False)
        # SoC (Right)
        fig.add_trace(go.Scatter(x=t, y=h['heat_storage_soc'], name=f'{name} Heat SoC', 
                                 line=dict(color=c, dash='dash', width=2), legendgroup=name, showlegend=False), row=3, col=1, secondary_y=True)

    # Update Layout
    fig.update_layout(height=900, title_text="HEMS Simulation Results", hovermode="x unified")

    # Axis Labels
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price ($/kWh)", row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Power (kW)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Elec. Power (kW)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Therm. Energy (kWh)", row=3, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Time (Hours)", row=3, col=1)

    # Align Zeros for Battery (Symmetric View)
    fig.update_yaxes(range=[-max_bat_p * 1.1, max_bat_p * 1.1], row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=[-max_bat_s * 1.1, max_bat_s * 1.1], row=2, col=1, secondary_y=True)

    # Align Zeros for Heat (Bottom View)
    fig.update_yaxes(rangemode="tozero", row=3, col=1, secondary_y=False)
    fig.update_yaxes(rangemode="tozero", row=3, col=1, secondary_y=True)

    # Print Costs
    print("\n--- Performance Summary ---")
    for name, h in results_dict.items():
        total_cost = np.sum(h['cost'])
        print(f"{name}: Total Cost = ${total_cost:.2f}")
    
    return fig




