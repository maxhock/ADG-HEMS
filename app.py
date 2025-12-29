import streamlit as st
import numpy as np
import pandas as pd
from hems_core import (
    HEMSConfig, 
    ResidualOptimizer, 
    RuleBasedOptimizer, 
    MPCOptimizer, 
    PassThroughDeviceController, 
    InterpolatingDeviceController, 
    S2EnvelopeController,
    run_simulation, 
    plot_results,
    check_docker
)

st.set_page_config(page_title="HEMS Simulation", layout="wide")

st.title("Home Energy Management System (HEMS) Simulation")

# --- Sidebar: Configuration ---
st.sidebar.header("System Configuration")

days = st.sidebar.number_input("Simulation Days", min_value=1, max_value=7, value=2)
battery_capacity = st.sidebar.number_input("Battery Capacity (kWh)", value=13.5)
battery_power = st.sidebar.number_input("Battery Max Power (kW)", value=5.0)
heat_storage_capacity = st.sidebar.number_input("Heat Storage Capacity (kWh)", value=15.0)
heat_pump_power = st.sidebar.number_input("Heat Pump Max Power (kW)", value=4.0)

config = HEMSConfig(
    days=days,
    battery_capacity_kwh=battery_capacity,
    battery_max_power_kw=battery_power,
    heat_storage_capacity_kwh=heat_storage_capacity,
    heat_pump_max_power_kw=heat_pump_power
)

# --- Sidebar: Scenario Management ---
st.sidebar.header("Scenarios")

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# Add Scenario Form
with st.sidebar.form("add_scenario"):
    st.subheader("Add Scenario")
    name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.scenarios) + 1}")
    
    optimizer_name = st.selectbox("Optimizer", ["Residual", "Rule-Based", "MPC"])
    
    st.markdown("### Device Controllers")
    bat_ctrl_name = st.selectbox("Battery Controller", ["Pass-Through", "Interpolating", "S2 Envelope"])
    hp_ctrl_name = st.selectbox("Heat Pump Controller", ["Pass-Through", "Interpolating", "S2 Envelope"])
    pv_ctrl_name = st.selectbox("PV Controller", ["Pass-Through", "Interpolating"])
    
    submitted = st.form_submit_button("Add Scenario")
    if submitted:
        st.session_state.scenarios.append({
            "name": name,
            "optimizer": optimizer_name,
            "bat_ctrl": bat_ctrl_name,
            "hp_ctrl": hp_ctrl_name,
            "pv_ctrl": pv_ctrl_name
        })
        st.success(f"Added {name}")

# List Scenarios
st.sidebar.subheader("Current Scenarios")
for i, sc in enumerate(st.session_state.scenarios):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.markdown(f"**{sc['name']}**")
        st.caption(f"Opt: {sc['optimizer']}")
        st.caption(f"Bat: {sc['bat_ctrl']} | HP: {sc['hp_ctrl']} | PV: {sc['pv_ctrl']}")
    if col2.button("X", key=f"del_{i}"):
        st.session_state.scenarios.pop(i)
        st.rerun()

if not st.session_state.scenarios:
    st.sidebar.warning("Add at least one scenario to run.")

# --- Main Area ---

# Docker Check
if st.button("Check/Start Docker (for MPC)"):
    with st.spinner("Checking Docker..."):
        check_docker()
    st.success("Docker check complete.")

def get_controller_instance(type_name, device_type):
    """Factory for creating controller instances based on selection."""
    if type_name == "Pass-Through":
        init_sp = 999.0 if device_type == 'pv' else 0.0
        return PassThroughDeviceController(initial_setpoint=init_sp)
        
    elif type_name == "Interpolating":
        init_sp = 999.0 if device_type == 'pv' else 0.0
        return InterpolatingDeviceController(initial_setpoint=init_sp)
        
    elif type_name == "S2 Envelope":
        if device_type == 'battery':
            return S2EnvelopeController(
                storage_key='battery_soc_kwh', 
                capacity_attr='battery_capacity_kwh', 
                max_power_attr='battery_max_power_kw'
            )
        elif device_type == 'heat_pump':
            return S2EnvelopeController(
                storage_key='heat_storage_kwh', 
                capacity_attr='heat_storage_capacity_kwh', 
                max_power_attr='heat_pump_max_power_kw'
            )
        else:
            return PassThroughDeviceController(initial_setpoint=999.0)
            
    return PassThroughDeviceController()

if st.button("Run Simulation", type="primary", disabled=len(st.session_state.scenarios) == 0):
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_scenarios = len(st.session_state.scenarios)
    
    for i, sc in enumerate(st.session_state.scenarios):
        status_text.text(f"Running {sc['name']}...")
        
        # Instantiate Optimizer
        if sc['optimizer'] == "Residual":
            opt = ResidualOptimizer()
        elif sc['optimizer'] == "Rule-Based":
            opt = RuleBasedOptimizer()
        elif sc['optimizer'] == "MPC":
            opt = MPCOptimizer()
            
        # Instantiate Controllers
        ctrl = {
            'battery_power_kw': get_controller_instance(sc['bat_ctrl'], 'battery'),
            'heat_pump_power_kw': get_controller_instance(sc['hp_ctrl'], 'heat_pump'),
            'pv_curtailment_limit_kw': get_controller_instance(sc['pv_ctrl'], 'pv')
        }
            
        # Run
        np.random.seed(42) # Reset seed for consistency across scenarios
        history = run_simulation(opt, ctrl, config)
        results[sc['name']] = history
        
        progress_bar.progress((i + 1) / total_scenarios)
        
    status_text.text("Simulation Complete!")
    
    # Plot
    st.subheader("Results Comparison")
    fig = plot_results(results)
    st.plotly_chart(fig, width='stretch')
    
    # Metrics Table
    st.subheader("Performance Metrics")
    metrics = []
    for name, h in results.items():
        total_cost = np.sum(h['cost'])
        metrics.append({
            "Scenario": name,
            "Total Cost ($)": f"{total_cost:.2f}",
            "Final Battery SoC (kWh)": f"{h['battery_soc'][-1]:.2f}",
            "Final Heat SoC (kWh)": f"{h['heat_storage_soc'][-1]:.2f}"
        })
    
    st.table(pd.DataFrame(metrics))

