# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 17:25:32 2024

@author: xavierpang
"""

import numpy as np

from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from skfuzzy import membership as mf

# Define input variables
temperature = ctrl.Antecedent(np.arange(10, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(20, 101, 1), 'humidity')
co2_level = ctrl.Antecedent(np.arange(300, 1001, 1), 'co2_level')
light_intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'light_intensity')
energy_cost = ctrl.Antecedent(np.arange(0, 2, 1), 'energy_cost')  # 0=Low, 1=High
plant_health_feedback = ctrl.Antecedent(np.arange(0, 3, 1), 'plant_health_feedback')  # 0=Poor, 1=Average, 2=Good

# Define output variables
hvac_adjustment = ctrl.Consequent(np.arange(0, 101, 1), 'hvac_adjustment')
humidification = ctrl.Consequent(np.arange(0, 101, 1), 'humidification')
co2_injection = ctrl.Consequent(np.arange(0, 101, 1), 'co2_injection')

# Define membership functions
temperature['low'] = mf.gaussmf(temperature.universe, 10, 5)
temperature['optimal'] = mf.gaussmf(temperature.universe, 25, 5)
temperature['high'] = mf.gaussmf(temperature.universe, 35, 5)

humidity['low'] = mf.gaussmf(humidity.universe, 20, 10)
humidity['optimal'] = mf.gaussmf(humidity.universe, 60, 10)
humidity['high'] = mf.gaussmf(humidity.universe, 90, 10)

co2_level['low'] = mf.gaussmf(co2_level.universe, 300, 100)
co2_level['optimal'] = mf.gaussmf(co2_level.universe, 650, 100)
co2_level['high'] = mf.gaussmf(co2_level.universe, 900, 100)

light_intensity['low'] = mf.gaussmf(light_intensity.universe, 10, 10)
light_intensity['medium'] = mf.gaussmf(light_intensity.universe, 50, 10)
light_intensity['high'] = mf.gaussmf(light_intensity.universe, 90, 10)

energy_cost['low'] = mf.trimf(energy_cost.universe, [0, 0, 1])
energy_cost['high'] = mf.trimf(energy_cost.universe, [0, 1, 1])

plant_health_feedback['poor'] = mf.trimf(plant_health_feedback.universe, [0, 0, 1])
plant_health_feedback['average'] = mf.trimf(plant_health_feedback.universe, [0, 1, 2])
plant_health_feedback['good'] = mf.trimf(plant_health_feedback.universe, [1, 2, 2])

hvac_adjustment['off'] = mf.trimf(hvac_adjustment.universe, [0, 0, 25])
hvac_adjustment['low'] = mf.trimf(hvac_adjustment.universe, [0, 25, 50])
hvac_adjustment['medium'] = mf.trimf(hvac_adjustment.universe, [25, 50, 75])
hvac_adjustment['high'] = mf.trimf(hvac_adjustment.universe, [50, 100, 100])

humidification['off'] = mf.trimf(humidification.universe, [0, 0, 25])
humidification['low'] = mf.trimf(humidification.universe, [0, 25, 50])
humidification['medium'] = mf.trimf(humidification.universe, [25, 50, 75])
humidification['high'] = mf.trimf(humidification.universe, [50, 100, 100])

co2_injection['off'] = mf.trimf(co2_injection.universe, [0, 0, 25])
co2_injection['low'] = mf.trimf(co2_injection.universe, [0, 25, 50])
co2_injection['medium'] = mf.trimf(co2_injection.universe, [25, 50, 75])
co2_injection['high'] = mf.trimf(co2_injection.universe, [50, 100, 100])

# Define rules
rules = [
    
    # HVAC Adjustment Rules
    ctrl.Rule(temperature['high'] & humidity['low'] & energy_cost['low'], hvac_adjustment['high']),
    ctrl.Rule(temperature['high'] & humidity['low'] & energy_cost['high'], hvac_adjustment['medium']),
    ctrl.Rule(temperature['optimal'] & humidity['high'] & co2_level['low'], hvac_adjustment['low']),
    ctrl.Rule(temperature['low'] & humidity['optimal'] & energy_cost['high'], hvac_adjustment['off']),
    ctrl.Rule(temperature['low'] & co2_level['optimal'], hvac_adjustment['low']),
    ctrl.Rule(temperature['low'] & co2_level['low'] & plant_health_feedback['poor'], hvac_adjustment['medium']),
    ctrl.Rule(temperature['high'] & light_intensity['high'], hvac_adjustment['high']),
    ctrl.Rule(temperature['optimal'] & plant_health_feedback['good'], hvac_adjustment['off']),
    ctrl.Rule(temperature['high'] & humidity['high'] & energy_cost['low'], hvac_adjustment['medium']),
    ctrl.Rule(temperature['optimal'] & humidity['optimal'], hvac_adjustment['off']),

    # Humidification Rules
    ctrl.Rule(humidity['low'] & temperature['high'] & energy_cost['low'], humidification['high']),
    ctrl.Rule(humidity['low'] & temperature['high'] & energy_cost['high'], humidification['medium']),
    ctrl.Rule(humidity['optimal'] & temperature['optimal'], humidification['off']),
    ctrl.Rule(humidity['high'] & temperature['low'], humidification['high']),
    ctrl.Rule(humidity['low'] & co2_level['high'] & plant_health_feedback['poor'], humidification['medium']),
    ctrl.Rule(humidity['high'] & light_intensity['high'] & plant_health_feedback['average'], humidification['medium']),
    ctrl.Rule(humidity['low'] & temperature['optimal'], humidification['high']),
    ctrl.Rule(humidity['high'] & temperature['high'] & energy_cost['low'], humidification['medium']),
    ctrl.Rule(humidity['optimal'] & plant_health_feedback['good'], humidification['low']),
    ctrl.Rule(humidity['low'] & energy_cost['low'], humidification['high']),
    ctrl.Rule(humidity['high'] & energy_cost['high'], humidification['off']),
    
    # CO₂ Injection Rules
    ctrl.Rule(co2_level['low'] & light_intensity['high'] & plant_health_feedback['good'], co2_injection['high']),
    ctrl.Rule(co2_level['low'] & light_intensity['medium'], co2_injection['medium']),
    ctrl.Rule(co2_level['optimal'] & light_intensity['high'], co2_injection['medium']),
    ctrl.Rule(co2_level['high'] & energy_cost['high'], co2_injection['off']),
    ctrl.Rule(co2_level['low'] & plant_health_feedback['poor'], co2_injection['medium']),
    ctrl.Rule(co2_level['optimal'] & light_intensity['low'], co2_injection['low']),
    ctrl.Rule(co2_level['high'] & light_intensity['low'], co2_injection['off']),
    ctrl.Rule(co2_level['low'] & humidity['low'], co2_injection['high']),
    ctrl.Rule(co2_level['low'] & temperature['high'], co2_injection['high']),
    ctrl.Rule(co2_level['optimal'] & plant_health_feedback['average'], co2_injection['low']),
    ctrl.Rule(co2_level['high'] & light_intensity['high'], co2_injection['medium']),
]

# Create control system and simulation
climate_control_system = ctrl.ControlSystem(rules)
climate_control_simulation = ctrl.ControlSystemSimulation(climate_control_system)


def prompt_user():
    # Main prompt to select the mode
    while True:
        print("Choose an option:")
        print("1. Show Membership Functions")
        print("2. Predict an Output")
        print("3. Analyze Relationship Between Inputs and Output")
        
        mode = input("Enter 1, 2, or 3: ").strip()
        
        if mode == '1':
            show_membership_functions()
            break
        elif mode == '2':
            predict_output()
            break
        elif mode == '3':
            analyze_relationship()
            break
        else:
            print("Invalid input. Please enter a valid option (1, 2, or 3).\n")

        
def show_membership_functions():
    print("\n--- Show Membership Functions ---")
    while True:
        variable = input("Enter variable name to view its membership functions (e.g., temperature, humidity, co2_level, light_intensity, hvac_adjustment, humidification, co2_injection): ").strip()
        try:
            plot_membership(variable)
            break
        except NameError:
            print("Variable not found. Please ensure the variable name is correct and try again.")

        
def predict_output():
    print("\n--- Prediction Mode ---")
    # Gather user inputs with validation
    inputs = {}
    inputs['temperature'] = get_input_float("Enter temperature (10 to 40 °C): ", 10, 40)
    inputs['humidity'] = get_input_float("Enter humidity (20 to 100 %): ", 20, 100)
    inputs['co2_level'] = get_input_float("Enter CO₂ level (300 to 1000 ppm): ", 300, 1000)
    inputs['light_intensity'] = get_input_float("Enter light intensity (0 to 100 %): ", 0, 100)
    inputs['energy_cost'] = get_input_int("Enter energy cost (0 = Low, 1 = High): ", [0, 1])
    inputs['plant_health_feedback'] = get_input_int("Enter plant health feedback (0 = Poor, 1 = Average, 2 = Good): ", [0, 1, 2])

    # Reinitialize the climate control simulation each time to ensure fresh inputs
    simulation = ctrl.ControlSystemSimulation(climate_control_system)

    # Set inputs in the simulation
    for key, value in inputs.items():
        simulation.input[key] = value

    # Perform the computation
    try:
        simulation.compute()
    except Exception as e:
        print(f"Error during computation: {e}")
        return

    # Display the output along with membership graphs
    output_labels = ['hvac_adjustment', 'humidification', 'co2_injection']
    print("\nPredicted Outputs:")
    for output_label in output_labels:
        try:
            output_value = simulation.output[output_label]
            print(f"{output_label.replace('_', ' ').capitalize()}: {output_value}")

            # Plot membership graph for output
            plot_membership(output_label, output_value)
        except KeyError:
            print(f"Warning: '{output_label}' could not be computed. Check input values or rules.")
        
def plot_membership(variable, value=None):
    # Plot membership function for the variable with an optional value highlighted
    plt.figure()
    var = eval(variable)  # Reference the variable from the fuzzy system definitions
    for term_name, membership_func in var.terms.items():
        plt.plot(var.universe, membership_func.mf, label=term_name)
    if value is not None:
        plt.axvline(x=value, color='red', linestyle='--', label=f'Predicted value = {value}')
    plt.title(f'Membership Function for {variable.capitalize()}')
    plt.xlabel(variable.capitalize())
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.show()
    
    
def analyze_relationship():
    print("\n--- Relationship Mode ---")
    output_label = get_input_choice("Enter the output to analyze (hvac_adjustment, humidification, co2_injection): ", ['hvac_adjustment', 'humidification', 'co2_injection'])

    # Ask for two primary inputs to vary in the analysis
    primary_input_1 = get_input_choice("Enter the first input to analyze (temperature, humidity, co2_level, light_intensity): ", ['temperature', 'humidity', 'co2_level', 'light_intensity'])
    while True:
        primary_input_2 = get_input_choice("Enter the second input to analyze (temperature, humidity, co2_level, light_intensity): ", ['temperature', 'humidity', 'co2_level', 'light_intensity'])
        if primary_input_1 != primary_input_2:
            break
        print("Please choose two different inputs.")

    # Directly get the universe (range) of each input variable
    input_ranges = {
        'temperature': temperature.universe,
        'humidity': humidity.universe,
        'co2_level': co2_level.universe,
        'light_intensity': light_intensity.universe,
        'energy_cost': energy_cost.universe,
        'plant_health_feedback': plant_health_feedback.universe
    }
    
    primary_input_range_1 = input_ranges[primary_input_1]
    primary_input_range_2 = input_ranges[primary_input_2]

    # Ask if the user wants to include plant health feedback and energy cost in the analysis
    include_cost = get_yes_no("Include energy cost in analysis? (yes/no): ")
    include_health = get_yes_no("Include plant health feedback in analysis? (yes/no): ")

    # Default values for other inputs
    input_values = {
        'temperature': 25,         # Default temperature
        'humidity': 50,            # Default humidity
        'co2_level': 650,          # Default CO2 level
        'light_intensity': 50,     # Default light intensity
        'energy_cost': 0,          # Default energy cost
        'plant_health_feedback': 1 # Default plant health feedback
    }

    # Create combinations for energy cost and plant health feedback
    cost_values = [0, 1] if include_cost else [0]
    health_values = [0, 1, 2] if include_health else [1]

    # Call function to analyze relationship
    plot_relationship(output_label, (primary_input_1, primary_input_2), 
                      (primary_input_range_1, primary_input_range_2), 
                      input_values, cost_values, health_values)



def plot_relationship(output_label, primary_inputs, primary_ranges, input_values, cost_values, health_values):
    # Automatically determine labels based on primary input names
    labels = { 
        'temperature': 'Temperature (°C)', 
        'humidity': 'Humidity (%)',
        'co2_level': 'CO₂ Level (ppm)', 
        'light_intensity': 'Light Intensity (%)'
    }
    primary_labels = [labels[primary_inputs[0]], labels[primary_inputs[1]]]
    
    # Initialize figure for subplots
    fig = plt.figure(figsize=(18, 12))
    plot_index = 1  # Track subplot index

    # Loop through each combination of energy cost and plant health feedback
    for cost in cost_values:
        for health in health_values:
            # Set energy cost and plant health feedback in input values
            input_values['energy_cost'] = cost
            input_values['plant_health_feedback'] = health

            # Prepare a 2D array to hold output values for each combination of primary inputs
            output_values = np.zeros((len(primary_ranges[0]), len(primary_ranges[1])))

            for i, val_1 in enumerate(primary_ranges[0]):
                for j, val_2 in enumerate(primary_ranges[1]):
                    # Initialize a new simulation for each combination of inputs
                    climate_control_simulation = ctrl.ControlSystemSimulation(climate_control_system)
                    
                    # Set primary inputs for the simulation with current values from the ranges
                    climate_control_simulation.input[primary_inputs[0]] = val_1
                    climate_control_simulation.input[primary_inputs[1]] = val_2

                    # Set fixed values for all other inputs
                    for key, value in input_values.items():
                        if key not in primary_inputs:
                            climate_control_simulation.input[key] = value
                    
                    # Debugging: Print input values to ensure they are set correctly
                    print(f"Computing for inputs: {primary_inputs[0]}={val_1}, {primary_inputs[1]}={val_2}, "
                          f"energy_cost={cost}, plant_health_feedback={health}")

                    # Compute the output and handle cases where the output is not defined
                    try:
                        climate_control_simulation.compute()
                        output_values[i, j] = climate_control_simulation.output[output_label]
                    except KeyError:
                        print(f"Warning: '{output_label}' could not be computed for these inputs.")
                        output_values[i, j] = np.nan  # Use NaN to represent undefined outputs

            # Plot the 3D surface for this combination of energy cost and plant health feedback
            ax = fig.add_subplot(len(cost_values), len(health_values), plot_index, projection='3d')
            X, Y = np.meshgrid(primary_ranges[1], primary_ranges[0])
            ax.plot_surface(X, Y, output_values, cmap='viridis', edgecolor='none')
            ax.set_xlabel(primary_labels[1])
            ax.set_ylabel(primary_labels[0])
            ax.set_zlabel(output_label.capitalize())
            ax.set_title(f'Energy Cost: {"Low" if cost == 0 else "High"}, '
                         f'Plant Health: {["Poor", "Average", "Good"][health]}')
            plot_index += 1  # Move to next subplot

    plt.suptitle(f'Relationship Analysis: {primary_labels[0]} and {primary_labels[1]} with {output_label.capitalize()}')
    plt.tight_layout()
    plt.show()

# Helper functions for user input validation

def get_input_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")


def get_input_int(prompt, valid_values):
    while True:
        try:
            value = int(input(prompt))
            if value in valid_values:
                return value
            else:
                print(f"Please enter one of the following values: {valid_values}")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def get_input_choice(prompt, valid_choices):
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        else:
            print(f"Invalid choice. Please choose from: {', '.join(valid_choices)}")


def get_yes_no(prompt):
    while True:
        choice = input(prompt).strip().lower()
        if choice in ['yes', 'no']:
            return choice == 'yes'
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")



    
# Call the main prompt
prompt_user()

