
"""
Hypatia interface module. Contains the Model class with all
the methods needed to create,solve and save the results of a hypatia
model.
"""
from cvxpy import installed_solvers
from hypatia.error_log.Exceptions import (
    WrongInputMode,
    DataNotImported,
    ResultOverWrite,
    SolverNotFound,
)
from hypatia.utility.excel import (
    read_settings,
    write_parameters_files,
    read_parameters,
)
from hypatia.utility.constants import ModelMode
from hypatia.utility.constants import OptimizationMode
from hypatia.utility.constants import EnsureFeasibility 
from hypatia.backend.Build import BuildModel
from copy import deepcopy
from hypatia.postprocessing.PostProcessingList import POSTPROCESSING_MODULES
import itertools
import os
import pandas as pd
import numpy as np
from datetime import (
    datetime,
    timedelta
)

import logging
import random
import tempfile
import shutil
logger = logging.getLogger(__name__)


from hypatia.core.main import Model

# Enhancements made to the original script
# Kianjn

def identify_technology_order(self, new_capacity_path, result_path):
        """
        Identify the order of technologies based on their appearance in the initial run.

        This function reads a CSV file containing new capacity data for various technologies. 
        It identifies the unique technologies and assigns an order based on their appearance 
        in the file, returning a dictionary that maps each technology to its respective order.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.

        Returns:
        dict: A dictionary mapping each technology name (str) to its corresponding order number (int).
        """

        # Read the CSV file into a DataFrame
        new_capacity_df = pd.read_csv(new_capacity_path)
        
        # Identify the unique technologies present in the 'Technology' column
        technologies_order = new_capacity_df['Technology'].unique()

        # Create a dictionary where each technology is mapped to its order (starting from 1)
        technology_order_mapping = {tech: i + 1 for i, tech in enumerate(technologies_order)}

        # Convert the dictionary to a DataFrame
        order_df = pd.DataFrame(list(technology_order_mapping.items()), columns=['Technology', 'Order'])
        
        # Ensure the result_path directory exists (optional)
        os.makedirs(result_path, exist_ok=True)
        
        # Save the mapping to a CSV file in the results folder
        order_df.to_csv(os.path.join(result_path, 'technology_order_mapping.csv'), index=False)
    
        # Return the dictionary with technology-order mappings
        return technology_order_mapping

def find_technologies(self, new_capacity_path, technology_order_mapping):
        """
        Find and summarize technology data from the new capacity dataset.

        This function identifies the top 3 technologies with the highest total values and selects
        4 random technologies from the remaining ones. It also keeps one `max_technology` (highest)
        and one `random_technology` (from the selected random technologies) for compatibility with 
        existing functions.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        technology_order_mapping (dict): A dictionary mapping technologies to their respective order.

        Returns:
        tuple: Contains the following elements:
            - max_technology (str): The technology with the highest total value (used in other functions).
            - max_value (float): Total value of the max_technology.
            - max_order (int): Order of the max_technology.
            - random_technology (str): A randomly selected technology from the remaining ones.
            - random_value (float): Total value of the random_technology.
            - random_order (int): Order of the random_technology.
            - max_technologies (list of str): Top 3 technologies with the highest total values.
            - random_technologies (list of str): 4 randomly selected technologies from the remaining ones.
            - technologies_list (list of tuples): List of all technologies with their orders and total values.
        """
        # Forbidden technologies that will not be considered as max or random
        forbidden_technologies = [
            "Elec_distribution", "Bio_oil_supply", "Crude_oil_supply", "NG_supply", "SFF_supply",
            "BW_supply", "Hydrogen_supply", "OP_supply", "Electricity_imports", "Oil_refinery",
            "Bio_refinery", "Transport_biofuels_blending", "BW_CHP_P", "NG_CHP_P", "Transport_mix",
            "Industry_mix", "Civil_and_agriculture_mix", "Export_mix", "Elec_transmission_distribution"
        ]

        new_capacity_df = pd.read_csv(new_capacity_path)
        tech_sum = new_capacity_df.groupby('Technology')['Value'].sum()

        # Create a list of (order, tech, value) but we mainly rely on tech_sum for sorting
        technologies_list = [(technology_order_mapping[tech], tech, tech_sum[tech]) for tech in tech_sum.index]

        allowed_technologies = tech_sum.drop(labels=forbidden_technologies, errors='ignore')

        # Get top 3 technologies by largest total values
        top_3 = allowed_technologies.nlargest(3)
        max_technologies = top_3.index.tolist()
        max_values = top_3.values.tolist()

        # Explicit check (not strictly needed, nlargest is already sorted)
        # Just here for clarity and reassurance
        tech_value_pairs = list(zip(max_technologies, max_values))
        tech_value_pairs.sort(key=lambda x: x[1], reverse=True)
        max_technologies = [tv[0] for tv in tech_value_pairs]
        max_values = [tv[1] for tv in tech_value_pairs]

        # Exclude the top 3 technologies from the remaining set
        remaining_technologies = allowed_technologies.drop(labels=max_technologies)
        
        # Introduce randomness seed for diversity (optional)
        random.seed(None)  # None uses system time
        
        # Ensure we have a sufficient pool. If very small, you might get repetition
        remaining_list = remaining_technologies.index.tolist()
        if len(remaining_list) == 0:
            # If no technologies left for random selection, handle gracefully
            random_technologies = []
        else:
            random_technologies = random.sample(remaining_list, min(3, len(remaining_list)))

        random_values = [remaining_technologies[tech] for tech in random_technologies] if random_technologies else []
        random_orders = [technology_order_mapping[tech] for tech in random_technologies] if random_technologies else []

        max_technology = max_technologies[0]
        max_value = max_values[0]
        max_orders = [technology_order_mapping[tech] for tech in max_technologies]
        max_order = max_orders[0]

        second_max_technology = max_technologies[1] if len(max_technologies) > 1 else None
        second_max_value = max_values[1] if len(max_values) > 1 else None
        second_max_order = max_orders[1] if len(max_orders) > 1 else None

        third_max_technology = max_technologies[2] if len(max_technologies) > 2 else None
        third_max_value = max_values[2] if len(max_values) > 2 else None
        third_max_order = max_orders[2] if len(max_orders) > 2 else None

        random_technology = random_technologies[0] if random_technologies else None
        random_value = random_values[0] if random_values else None
        random_order = random_orders[0] if random_orders else None

        second_random_technology = random_technologies[1] if len(random_technologies) > 1 else None
        second_random_value = random_values[1] if len(random_values) > 1 else None
        second_random_order = random_orders[1] if len(random_orders) > 1 else None

        third_random_technology = random_technologies[2] if len(random_technologies) > 2 else None
        third_random_value = random_values[2] if len(random_values) > 2 else None
        third_random_order = random_orders[2] if len(random_orders) > 2 else None

        return (
            max_technology, max_value, max_order,
            second_max_technology, second_max_value, second_max_order,
            third_max_technology, third_max_value, third_max_order,
            random_technology, random_value, random_order,
            second_random_technology, second_random_value, second_random_order,
            third_random_technology, third_random_value, third_random_order,
            max_technologies, random_technologies,
            technologies_list
        )

def find_last_year_values(self, total_capacity_path, max_technology, random_technology):
        """
        Retrieve the capacity values for the max and random technologies in the last year of the dataset.

        Parameters:
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        max_technology (str): The technology with the highest total value.
        random_technology (str): The randomly selected technology.

        Returns:
        tuple: Contains the following elements:
            - last_year (int/str): The last year in the dataset.
            - max_technology_value (float): The capacity value for the max_technology in the last year.
            - random_technology_value (float): The capacity value for the random_technology in the last year.
        """
        # Read the total capacity CSV file into a DataFrame
        total_capacity_df = pd.read_csv(total_capacity_path)

        # Identify the last year available in the 'Datetime' column
        last_year = total_capacity_df['Datetime'].max()

        # Filter the DataFrame to only include data for the last year
        last_year_df = total_capacity_df[total_capacity_df['Datetime'] == last_year]

        # Retrieve the capacity value for the max_technology in the last year
        max_technology_value = last_year_df[last_year_df['Technology'] == max_technology]['Value'].values[0]

        # Retrieve the capacity value for the random_technology in the last year
        random_technology_value = last_year_df[last_year_df['Technology'] == random_technology]['Value'].values[0]

        return last_year, max_technology_value, random_technology_value

def calculate_total_cost(self, run_result_path):
        """
        Calculate the total cost for a given run based on the 'tech_cost.csv' file.

        This function reads a CSV file containing the cost data for various technologies and sums the values 
        to calculate the total cost for the run. If the file is not found, it attempts to generate it by 
        calling another method and raising an error if the file still doesn't exist.

        Parameters:
        run_result_path (str): Path to the directory where the run results are stored.

        Returns:
        float: The total cost for the run.
        """
        
        # Define the path to the 'tech_cost.csv' file within the results directory
        tech_cost_file_path = os.path.join(run_result_path, "tech_cost.csv")

        # Check if the 'tech_cost.csv' file exists at the specified path
        if not os.path.exists(tech_cost_file_path):
            print(f"Warning: {tech_cost_file_path} not found. Attempting to generate it.")
            
            # Attempt to generate or save the file by calling the 'to_csv' method
            self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")
            
            # If the file still doesn't exist after attempting to generate it, raise an error
            if not os.path.exists(tech_cost_file_path):
                raise FileNotFoundError(f"{tech_cost_file_path} not found after attempting to generate it. Ensure the results are saved correctly.")
        
        # Read the 'tech_cost.csv' file into a DataFrame
        tech_cost_df = pd.read_csv(tech_cost_file_path)
        
        # Calculate the total cost by summing the values in the 'Value' column of the DataFrame
        total_cost = tech_cost_df['Value'].sum()

        # Return the total cost for the run
        return total_cost

def calculate_total_emissions(self, run_result_path):
        """
        Calculate the total CO2 emissions for a given run based on the 'emissions.csv' file.

        This function reads a CSV file containing emissions data for different technologies. It sums the 
        values in the 'Value' column to calculate the total CO2 emissions for the run. If the file is missing, 
        it returns 0.0 and prints a warning.

        Parameters:
        run_result_path (str): Path to the directory where the run results are stored.

        Returns:
        float: The total CO2 emissions for the run.
        """
        
        # Define the path to the 'emissions.csv' file in the results directory
        emissions_file_path = os.path.join(run_result_path, "emissions.csv")

        # Check if the 'emissions.csv' file exists at the specified path
        if not os.path.exists(emissions_file_path):
            print(f"Warning: {emissions_file_path} not found. Unable to calculate emissions.")
            return 0.0  # Return 0.0 if the file is not found
        
        # Read the 'emissions.csv' file into a DataFrame
        emissions_df = pd.read_csv(emissions_file_path)
        
        # Calculate the total CO2 emissions by summing the values in the 'Value' column
        total_emissions = emissions_df['Value'].sum()

        # Return the total CO2 emissions for the run
        return total_emissions

def record_run_results(self, result_path, run_data):
        """
        Record the results of each run into an Excel file.

        This function saves the details of multiple runs (stored in `run_data`) into an Excel file. 
        It uses the provided path to store the file, creates a DataFrame with the run data, and 
        writes it to an Excel file with the appropriate columns, including max and random 
        technologies with their values and orders.

        Parameters:
        result_path (str): Path to the directory where the results are stored.
        run_data (list): A list of tuples, where each tuple contains details of a single run. Each 
                        tuple should have the following information:
                        - Run Number
                        - Max Technologies (names, values, orders)
                        - Random Technologies (names, values, orders)
                        - Total Cost
                        - Total Emissions
        """
        
        # Define the path for the results Excel file
        results_file = os.path.join(result_path, "run_results.xlsx")

        # List of columns to store in the Excel file
        columns = [
            "Run Number",
            "Max Tech 1", "Max Tech 1 Value",
            "Max Tech 2", "Max Tech 2 Value",
            "Max Tech 3", "Max Tech 3 Value",
            "Random Tech 1",
            "Random Tech 2",
            "Random Tech 3",
            "Total Cost",
            "Total Emissions"
        ]
        
        # Create a DataFrame from the run data using the specified columns
        results_df = pd.DataFrame(run_data, columns=columns)

        # Save the DataFrame as an Excel file, without the index column
        results_df.to_excel(results_file, index=False)

def modify_input_parameters_combine(self, new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping, reset_specific=False):
        """
        Modify the input parameters in the 'Min_totalcap' and 'Max_totalcap' sheets based on the results of 
        the current run and adjust capacity limits for max and random technologies.

        This function reads an Excel file containing parameter sheets, modifies the capacity limits for the 
        specified max and random technologies, and then saves the modified sheets back to the file.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the directory containing the 'parameters_reg1.xlsx' file.
        C (float): Adjustment factor to modify the capacity limits.
        max_technology (str): The technology with the maximum value in the new capacity data.
        random_technology (str): The randomly selected technology from the new capacity data.
        technology_order_mapping (dict): A dictionary mapping technologies to their order in the parameter file.
        reset_specific (bool): If True, resets specific capacity values for max and random technologies.

        Returns:
        None
        """
        
        # Define the path to the 'parameters_reg1.xlsx' file
        param_file_path = os.path.join(param_path, "parameters_reg1.xlsx")


        # Find technologies and their corresponding values and orders
        (max_technology, x, max_order,
            _, _, _,
            _, _, _,
            random_technology, u, random_order,
            _, _, _,
            _, _, _,
            max_technologies, random_technologies,
            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # Find the values for the max and random technologies in the last year of total capacity
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        
        # Load the Excel file containing the parameters
        excel_data = pd.ExcelFile(param_file_path)

        # Read all sheets from the Excel file into a dictionary
        sheets_dict = pd.read_excel(excel_data, sheet_name=None, header=None)

        # Read the 'Min_totalcap' and 'Max_totalcap' sheets into DataFrames
        min_totalcap_df = sheets_dict['Min_totalcap'].copy()
        max_totalcap_df = sheets_dict['Max_totalcap'].copy()

        # Debugging: Print the initial state of the DataFrames
        print("Initial Min_totalcap DataFrame:")
        print(min_totalcap_df.head())
        print("Initial Max_totalcap DataFrame:")
        print(max_totalcap_df.head())

        if reset_specific:
            # If reset_specific is True, reset the capacity limits for max and random technologies only
            max_tech_col_index = max_order # Column index for max_technology (1-based index)
            random_tech_col_index = random_order # Column index for random_technology (1-based index)

            # Set the minimum capacity for max_technology and random_technology to zero
            min_totalcap_df.iloc[3:, max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, random_tech_col_index] = 0

            # Set the maximum capacity for both technologies to a large number
            max_totalcap_df.iloc[3:, max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, random_tech_col_index] = 10**10
        else:
            # Get the list of years from column A (starting from the fourth row)
            years = min_totalcap_df.iloc[3:, 0].tolist()

            # Find the column indices for max and random technologies based on their order
            max_tech_col_index = max_order
            random_tech_col_index = random_order

            # Loop through the years and modify the capacity values
            for i, year in enumerate(years):
                if i == len(years) - 1:  # Only modify the last year for random technology
                    new_min_value = T2 + C  # Adjust the minimum capacity for random_technology
                    min_totalcap_df.iat[i + 3, random_tech_col_index] = new_min_value

                # Adjust the maximum capacity values for both max_technology and random_technology
                new_max_value_max_tech = T1 - C
                new_max_value_random_tech = T2 + C
                max_totalcap_df.iat[i + 3, max_tech_col_index] = new_max_value_max_tech
                max_totalcap_df.iat[i + 3, random_tech_col_index] = new_max_value_random_tech

        # Debugging: Print the modified DataFrames
        print("Modified Min_totalcap DataFrame:")
        print(min_totalcap_df.iloc[3:].head())
        print("Modified Max_totalcap DataFrame:")
        print(max_totalcap_df.iloc[3:].head())

        # Update the sheets dictionary with the modified DataFrames
        sheets_dict['Min_totalcap'] = min_totalcap_df
        sheets_dict['Max_totalcap'] = max_totalcap_df

        # Save all modified sheets back to the Excel file using a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            temp_file_path = tmp.name

        # Write the updated data back to the Excel file
        with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets_dict.items():
                # Write the modified sheets back to the Excel file, excluding headers
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        # Replace the original parameters file with the modified temporary file
        shutil.move(temp_file_path, param_file_path)

        # Debugging: Confirm that the file has been saved successfully
        print(f"Parameters modified and saved to {param_file_path}")

def modify_input_parameters_combine2(self, new_capacity_path, total_capacity_path, param_path, C1, C2, 
                                        max_technology, random_technology, second_max_technology, second_random_technology, 
                                        technology_order_mapping, reset_specific=False):
        """
        Modify the input parameters in the 'Min_totalcap' and 'Max_totalcap' sheets for two sets of technologies.

        This function adjusts capacity limits for primary and secondary max and random technologies, using their
        respective adjustment factors (C1 and C2).

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the directory containing the 'parameters_reg1.xlsx' file.
        C1 (float): Adjustment factor for the primary max and random technologies.
        C2 (float): Adjustment factor for the secondary max and random technologies.
        max_technology (str): The primary max technology.
        random_technology (str): The primary random technology.
        second_max_technology (str): The secondary max technology.
        second_random_technology (str): The secondary random technology.
        technology_order_mapping (dict): A dictionary mapping technologies to their order in the parameter file.
        reset_specific (bool): If True, resets specific capacity values for all technologies.

        Returns:
        None
        """

        # Define the path to the 'parameters_reg1.xlsx' file
        param_file_path = os.path.join(param_path, "parameters_reg1.xlsx")

        # Find technologies and their corresponding values and orders
        (max_technology, x, max_order,
            second_max_technology, x2, second_max_order,
            _, _, _,
            random_technology, u, random_order,
            second_random_technology, u2, second_random_order,
            _, _, _,
            max_technologies, random_technologies,
            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # Load the Excel file containing the parameters
        excel_data = pd.ExcelFile(param_file_path)

        # Read all sheets from the Excel file into a dictionary
        sheets_dict = pd.read_excel(excel_data, sheet_name=None, header=None)

        # Read the 'Min_totalcap' and 'Max_totalcap' sheets into DataFrames
        min_totalcap_df = sheets_dict['Min_totalcap'].copy()
        max_totalcap_df = sheets_dict['Max_totalcap'].copy()

        # Debugging: Print the initial state of the DataFrames
        print("Initial Min_totalcap DataFrame:")
        print(min_totalcap_df.head())
        print("Initial Max_totalcap DataFrame:")
        print(max_totalcap_df.head())

        # Find the last year values for the technologies
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        _, T3, T4 = self.find_last_year_values(total_capacity_path, second_max_technology, second_random_technology)

        if reset_specific:
            # Reset capacity limits for the four technologies
            max_tech_col_index = max_order
            second_max_tech_col_index = second_max_order
            random_tech_col_index = random_order
            second_random_tech_col_index = second_random_order

            min_totalcap_df.iloc[3:, max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, second_max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, random_tech_col_index] = 0
            min_totalcap_df.iloc[3:, second_random_tech_col_index] = 0
            
            max_totalcap_df.iloc[3:, max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, second_max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, random_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, second_random_tech_col_index] = 10**10
  
        else:
            # Get the list of years from column A (starting from the fourth row)
            years = min_totalcap_df.iloc[3:, 0].tolist()

            # Get the column indices for each technology
            max_tech_col_index = max_order
            random_tech_col_index = random_order
            second_max_tech_col_index = second_max_order
            second_random_tech_col_index = second_random_order

            # Loop through the years and modify the capacity values
            for i, year in enumerate(years):
                if i == len(years) - 1:  # Only modify the last year for random technologies
                    # Adjust the minimum capacity for the random technologies
                    min_totalcap_df.iat[i + 3, random_tech_col_index] = T2 + C1
                    min_totalcap_df.iat[i + 3, second_random_tech_col_index] = T4 + C2

                # Adjust the maximum capacity values for all max and random technologies
                max_totalcap_df.iat[i + 3, max_tech_col_index] = T1 - C1
                max_totalcap_df.iat[i + 3, random_tech_col_index] = T2 + C1
                max_totalcap_df.iat[i + 3, second_max_tech_col_index] = T3 - C2
                max_totalcap_df.iat[i + 3, second_random_tech_col_index] = T4 + C2

        # Debugging: Print the modified DataFrames
        print("Modified Min_totalcap DataFrame:")
        print(min_totalcap_df.iloc[3:].head())
        print("Modified Max_totalcap DataFrame:")
        print(max_totalcap_df.iloc[3:].head())

        # Update the sheets dictionary with the modified DataFrames
        sheets_dict['Min_totalcap'] = min_totalcap_df
        sheets_dict['Max_totalcap'] = max_totalcap_df

        # Save all modified sheets back to the Excel file using a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            temp_file_path = tmp.name

        # Write the updated data back to the Excel file
        with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets_dict.items():
                # Write the modified sheets back to the Excel file, excluding headers
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        # Replace the original parameters file with the modified temporary file
        shutil.move(temp_file_path, param_file_path)

        # Debugging: Confirm that the file has been saved successfully
        print(f"Parameters modified and saved to {param_file_path}")    

def modify_input_parameters_combine3(self, new_capacity_path, total_capacity_path, param_path, C1, C2, C3, 
                                        max_technology, random_technology, second_max_technology, second_random_technology,
                                        third_max_technology, third_random_technology, technology_order_mapping, reset_specific=False):
        """
        Modify the input parameters in the 'Min_totalcap' and 'Max_totalcap' sheets for three sets of technologies.

        This function adjusts capacity limits for primary, secondary, and tertiary max and random technologies,
        using their respective adjustment factors (C1, C2, C3).

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the directory containing the 'parameters_reg1.xlsx' file.
        C1, C2, C3 (float): Adjustment factors for the primary, secondary, and tertiary technologies.
        max_technology (str): The primary max technology.
        random_technology (str): The primary random technology.
        second_max_technology (str): The secondary max technology.
        second_random_technology (str): The secondary random technology.
        third_max_technology (str): The tertiary max technology.
        third_random_technology (str): The tertiary random technology.
        technology_order_mapping (dict): A dictionary mapping technologies to their order in the parameter file.
        reset_specific (bool): If True, resets specific capacity values for all technologies.

        Returns:
        None
        """
        # Define the path to the 'parameters_reg1.xlsx' file
        param_file_path = os.path.join(param_path, "parameters_reg1.xlsx")

        # Find technologies and their corresponding values and orders
        (max_technology, _, max_order,
            second_max_technology, _, second_max_order,
            third_max_technology, _, third_max_order,
            random_technology, _, random_order,
            second_random_technology, _, second_random_order,
            third_random_technology, _, third_random_order,
            _, _,
            _) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # Load the Excel file containing the parameters
        excel_data = pd.ExcelFile(param_file_path)

        # Read all sheets from the Excel file into a dictionary
        sheets_dict = pd.read_excel(excel_data, sheet_name=None, header=None)

        # Read the 'Min_totalcap' and 'Max_totalcap' sheets into DataFrames
        min_totalcap_df = sheets_dict['Min_totalcap'].copy()
        max_totalcap_df = sheets_dict['Max_totalcap'].copy()

        # Debugging: Print the initial state of the DataFrames
        print("Initial Min_totalcap DataFrame:")
        print(min_totalcap_df.head())
        print("Initial Max_totalcap DataFrame:")
        print(max_totalcap_df.head())

        # Find the last year values for the technologies
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        _, T3, T4 = self.find_last_year_values(total_capacity_path, second_max_technology, second_random_technology)
        _, T5, T6 = self.find_last_year_values(total_capacity_path, third_max_technology, third_random_technology)

        if reset_specific:
            # Reset capacity limits for the six technologies
            max_tech_col_index = max_order
            second_max_tech_col_index = second_max_order
            third_max_tech_col_index = third_max_order
            random_tech_col_index = random_order
            second_random_tech_col_index = second_random_order
            third_random_tech_col_index = third_random_order
            

            min_totalcap_df.iloc[3:, max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, second_max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, third_max_tech_col_index] = 0
            min_totalcap_df.iloc[3:, random_tech_col_index] = 0
            min_totalcap_df.iloc[3:, second_random_tech_col_index] = 0
            min_totalcap_df.iloc[3:, third_max_tech_col_index] = 0
            
            max_totalcap_df.iloc[3:, max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, second_max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, third_max_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, random_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, second_random_tech_col_index] = 10**10
            max_totalcap_df.iloc[3:, third_max_tech_col_index] = 10**10

        else:
            # Get the list of years from column A (starting from the fourth row)
            years = min_totalcap_df.iloc[3:, 0].tolist()

            # Get the column indices for each technology
            max_tech_col_index = max_order
            random_tech_col_index = random_order
            second_max_tech_col_index = second_max_order
            second_random_tech_col_index = second_random_order
            third_max_tech_col_index = third_max_order
            third_random_tech_col_index = third_random_order

            # Loop through the years and modify the capacity values
            for i, year in enumerate(years):
                if i == len(years) - 1:  # Only modify the last year for random technologies
                    # Adjust the minimum capacity for the random technologies
                    min_totalcap_df.iat[i + 3, random_tech_col_index] = T2 + C1
                    min_totalcap_df.iat[i + 3, second_random_tech_col_index] = T4 + C2
                    min_totalcap_df.iat[i + 3, third_random_tech_col_index] = T6 + C3

                # Adjust the maximum capacity values for all max technologies
                max_totalcap_df.iat[i + 3, max_tech_col_index] = T1 - C1
                max_totalcap_df.iat[i + 3, random_tech_col_index] = T2 + C1
                max_totalcap_df.iat[i + 3, second_max_tech_col_index] = T3 - C2
                max_totalcap_df.iat[i + 3, second_random_tech_col_index] = T4 + C2
                max_totalcap_df.iat[i + 3, third_max_tech_col_index] = T5 - C3
                max_totalcap_df.iat[i + 3, third_random_tech_col_index] = T6 + C3

        # Debugging: Print the modified DataFrames
        print("Modified Min_totalcap DataFrame:")
        print(min_totalcap_df.iloc[3:].head())
        print("Modified Max_totalcap DataFrame:")
        print(max_totalcap_df.iloc[3:].head())

        # Update the sheets dictionary with the modified DataFrames
        sheets_dict['Min_totalcap'] = min_totalcap_df
        sheets_dict['Max_totalcap'] = max_totalcap_df

        # Save all modified sheets back to the Excel file using a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            temp_file_path = tmp.name

        # Write the updated data back to the Excel file
        with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
            for sheet_name, df in sheets_dict.items():
                # Write the modified sheets back to the Excel file, excluding headers
                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

        # Replace the original parameters file with the modified temporary file
        shutil.move(temp_file_path, param_file_path)

        # Debugging: Confirm that the file has been saved successfully
        print(f"Parameters modified and saved to {param_file_path}")

def run_with_adjustments(self, new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver='gurobi'):
        """
        Run the model and adjust parameters if no results are obtained or if the total cost exceeds the allowable limit.

        This function runs a model using an optimization solver, adjusting input parameters if no results are obtained 
        or if the total cost exceeds the specified maximum allowable cost. It repeatedly modifies the capacity limits 
        (adjusted by parameter C) and tries again until a valid result within the allowable cost is found or it changes 
        the randomly selected technology after a set number of attempts.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the parameters Excel file.
        technology_order_mapping (dict): A dictionary mapping technology names to their order.
        maximum_allowable_cost (float): The maximum total cost allowed for a successful run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.

        Returns:
        tuple: Contains the following elements:
            - C (float): The final adjustment factor used.
            - max_technology (str): The technology with the maximum value.
            - random_technology (str): The randomly selected technology.
            - T1 (float): The total capacity of max_technology in the last year.
            - T2 (float): The total capacity of random_technology in the last year.
            - total_cost (float): The total cost of the final run.
        """
            
        # Unpack values from find_technologies
        (max_technology, x, max_tech_order,
            second_max_technology, x2, second_max_tech_order,
            third_max_technology, x3, third_max_tech_order,
            random_technology, u, random_tech_order,
            second_random_technology, u2, second_random_tech_order,
            third_random_technology, u3, third_random_tech_order,
            max_technologies, random_technologies,
            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # Existing logic that uses max_technology and random_technology remains unchanged
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        
        # Adjust the logic to keep using only max_technology and random_technology
        initial_C = (x + u) / 2
        C = initial_C
        attempt = 0

        while self.results is None or (self.results is not None and self.calculate_total_cost(param_path) > maximum_allowable_cost):
            if self.results is not None:
                total_cost = self.calculate_total_cost(param_path)
                if total_cost <= maximum_allowable_cost:
                    return C, max_technology, random_technology, T1, T2, total_cost

            # Modify input parameters for the single max_technology and random_technology
            self.modify_input_parameters_combine(
                new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping
            )

            # Reload input data and re-run the model
            self.read_input_data(param_path)
            self.run(solver=solver, verbosity=True, force_rewrite=True)

            # Adjust C for the next attempt
            attempt += 1
            C = initial_C - (0.05 * initial_C * attempt)

            # Change random technology if too many attempts are made
            if attempt % 10 == 0 or C <= 0:
                return self.run_with_adjustments3(
                    new_capacity_path,
                    total_capacity_path,
                    param_path,
                    technology_order_mapping,
                    maximum_allowable_cost,
                    solver=solver
                )

        total_cost = self.calculate_total_cost(param_path)
        return C, max_technology, random_technology, T1, T2, total_cost

def run_with_adjustments2(self, new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver='gurobi'):
        """
        Run the model and adjust parameters if no results are obtained or if the total cost exceeds the allowable limit.

        This function is similar to `run_with_adjustments`, but it handles two sets of technologies:
        max and random technologies (primary and secondary). Adjustments are made for both sets simultaneously.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the parameters Excel file.
        technology_order_mapping (dict): A dictionary mapping technology names to their order.
        maximum_allowable_cost (float): The maximum total cost allowed for a successful run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.

        Returns:
        tuple: Contains the following elements:
            - C1 (float): The final adjustment factor for the primary set of technologies.
            - C2 (float): The final adjustment factor for the secondary set of technologies.
            - max_technology (str): The primary max technology.
            - random_technology (str): The primary random technology.
            - second_max_technology (str): The secondary max technology.
            - second_random_technology (str): The secondary random technology.
            - T1 (float): The total capacity of max_technology in the last year.
            - T2 (float): The total capacity of random_technology in the last year.
            - T3 (float): The total capacity of second_max_technology in the last year.
            - T4 (float): The total capacity of second_random_technology in the last year.
            - total_cost (float): The total cost of the final run.
        """
        # Unpack values from find_technologies
        (max_technology, x, max_tech_order,
            second_max_technology, x2, second_max_tech_order,
            third_max_technology, x3, third_max_tech_order,
            random_technology, u, random_tech_order,
            second_random_technology, u2, second_random_tech_order,
            third_random_technology, u3, third_random_tech_order,
            max_technologies, random_technologies,
            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # # Get values for secondary technologies
        # a1 = next((value for tech, value, _ in technologies_list if tech == second_max_technology), 0)
        # b1 = next((value for tech, value, _ in technologies_list if tech == second_random_technology), 0)

        # Get last-year values for primary and secondary technologies
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        _, T3, T4 = self.find_last_year_values(total_capacity_path, second_max_technology, second_random_technology)

        # Initialize adjustment factors
        initial_C1 = (x + u) / 2
        C1 = initial_C1

        initial_C2 = (x2 + u2) / 2
        C2 = initial_C2

        attempt = 0

        while self.results is None or (self.results is not None and self.calculate_total_cost(param_path) > maximum_allowable_cost):
            if self.results is not None:
                total_cost = self.calculate_total_cost(param_path)
                if total_cost <= maximum_allowable_cost:
                    return C1, C2, max_technology, random_technology, second_max_technology, second_random_technology, T1, T2, T3, T4, total_cost

            # Modify input parameters for the primary and secondary technologies
            self.modify_input_parameters_combine2(
                new_capacity_path, total_capacity_path, param_path,
                C1, C2,
                max_technology, random_technology,
                second_max_technology, second_random_technology,
                technology_order_mapping
            )

            # Reload input data and re-run the model
            self.read_input_data(param_path)
            self.run(solver=solver, verbosity=True, force_rewrite=True)

            # Adjust both C1 and C2 for the next attempt
            attempt += 1
            C1 = initial_C1 - (0.05 * initial_C1 * attempt)
            C2 = initial_C2 - (0.05 * initial_C2 * attempt)

            # Change random technologies if too many attempts are made
            if attempt % 10 == 0 or C1 <= 0 or C2 <= 0:
                return self.run_with_adjustments3(
                    new_capacity_path,
                    total_capacity_path,
                    param_path,
                    technology_order_mapping,
                    maximum_allowable_cost,
                    solver=solver
                )

        total_cost = self.calculate_total_cost(param_path)
        return C1, C2, max_technology, random_technology, second_max_technology, second_random_technology, T1, T2, T3, T4, total_cost

def run_with_adjustments3(self, new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver='gurobi'):
        """
        Run the model and adjust parameters if no results are obtained or if the total cost exceeds the allowable limit.

        This function is similar to `run_with_adjustments2`, but it handles three sets of technologies:
        max and random technologies (primary, secondary, and tertiary). Adjustments are made for all sets simultaneously.

        Parameters:
        new_capacity_path (str): Path to the CSV file containing new capacity data.
        total_capacity_path (str): Path to the CSV file containing total capacity data.
        param_path (str): Path to the parameters Excel file.
        technology_order_mapping (dict): A dictionary mapping technology names to their order.
        maximum_allowable_cost (float): The maximum total cost allowed for a successful run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.

        Returns:
        tuple: Contains adjustment factors, technologies, capacities, and total cost.
        """
        # Unpack values from find_technologies
        (max_technology, x, max_tech_order,
            second_max_technology, x2, second_max_tech_order,
            third_max_technology, x3, third_max_tech_order,
            random_technology, u, random_tech_order,
            second_random_technology, u2, second_random_tech_order,
            third_random_technology, u3, third_random_tech_order,
            max_technologies, random_technologies,
            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

        # # Get values for secondary and tertiary technologies
        # a1 = next((value for tech, value, _ in technologies_list if tech == second_max_technology), 0)
        # b1 = next((value for tech, value, _ in technologies_list if tech == second_random_technology), 0)
        # c1 = next((value for tech, value, _ in technologies_list if tech == third_max_technology), 0)
        # d1 = next((value for tech, value, _ in technologies_list if tech == third_random_technology), 0)

        # Get last-year values for all technologies
        _, T1, T2 = self.find_last_year_values(total_capacity_path, max_technology, random_technology)
        _, T3, T4 = self.find_last_year_values(total_capacity_path, second_max_technology, second_random_technology)
        _, T5, T6 = self.find_last_year_values(total_capacity_path, third_max_technology, third_random_technology)

        # Initialize adjustment factors
        initial_C1 = (x + u) / 2
        C1 = initial_C1

        initial_C2 = (x2 + u2) / 2
        C2 = initial_C2

        initial_C3 = (x3 + u3) / 2
        C3 = initial_C3

        attempt = 0

        while self.results is None or (self.results is not None and self.calculate_total_cost(param_path) > maximum_allowable_cost):
            if self.results is not None:
                total_cost = self.calculate_total_cost(param_path)
                if total_cost <= maximum_allowable_cost:
                    return (
                        C1, C2, C3,
                        max_technology, random_technology,
                        second_max_technology, second_random_technology,
                        third_max_technology, third_random_technology,
                        T1, T2, T3, T4, T5, T6, total_cost
                    )

            # Modify input parameters for all three sets of technologies
            self.modify_input_parameters_combine3(
                new_capacity_path, total_capacity_path, param_path,
                C1, C2, C3,
                max_technology, random_technology,
                second_max_technology, second_random_technology,
                third_max_technology, third_random_technology,
                technology_order_mapping
            )

            # Reload input data and re-run the model
            self.read_input_data(param_path)
            self.run(solver=solver, verbosity=True, force_rewrite=True)

            # Adjust all Cs for the next attempt
            attempt += 1
            C1 = initial_C1 - (0.05 * initial_C1 * attempt)
            C2 = initial_C2 - (0.05 * initial_C2 * attempt)
            C3 = initial_C3 - (0.05 * initial_C3 * attempt)

            # Change random technologies if too many attempts are made
            if attempt % 10 == 0 or C1 <= 0 or C2 <= 0 or C3 <= 0:
                # Start the function over again
                return self.run_with_adjustments3(
                    new_capacity_path,
                    total_capacity_path,
                    param_path,
                    technology_order_mapping,
                    maximum_allowable_cost,
                    solver=solver
                )

        total_cost = self.calculate_total_cost(param_path)
        return (
            C1, C2, C3,
            max_technology, random_technology,
            second_max_technology, second_random_technology,
            third_max_technology, third_random_technology,
            T1, T2, T3, T4, T5, T6, total_cost
        )

def run_multiple_solutions(self, number_solutions, param_path, result_path, slack, solver='gurobi', NOS_quality=1):
        """
        Run the model multiple times, adjusting parameters and recording results for each run.

        Parameters:
        number_solutions (int): The total number of successful runs to complete.
        param_path (str): Path to the directory containing the parameters for the model.
        result_path (str): Path to the directory where run results will be saved.
        slack (float): Percentage to add as a buffer to the optimal cost for each run.
        solver (str): Solver to be used for optimization. Default is 'gurobi'.
        NOS_quality (int): Determines the quality level of the decision space (1, 2, or 3).
        """

        # Validate NOS_quality input
        if NOS_quality not in [1, 2, 3]:
            raise ValueError(f"NOS_quality must be 1, 2, or 3. Received: {NOS_quality}")

        # Create a copy of the initial parameters directory for each run
        initial_param_dir = "Thesishypatia/SM01/parameters_initial"
        if os.path.exists(initial_param_dir):
            shutil.rmtree(initial_param_dir)  # Remove the directory if it exists
            shutil.copytree(param_path, initial_param_dir)  # Copy the parameter directory for the initial state

        second_max_tech_value_new = None
        third_max_tech_value_new = None

        run_data = []  # List to store data from each run

        if NOS_quality == 1:
            successful_runs = 0  # Counter for successful runs
            attempt = 0  # Counter for attempts made
            technology_order_mapping = None  # Will store the mapping of technologies to their order
            optimal_cost = None  # Store the optimal cost from the first successful run
            maximum_allowable_cost = None  # Store the maximum allowable cost for subsequent runs

            while successful_runs < number_solutions:
                # Reset parameters to initial state before each run
                if os.path.exists(param_path):
                    shutil.rmtree(param_path)
                shutil.copytree(initial_param_dir, param_path)

                # Define the result path for the current run
                run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                os.makedirs(run_result_path, exist_ok=True)

                if successful_runs == 0:
                    # For the first run, read input data and run the model
                    self.read_input_data(param_path)
                    self.run(solver=solver, verbosity=True, force_rewrite=True)

                    # Initialize paths for capacity data from the first run
                    new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                    total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                    # Check if the new_capacity file was created, generate it if missing
                    if not os.path.exists(new_capacity_path):
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                    if not os.path.exists(new_capacity_path):
                        raise FileNotFoundError(f"{new_capacity_path} not found after the initial run.")

                    # Identify the order of technologies based on the first run
                    technology_order_mapping = self.identify_technology_order(new_capacity_path, result_path)

                    # Find technologies for the first run
                    (max_technology, x, max_tech_order,
                        second_max_technology, x2, second_max_tech_order,
                        third_max_technology, x3, third_max_tech_order,
                        random_technology, u, random_tech_order,
                        second_random_technology, u2, second_random_tech_order,
                        third_random_technology, u3, third_random_tech_order,
                        max_technologies, random_technologies,
                        technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                    initial_C = (x + u) / 2  # Calculate initial adjustment factor C
                    C = initial_C

                    # Calculate the total cost for the first run and determine the maximum allowable cost
                    optimal_cost = self.calculate_total_cost(run_result_path)
                    maximum_allowable_cost = optimal_cost * (1 + slack / 100)  # Add slack percentage to the cost limit

                    # Calculate the total emissions for the first run
                    total_emissions = self.calculate_total_emissions(run_result_path)

                else:
                    # For subsequent runs, load data from the previous successful run
                    new_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "new_capacity.csv")
                    total_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "total_capacity.csv")

                    if not os.path.exists(new_capacity_path):
                        raise FileNotFoundError(f"{new_capacity_path} not found for the run.")

                    if attempt == 0:
                        # Adjust parameters and find max and random technologies only once per run
                        C, max_technology, random_technology, T1, T2, total_cost = self.run_with_adjustments(
                            new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                        )
                        initial_C = C  # Set initial C for subsequent adjustments
                    else:
                        # Adjust C by reducing it further with each attempt
                        C = initial_C - (0.05 * initial_C * attempt)
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                    self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping)
                    self.read_input_data(param_path)  # Ensure modified parameters are loaded
                    self.run(solver=solver, verbosity=True, force_rewrite=True)

                if self.results is not None:
                    # Save the results after a successful run
                    self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                    # Update paths for the current successful run
                    new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                    total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                    # Calculate the total cost for the current run
                    total_cost = self.calculate_total_cost(run_result_path)

                    # Calculate the total emissions for the current run
                    total_emissions = self.calculate_total_emissions(run_result_path)

                    # If the total cost is within the allowable range, record the results
                    if total_cost <= maximum_allowable_cost:
                        _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                            total_capacity_path, max_technology, random_technology
                        )

                        # Recalculate the sum of new capacities for consistency
                        new_capacity_df = pd.read_csv(new_capacity_path)
                        max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                        #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()

                        # Record the results
                        run_data.append((
                            successful_runs,
                            max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                            max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                            second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                            second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                            third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                            third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                            random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                            second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                            third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                            total_cost,  # Total Cost
                            total_emissions  # Total Emissions
                        ))

                        successful_runs += 1  # Increment successful runs count
                        attempt = 0  # Reset attempt counter for the next run

                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                        C = initial_C - (0.05 * initial_C * attempt)  # Reduce C further
                else:
                    attempt += 1
                    print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                    C = initial_C - (0.05 * initial_C * attempt)  # Reduce C for retries

        elif NOS_quality == 2:
            # NOS_quality = 2: Split into two phases
            successful_runs = 0
            attempt = 0
            technology_order_mapping = None  # Will store the mapping of technologies to their order
            optimal_cost = None  # Store the optimal cost from the first successful run
            maximum_allowable_cost = None  # Store the maximum allowable cost for subsequent runs

            while successful_runs < number_solutions:
                if successful_runs <= number_solutions / 2:
                    # Reset parameters to initial state before each run
                    if os.path.exists(param_path):
                        shutil.rmtree(param_path)
                    shutil.copytree(initial_param_dir, param_path)

                    # Define the result path for the current run
                    run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                    os.makedirs(run_result_path, exist_ok=True)

                    if successful_runs == 0:
                        # For the first run, read input data and run the model
                        self.read_input_data(param_path)
                        self.run(solver=solver, verbosity=True, force_rewrite=True)

                        # Initialize paths for capacity data from the first run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Check if the new_capacity file was created, generate it if missing
                        if not os.path.exists(new_capacity_path):
                            self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        if not os.path.exists(new_capacity_path):
                            raise FileNotFoundError(f"{new_capacity_path} not found after the initial run.")

                        # Identify the order of technologies based on the first run
                        technology_order_mapping = self.identify_technology_order(new_capacity_path, result_path)

                        # Find technologies for the first run
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                        initial_C = (x + u) / 2  # Calculate initial adjustment factor C
                        C = initial_C

                        # Calculate the total cost for the first run and determine the maximum allowable cost
                        optimal_cost = self.calculate_total_cost(run_result_path)
                        maximum_allowable_cost = optimal_cost * (1 + slack / 100)  # Add slack percentage to the cost limit

                        # Calculate the total emissions for the first run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                    else:
                        # For subsequent runs, load data from the previous successful run
                        new_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "new_capacity.csv")
                        total_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "total_capacity.csv")

                        if not os.path.exists(new_capacity_path):
                            raise FileNotFoundError(f"{new_capacity_path} not found for the run.")

                        if attempt == 0:
                            # Adjust parameters and find max and random technologies only once per run
                            C, max_technology, random_technology, T1, T2, total_cost = self.run_with_adjustments(
                                new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                            )
                            initial_C = C  # Set initial C for subsequent adjustments

                        else:
                            # Adjust C by reducing it further with each attempt
                            C = initial_C - (0.05 * initial_C * attempt)
                            
                            (max_technology, x, max_tech_order,
                                second_max_technology, x2, second_max_tech_order,
                                third_max_technology, x3, third_max_tech_order,
                                random_technology, u, random_tech_order,
                                second_random_technology, u2, second_random_tech_order,
                                third_random_technology, u3, third_random_tech_order,
                                max_technologies, random_technologies,
                                technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                        self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping)
                        self.read_input_data(param_path)  # Ensure modified parameters are loaded
                        self.run(solver=solver, verbosity=True, force_rewrite=True)

                    if self.results is not None:
                        # Save the results after a successful run
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        # Update paths for the current successful run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Calculate the total cost for the current run
                        total_cost = self.calculate_total_cost(run_result_path)

                        # Calculate the total emissions for the current run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                        # If the total cost is within the allowable range, record the results
                        if total_cost <= maximum_allowable_cost:
                            _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, max_technology, random_technology
                            )

                            # Recalculate the sum of new capacities for consistency
                            new_capacity_df = pd.read_csv(new_capacity_path)
                            max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                            #second_max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_max_technology]['Value'].sum()
                            #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()
                            #second_random_technology = new_capacity_df[new_capacity_df['Technology'] == second_random_technology]['Value'].sum()

                            # Record the results
                            run_data.append((
                                successful_runs,
                                max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                                max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                                second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                                second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                                third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                                third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                                random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                                second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                                third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                                total_cost,  # Total Cost
                                total_emissions  # Total Emissions
                            ))

                            successful_runs += 1  # Increment successful runs count
                            attempt = 0  # Reset attempt counter for the next run

                        else:
                            attempt += 1
                            print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                            C = initial_C - (0.05 * initial_C * attempt)  # Reduce C further
                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                        C = initial_C - (0.05 * initial_C * attempt)  # Reduce C further
                    
                else:  # Second half (successful_run >= number_solutions / 2)

                    # Reset parameters to initial state before each run
                    if os.path.exists(param_path):
                        shutil.rmtree(param_path)
                    shutil.copytree(initial_param_dir, param_path)

                    # Define the result path for the current run
                    run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                    os.makedirs(run_result_path, exist_ok=True)

                    # For subsequent runs, load data from the previous successful run
                    new_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "new_capacity.csv")
                    total_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "total_capacity.csv")

                    if not os.path.exists(new_capacity_path):
                        raise FileNotFoundError(f"{new_capacity_path} not found for the run.")

                    if attempt == 0:
                        # Adjust parameters and find max and random technologies for primary and secondary sets
                        C1, C2, max_technology, random_technology, second_max_technology, second_random_technology, T1, T2, T3, T4, total_cost = self.run_with_adjustments2(
                            new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                        )
                        initial_C1 = C1  # Set initial C1 for subsequent adjustments
                        initial_C2 = C2  # Set initial C2 for subsequent adjustments
                    else:
                        # Adjust C1 and C2 by reducing them further with each attempt
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)

                        # Identify the technologies again
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

                    # Modify input parameters for both sets of technologies
                    self.modify_input_parameters_combine2(
                        new_capacity_path, total_capacity_path, param_path,
                        C1, C2,
                        max_technology, random_technology,
                        second_max_technology, second_random_technology,
                        technology_order_mapping
                    )

                    self.read_input_data(param_path)  # Ensure modified parameters are loaded
                    self.run(solver=solver, verbosity=True, force_rewrite=True)

                    if self.results is not None:
                        # Save the results after a successful run
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        # Update paths for the current successful run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Calculate the total cost for the current run
                        total_cost = self.calculate_total_cost(run_result_path)

                        # Calculate the total emissions for the current run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                        # If the total cost is within the allowable range, record the results
                        if total_cost <= maximum_allowable_cost:
                            _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, max_technology, random_technology
                            )
                            _, second_max_tech_value_total, second_random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, second_max_technology, second_random_technology
                            )

                            # Recalculate the sum of new capacities for consistency
                            new_capacity_df = pd.read_csv(new_capacity_path)
                            max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                            #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()
                            second_max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_max_technology]['Value'].sum()
                            #second_random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_random_technology]['Value'].sum()

                            # Record the results
                            run_data.append((
                                successful_runs,
                                max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                                max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                                second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                                second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                                third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                                third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                                random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                                second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                                third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                                total_cost,  # Total Cost
                                total_emissions  # Total Emissions
                            ))

                            successful_runs += 1  # Increment successful runs count
                            attempt = 0  # Reset attempt counter for the next run

                        else:
                            attempt += 1
                            print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                            C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                            C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)

        elif NOS_quality == 3:
            successful_runs = 0
            attempt = 0
            technology_order_mapping = None  # Will store the mapping of technologies to their order
            optimal_cost = None  # Store the optimal cost from the first successful run
            maximum_allowable_cost = None  # Store the maximum allowable cost for subsequent runs

            while successful_runs < number_solutions:
                if successful_runs <= number_solutions / 3:
                    # Reset parameters to initial state before each run
                    if os.path.exists(param_path):
                        shutil.rmtree(param_path)
                    shutil.copytree(initial_param_dir, param_path)

                    # Define the result path for the current run
                    run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                    os.makedirs(run_result_path, exist_ok=True)

                    if successful_runs == 0:
                        # For the first run, read input data and run the model
                        self.read_input_data(param_path)
                        self.run(solver=solver, verbosity=True, force_rewrite=True)

                        # Initialize paths for capacity data from the first run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Check if the new_capacity file was created, generate it if missing
                        if not os.path.exists(new_capacity_path):
                            self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        if not os.path.exists(new_capacity_path):
                            raise FileNotFoundError(f"{new_capacity_path} not found after the initial run.")

                        # Identify the order of technologies based on the first run
                        technology_order_mapping = self.identify_technology_order(new_capacity_path, result_path)

                        # Find technologies for the first run
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                        initial_C = (x + u) / 2  # Calculate initial adjustment factor C
                        C = initial_C

                        # Calculate the total cost for the first run and determine the maximum allowable cost
                        optimal_cost = self.calculate_total_cost(run_result_path)
                        maximum_allowable_cost = optimal_cost * (1 + slack / 100)  # Add slack percentage to the cost limit

                        # Calculate the total emissions for the first run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                    else:
                        # For subsequent runs, load data from the previous successful run
                        new_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "new_capacity.csv")
                        total_capacity_path = os.path.join(result_path, f"Run_{successful_runs - 1}", "total_capacity.csv")

                        if not os.path.exists(new_capacity_path):
                            raise FileNotFoundError(f"{new_capacity_path} not found for the run.")

                        if attempt == 0:
                            # Adjust parameters and find max and random technologies only once per run
                            C, max_technology, random_technology, T1, T2, total_cost = self.run_with_adjustments(
                                new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                            )
                            initial_C = C  # Set initial C for subsequent adjustments
                        else:
                            # Adjust C by reducing it further with each attempt
                            C = initial_C - (0.05 * initial_C * attempt)
                            (max_technology, x, max_tech_order,
                                second_max_technology, x2, second_max_tech_order,
                                third_max_technology, x3, third_max_tech_order,
                                random_technology, u, random_tech_order,
                                second_random_technology, u2, second_random_tech_order,
                                third_random_technology, u3, third_random_tech_order,
                                max_technologies, random_technologies,
                                technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                        self.modify_input_parameters_combine(new_capacity_path, total_capacity_path, param_path, C, max_technology, random_technology, technology_order_mapping)
                        self.read_input_data(param_path)  # Ensure modified parameters are loaded
                        self.run(solver=solver, verbosity=True, force_rewrite=True)

                    if self.results is not None:
                        # Save the results after a successful run
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        # Update paths for the current successful run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Calculate the total cost for the current run
                        total_cost = self.calculate_total_cost(run_result_path)

                        # Calculate the total emissions for the current run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                        # If the total cost is within the allowable range, record the results
                        if total_cost <= maximum_allowable_cost:
                            _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, max_technology, random_technology
                            )

                            # Recalculate the sum of new capacities for consistency
                            new_capacity_df = pd.read_csv(new_capacity_path)
                            max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                            #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()

                            # Record the results
                            run_data.append((
                                successful_runs,
                                max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                                max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                                second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                                second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                                third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                                third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                                random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                                second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                                third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                                total_cost,  # Total Cost
                                total_emissions  # Total Emissions
                            ))

                            successful_runs += 1  # Increment successful runs count
                            attempt = 0  # Reset attempt counter for the next run

                        else:
                            attempt += 1
                            print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                            C = initial_C - (0.05 * initial_C * attempt)  # Reduce C further
                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                        C = initial_C - (0.05 * initial_C * attempt)  # Reduce C for retries

                elif number_solutions / 3 < successful_runs <= 2 * number_solutions / 3:
                    if os.path.exists(param_path):
                        shutil.rmtree(param_path)
                    shutil.copytree(initial_param_dir, param_path)

                    run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                    os.makedirs(run_result_path, exist_ok=True)

                    if attempt == 0:
                        C1, C2, max_technology, random_technology, second_max_technology, second_random_technology, T1, T2, T3, T4, total_cost = self.run_with_adjustments2(
                            new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                        )
                        initial_C1, initial_C2 = C1, C2
                    else:
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)

                    self.modify_input_parameters_combine2(
                        new_capacity_path, total_capacity_path, param_path,
                        C1, C2,
                        max_technology, random_technology,
                        second_max_technology, second_random_technology,
                        technology_order_mapping
                    )
                    self.read_input_data(param_path)
                    self.run(solver=solver, verbosity=True, force_rewrite=True)

                    if self.results is not None:
                        # Save the results after a successful run
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        # Update paths for the current successful run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Calculate the total cost for the current run
                        total_cost = self.calculate_total_cost(run_result_path)

                        # Calculate the total emissions for the current run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                        # If the total cost is within the allowable range, record the results
                        if total_cost <= maximum_allowable_cost:
                            _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, max_technology, random_technology
                            )
                            _, second_max_tech_value_total, second_random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, second_max_technology, second_random_technology
                            )

                            # Recalculate the sum of new capacities for consistency
                            new_capacity_df = pd.read_csv(new_capacity_path)
                            max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                            #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()
                            second_max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_max_technology]['Value'].sum()
                            #second_random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_random_technology]['Value'].sum()

                            # Record the results
                            run_data.append((
                                successful_runs,
                                max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                                max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                                second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                                second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                                third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                                third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                                random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                                second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                                third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                                total_cost,  # Total Cost
                                total_emissions  # Total Emissions
                            ))

                            successful_runs += 1  # Increment successful runs count
                            attempt = 0  # Reset attempt counter for the next run

                        else:
                            attempt += 1
                            print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                            C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                            C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)

                else:
                    if os.path.exists(param_path):
                        shutil.rmtree(param_path)
                    shutil.copytree(initial_param_dir, param_path)

                    run_result_path = os.path.join(result_path, f"Run_{successful_runs}")
                    os.makedirs(run_result_path, exist_ok=True)

                    if attempt == 0:
                        C1, C2, C3, max_technology, random_technology, second_max_technology, second_random_technology, third_max_technology, third_random_technology, T1, T2, T3, T4, T5, T6, total_cost = self.run_with_adjustments3(
                            new_capacity_path, total_capacity_path, param_path, technology_order_mapping, maximum_allowable_cost, solver
                        )
                        initial_C1, initial_C2, initial_C3 = C1, C2, C3
                    else:
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                        C3 = initial_C3 - (0.05 * initial_C3 * attempt)
                        (max_technology, x, max_tech_order,
                            second_max_technology, x2, second_max_tech_order,
                            third_max_technology, x3, third_max_tech_order,
                            random_technology, u, random_tech_order,
                            second_random_technology, u2, second_random_tech_order,
                            third_random_technology, u3, third_random_tech_order,
                            max_technologies, random_technologies,
                            technologies_list) = self.find_technologies(new_capacity_path, technology_order_mapping)
                    self.modify_input_parameters_combine3(
                        new_capacity_path, total_capacity_path, param_path,
                        C1, C2, C3,
                        max_technology, random_technology,
                        second_max_technology, second_random_technology,
                        third_max_technology, third_random_technology,
                        technology_order_mapping
                    )
                    self.read_input_data(param_path)
                    self.run(solver=solver, verbosity=True, force_rewrite=True)

                    if self.results is not None:
                        # Save the results after a successful run
                        self.to_csv(run_result_path, force_rewrite=True, postprocessing_module="aggregated")

                        # Update paths for the current successful run
                        new_capacity_path = os.path.join(run_result_path, "new_capacity.csv")
                        total_capacity_path = os.path.join(run_result_path, "total_capacity.csv")

                        # Calculate the total cost for the current run
                        total_cost = self.calculate_total_cost(run_result_path)

                        # Calculate the total emissions for the current run
                        total_emissions = self.calculate_total_emissions(run_result_path)

                        # If the total cost is within the allowable range, record the results
                        if total_cost <= maximum_allowable_cost:
                            _, max_tech_value_total, random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, max_technology, random_technology
                            )
                            _, second_max_tech_value_total, second_random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, second_max_technology, second_random_technology
                            )
                            _, third_max_tech_value_total, third_random_tech_value_total = self.find_last_year_values(
                                total_capacity_path, third_max_technology, third_random_technology
                            )

                            # Recalculate the sum of new capacities for consistency
                            new_capacity_df = pd.read_csv(new_capacity_path)
                            max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == max_technology]['Value'].sum()
                            #random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == random_technology]['Value'].sum()
                            second_max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_max_technology]['Value'].sum()
                            #second_random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == second_random_technology]['Value'].sum()
                            third_max_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == third_max_technology]['Value'].sum()
                            #third_random_tech_value_new = new_capacity_df[new_capacity_df['Technology'] == third_random_technology]['Value'].sum()

                            # Record the results
                            run_data.append((
                                successful_runs,
                                max_technology if len(max_technologies) > 0 else None,  # Max Tech 1
                                max_tech_value_new if len(max_technologies) > 0 else None,  # Max Tech 1 Value
                                second_max_technology if len(max_technologies) > 1 else None,  # Max Tech 2
                                second_max_tech_value_new if len(max_technologies) > 1 else None,  # Max Tech 2 Value
                                third_max_technology if len(max_technologies) > 2 else None,  # Max Tech 3
                                third_max_tech_value_new if len(max_technologies) > 2 else None,  # Max Tech 3 Value
                                random_technology if len(random_technologies) > 0 else None,  # Random Tech 1
                                second_random_technology if len(random_technologies) > 1 else None,  # Random Tech 2
                                third_random_technology if len(random_technologies) > 2 else None,  # Random Tech 3
                                total_cost,  # Total Cost
                                total_emissions  # Total Emissions
                            ))

                            successful_runs += 1  # Increment successful runs count
                            attempt = 0  # Reset attempt counter for the next run

                        else:
                            attempt += 1
                            print(f"Attempt {attempt}: Total cost {total_cost} exceeds maximum allowable cost {maximum_allowable_cost}. Adjusting parameters and retrying.")
                            C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                            C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                            C3 = initial_C3 - (0.05 * initial_C3 * attempt)
                    else:
                        attempt += 1
                        print(f"Attempt {attempt}: No results. Adjusting parameters and retrying.")
                        C1 = initial_C1 - (0.05 * initial_C1 * attempt)
                        C2 = initial_C2 - (0.05 * initial_C2 * attempt)
                        C3 = initial_C3 - (0.05 * initial_C3 * attempt)

        # Record all run results in an Excel file
        self.record_run_results(result_path, run_data)

# Now we attach these methods dynamically to Model
Model.identify_technology_order = identify_technology_order
Model.find_technologies = find_technologies
Model.find_last_year_values = find_last_year_values
Model.calculate_total_cost = calculate_total_cost
Model.calculate_total_emissions = calculate_total_emissions
Model.record_run_results = record_run_results
Model.modify_input_parameters_combine = modify_input_parameters_combine
Model.modify_input_parameters_combine2 = modify_input_parameters_combine2
Model.modify_input_parameters_combine3 = modify_input_parameters_combine3
Model.run_with_adjustments = run_with_adjustments
Model.run_with_adjustments2 = run_with_adjustments2
Model.run_with_adjustments3 = run_with_adjustments3
Model.run_multiple_solutions = run_multiple_solutions

# End of Enhancements
# Kianjn