# Copyright 2023 Juan Francisco Rodriguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import dwave
import dimod
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from dimod import QuadraticModel, ConstrainedQuadraticModel, Binary, Integer
import numpy as np

from pulp import *
import timeit

class classical_implementation:

    def __init__(self, num_periods: int, generators: pd.DataFrame, demand: pd.DataFrame, variability: pd.DataFrame, start_shut: bool = False):
        """
        Initialize the Unit Commitment (UC) Annealing class.

        Parameters:
        - num_periods (int): The number of time periods.
        - generators (pd.DataFrame): DataFrame containing generator data.
        - demand (pd.DataFrame): DataFrame containing demand data.
        - variability (pd.DataFrame): DataFrame containing generator variability data.
        - start_shut (bool, optional): Flag indicating whether to consider startup/shutdown constraints. Default is False.

        Initializes various attributes and variables needed for UC problem solving.
        """

        # Save data
        self.demand = demand['Demand'][:num_periods].values
        self.generators = generators
        self.variability = variability

        # Parameters
        self.num_generators = len(generators)
        self.periods = num_periods
        self.size = self.num_generators * self.periods
        self.start_shut = start_shut

        # Variables
        self.gen = None
        self.commit = None
        if self.start_shut:
            self.start = None
            self.shut = None
    
    def classify_dataset(self) -> [pd.DataFrame]:
        """
        Classify generators into different types based on their characteristics.

        Returns:
        List of DataFrames, each containing generators of a specific type.
        - Thermal Generators
        - Non-Thermal Generators
        - Non-Variable Generators
        - Variable Generators
        - Non-Thermal Non-Variable Generators
        """

        # Classify generators into different types
        thermal_generators = self.generators[self.generators['Up_time'] > 0]
        non_thermal_generators = self.generators[self.generators['Up_time'] == 0]
        non_var_generators = self.generators[self.generators['IsVariable'] == False]
        var_generators = self.generators[self.generators['IsVariable'] == True]
        non_thermal_non_var_generators = non_thermal_generators.merge(non_var_generators, on='Resource')

        return thermal_generators, non_thermal_generators, non_var_generators, var_generators, non_thermal_non_var_generators

    def define_variables(self, thermal_generators) -> None:
        """
        Define optimization variables for the Unit Commitment problem.

        Parameters:
        - thermal_generators (pd.DataFrame): DataFrame containing thermal generator data.

        Initializes variables:
        - commitment variables
        - generation variables
        - startup variables (if start_shut is True)
        - shutdown variables (if start_shut is True)
        """

        # Commitment variable
        self.commit = LpVariable.dicts("commit", [(generator, t) for generator, rows in thermal_generators.iterrows() for t in range(self.periods)], cat="Binary")
        
        # Generation variable
        self.gen = LpVariable.dicts("gen", [(generator, t) for generator, rows in self.generators.iterrows() for t in range(self.periods)], cat="Integer")

        if self.start_shut:
            # Start-up variable
            self.start = LpVariable.dicts("start", [(generator, t) for generator, rows in thermal_generators.iterrows() for t in range(self.periods)], cat="Binary")

            # Shut-down variable
            self.shut = LpVariable.dicts("shut", [(generator, t) for generator, rows in thermal_generators.iterrows() for t in range(self.periods)], cat="Binary")
            

    def define_objective(self, model, non_var_generators, var_generators, thermal_generators) ->  None:
        """
        Define the optimization objective function for the Unit Commitment problem.

        Parameters:
        - model: The optimization model being defined.
        - non_var_generators (pd.DataFrame): DataFrame containing non-variable generator data.
        - var_generators (pd.DataFrame): DataFrame containing variable generator data.
        - thermal_generators (pd.DataFrame): DataFrame containing thermal generator data.

        Initializes the operating cost objective function based on generator characteristics.
        """

        # Initialize lists to store linear expressions for each term
        non_varying_terms = []
        varying_terms = []
        thermal_terms = []

        # non-varying generators
        for generator, row in non_var_generators.iterrows():
            for hour in range(self.periods):
                heat_rate = row['Heat_rate_MMBTU_per_MWh']
                fuel_cost = row['Cost_per_MMBtu']
                VarOM = row['Var_OM_cost_per_MWh']
                non_varying_terms.append((heat_rate * fuel_cost + VarOM) * self.gen[generator, hour])

        # varying generators
        for generator, row in var_generators.iterrows():
            for hour in range(self.periods):
                VarOM = row['Var_OM_cost_per_MWh']
                varying_terms.append(VarOM * self.gen[generator, hour])

        if self.start_shut:
            # thermal generators
            for generator, row in thermal_generators.iterrows():
                for hour in range(self.periods):
                    existing_cap = row['Existing_Cap_MW']
                    start_cost = row['Start_cost_per_MW']
                    thermal_terms.append(existing_cap * start_cost * self.start[generator, hour])

        # Add all terms using lpSum
        total_obj = lpSum(non_varying_terms) + lpSum(varying_terms) + lpSum(thermal_terms)
        model += total_obj

    def define_energy_constraints(self, model) ->  None:
        """
        Define energy constraints for the Unit Commitment problem.

        Parameters:
        - model: The optimization model being defined.

        Ensures that the total energy supplied equals the demand in all time periods.
        """

        # Define the energy demand constraint
        for hour in range(self.periods):
            sum_energies = lpSum(self.gen[generator, hour] for generator, _ in self.generators.iterrows())
            model += (sum_energies == self.demand[hour])

    def define_capacity_constraints(self, model, thermal_generators, non_thermal_non_var_generators, var_generators) ->  None:
        """
        Define capacity constraints for the Unit Commitment problem.

        Parameters:
        - model: The optimization model being defined.
        - thermal_generators (pd.DataFrame): DataFrame containing thermal generator data.
        - non_thermal_non_var_generators (pd.DataFrame): DataFrame containing non-thermal, non-variable generator data.
        - var_generators (pd.DataFrame): DataFrame containing variable generator data.

        Ensures that energy generation is within capacity bounds for each generator.
        """

        # thermal generators requiring commitment
        for hour in range(self.periods):
            for generator, row in thermal_generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                min_power = row['Min_power']

                model += lpSum([self.gen[generator, hour] - self.commit[generator, hour] * existing_cap * min_power]) >= 0
                model += lpSum([self.gen[generator, hour] - self.commit[generator, hour] * existing_cap]) <= 0

        # non-variable generation not requiring commitment
        for hour in range(self.periods):
            for generator, row in non_thermal_non_var_generators.iterrows():
                existing_cap = row['Existing_Cap_MW_x']
                model += lpSum([self.gen[generator, hour] - existing_cap]) <= 0

        # variable generation, accounting for hourly capacity factor
        for hour in range(self.periods):
            for generator, row in var_generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                name = str(row['region']) + '_' + str(row['Resource']) + '_1.0'
                variability = self.variability.loc[(self.variability['generator'] == name) & (self.variability['Hour'] == hour +1), 'Variability'].values[0]

                model += lpSum([self.gen[generator, hour] - existing_cap * variability]) <= 0

    def unit_commitment_constraints(self, model, thermal_generators) -> None:
        """
        Define unit commitment constraints for the Unit Commitment problem.

        Parameters:
        - model: The optimization model being defined.
        - thermal_generators (pd.DataFrame): DataFrame containing thermal generator data.

        Ensures minimum up and down times for thermal generators and commitment state consistency.
        """

        # minimum up and down time
        for generator, row in thermal_generators.iterrows():
            for hour in range(self.periods):
                if hour >= row['Up_time']:
                    model += lpSum(self.commit[generator, hour] - sum(self.start[generator, t] for t in range(hour - row['Up_time'], hour))) >= 0

                if hour >= row['Down_time']:
                    model += lpSum(1 - self.commit[generator, hour] - sum(self.shut[generator, t] for t in range(hour - row['Down_time'], hour))) <= 0

        # Commmitment state
        for hour in range(1, self.periods):
            for generator, row in thermal_generators.iterrows():
                model += lpSum(self.commit[generator, hour]  - self.commit[generator, hour - 1] - self.start[generator, hour] + self.shut[generator, hour]) == 0

    
    def sample_problem(self) -> dict:
        """
        Solve the Unit Commitment problem using Quantum Annealing.

        Returns:
        - dict: A dictionary containing the best quantum annealing solution.
        """

        # Create a MILP problem
        prob = LpProblem("Unit_self.commitment_Problem", LpMinimize)
        thermal, non_thermal, non_var, var, non_thermal_non_var = self.classify_dataset()

        self.define_variables(thermal)
        self.define_objective(prob, non_var, var, thermal)
        self.define_energy_constraints(prob)
        self.define_capacity_constraints(prob, thermal, non_thermal_non_var, var)

        if self.start_shut:
            self.unit_commitment_constraints(prob, thermal)

                # Solve the problem
        def solve_problem():
            prob.solve()

        # Measure the execution time
        execution_time = timeit.timeit(solve_problem, number=1)

        # Check the solution status
        if LpStatus[prob.status] == 'Optimal':
            # Retrieve the optimal solution
            commit_solution = {(generator, t): self.commit[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
            gen_solution = {(generator, t): self.gen[generator, t].varValue for generator, row in self.generators.iterrows() for t in range(self.periods)}
            solution = [commit_solution, gen_solution]
            if self.start_shut:
                start_solution = {(generator, t): self.start[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
                shut_solution = {(generator, t): self.shut[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
                solution += [start_solution] + [shut_solution]
            
            total_cost = prob.objective.value()

            return solution, total_cost, execution_time
        else:
            print("No feasible solution found.")

    
    def classical_implementation(self):
        """
        Solve the Unit Commitment problem using a classical MILP solver.

        Returns:
        - Tuple: A tuple containing the classical solution, total cost, and execution time.
        """

        thermal, non_thermal, non_var, var, non_thermal_non_var = self.classify_dataset()
        
        # Create a MILP problem
        prob = LpProblem("Unit_self.commitment_Problem", LpMinimize)

        # Decision variables
        commit = LpVariable.dicts("commit", [(generator, t) for generator, rows in thermal.iterrows() for t in range(self.periods)], cat="Binary")
        gen = LpVariable.dicts("gen", [(generator, t) for generator, rows in self.generators.iterrows() for t in range(self.periods)], cat="Integer")

        if self.start_shut:
            start = LpVariable.dicts("start", [(generator, t) for generator, rows in thermal.iterrows() for t in range(self.periods)], cat="Binary")
            shut = LpVariable.dicts("shut", [(generator, t) for generator, rows in thermal.iterrows() for t in range(self.periods)], cat="Binary")
        
        # Initialize lists to store linear expressions for each term
        non_varying_terms = []
        varying_terms = []
        thermal_terms = []

        # non-varying generators
        for generator, row in non_var.iterrows():
            for hour in range(self.periods):
                heat_rate = row['Heat_rate_MMBTU_per_MWh']
                fuel_cost = row['Cost_per_MMBtu']
                VarOM = row['Var_OM_cost_per_MWh']
                non_varying_terms.append((heat_rate * fuel_cost + VarOM) * gen[generator, hour])

        # varying generators
        for generator, row in var.iterrows():
            for hour in range(self.periods):
                VarOM = row['Var_OM_cost_per_MWh']
                varying_terms.append(VarOM * gen[generator, hour])

        if self.start_shut:
            # thermal generators
            for generator, row in thermal.iterrows():
                for hour in range(self.periods):
                    existing_cap = row['Existing_Cap_MW']
                    start_cost = row['Start_cost_per_MW']
                    thermal_terms.append(existing_cap * start_cost * start[generator, hour])

        # Add all terms using lpSum
        total_obj = lpSum(non_varying_terms) + lpSum(varying_terms) + lpSum(thermal_terms)
        prob += total_obj

        # Define the energy demand constraint
        for hour in range(self.periods):
            sum_energies = lpSum(gen[generator, hour] for generator, _ in self.generators.iterrows())
            prob += (sum_energies == self.demand[hour])

        # thermal generators requiring commitment
        for hour in range(self.periods):
            for generator, row in thermal.iterrows():
                existing_cap = row['Existing_Cap_MW']
                min_power = row['Min_power']

                prob += lpSum([gen[generator, hour] - commit[generator, hour] * existing_cap * min_power]) >= 0
                prob += lpSum([gen[generator, hour] - commit[generator, hour] * existing_cap]) <= 0

        # non-variable generation not requiring commitment
        for hour in range(self.periods):
            for generator, row in non_thermal_non_var.iterrows():
                existing_cap = row['Existing_Cap_MW_x']
                prob += lpSum([gen[generator, hour] - existing_cap]) <= 0

        # variable generation, accounting for hourly capacity factor
        for hour in range(self.periods):
            for generator, row in var.iterrows():
                existing_cap = row['Existing_Cap_MW']
                name = str(row['region']) + '_' + str(row['Resource']) + '_1.0'
                variability = self.variability.loc[(self.variability['generator'] == name) & (self.variability['Hour'] == hour +1), 'Variability'].values[0]

                prob += lpSum([gen[generator, hour] - existing_cap * variability]) <= 0

        if self.start_shut:
            # minimum up and down time
            for generator, row in thermal.iterrows():
                for hour in range(self.periods):
                    if hour >= row['Up_time']:
                        prob += lpSum(commit[generator, hour] - sum(start[generator, t] for t in range(hour - row['Up_time'], hour))) >= 0

                    if hour >= row['Down_time']:
                        prob += lpSum(1 - commit[generator, hour] - sum(shut[generator, t] for t in range(hour - row['Down_time'], hour))) <= 0

            # Commmitment state
            for hour in range(1, self.periods):
                for generator, row in thermal.iterrows():
                    prob += lpSum(commit[generator, hour]  - commit[generator, hour - 1] - start[generator, hour] + shut[generator, hour]) == 0
           
        # Solve the problem
        def solve_problem():
            prob.solve()

        # Measure the execution time
        execution_time = timeit.timeit(solve_problem, number=1)

        # Check the solution status
        if LpStatus[prob.status] == 'Optimal':
            # Retrieve the optimal solution
            commit_solution = {(generator, t): commit[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
            gen_solution = {(generator, t): gen[generator, t].varValue for generator, row in self.generators.iterrows() for t in range(self.periods)}
            solution = [commit_solution, gen_solution]
            if self.start_shut:
                start_solution = {(generator, t): start[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
                shut_solution = {(generator, t): shut[generator, t].varValue for generator, row in thermal.iterrows() for t in range(self.periods)}
                solution += [start_solution] + [shut_solution]
            
            total_cost = prob.objective.value()

            return solution, total_cost, execution_time
        else:
            print("No feasible solution found.")
        
    def return_csv(self) -> pd.DataFrame:
        """
        Get results from both quantum and classical implementations.

        Returns:
        - pd.DataFrame: A DataFrame containing the results of both implementations.
        """

        solution, cost, time = self.classical_implementation()

         # Prepare the data
        gens = []
        periods = []
        energies = []
        resource = []

        for key, value in gen_results.items():
            key_parts = key.split('_')
            _, generator, period = key_parts
            gens.append(int(generator))
            periods.append(int(period))
            energies.append(value)
            resource.append(self.generators['Resource'][int(generator)])

        # Create a DataFrame
        data_df = pd.DataFrame({
            'Generators': gens,
            'Resources' : resource,
            'Periods': periods,
            'Problem size': self.size,
            'Generated energy variable classical': solution[1],
            'Commitment variable classical': solution[0],
            'Start-up variable classical': solution[2],
            'Shut-down variable classical': solution[3],
            'Operational cost classical': cost,
            'CPU time': time
        })

        for key, value in commit_results.items():
            key_parts = key.split('_')
            _, generator, time = key_parts
            row_index = data_df[(data_df['Generators'] == int(generator)) & (data_df['Periods'] == int(time))].index[0]
            data_df.loc[row_index, 'Commitment annealer'] = value

        for key, value in start_results.items():
            key_parts = key.split('_')
            _, generator, time = key_parts
            row_index = data_df[(data_df['Generators'] == int(generator)) & (data_df['Periods'] == int(time))].index[0]
            data_df.loc[row_index, 'Start-up annealer'] = value

        for key, value in shut_results.items():
            key_parts = key.split('_')
            _, generator, time = key_parts
            row_index = data_df[(data_df['Generators'] == int(generator)) & (data_df['Periods'] == int(time))].index[0]
            data_df.loc[row_index, 'Shut-down annealer'] = value

        return data_df


