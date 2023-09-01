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

class UC_Annealing:

    def __init__(self, num_periods: int, generators: pd.DataFrame, demand: pd.DataFrame, variability: pd.DataFrame, start_shut: bool = False):

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

        # Classify generators into different types
        thermal_generators = self.generators[self.generators['Up_time']>0]
        non_thermal_generators = self.generators[self.generators['Up_time']==0]
        non_var_generators = self.generators[self.generators['IsVariable'] == False]
        var_generators = self.generators[self.generators['IsVariable'] == True]
        non_thermal_non_var_generators = non_thermal_generators.merge(non_var_generators, on='Resource')

        return thermal_generators, non_thermal_generators, non_var_generators, var_generators, non_thermal_non_var_generators

    def define_variables(self, thermal_generators) -> None:
        # commitment variable
        self.commit = {(n, t): Binary('commit_{}_{}'.format(n, t)) for n, rows in thermal_generators.iterrows() for t in range(self.periods)}

        # generation variable 
        self.gen = {(n, t): Integer('gen_{}_{}'.format(n, t)) for n, rows in self.generators.iterrows() for t in range(self.periods)}

        if self.start_shut == True:

            # start up variable 
            self.start = {(n, t): Binary('start_{}_{}'.format(n, t)) for n, rows in thermal_generators.iterrows() for t in range(self.periods)}

            # shut down variable 
            self.shut = {(n, t): Binary('shut_{}_{}'.format(n, t)) for n, rows in thermal_generators.iterrows() for t in range(self.periods)}

    def define_objective(self, model, non_var_generators, var_generators, thermal_generators) ->  None:
        operating_cost = QuadraticModel()

        # cost for non_varying generators
        for generator, row in non_var_generators.iterrows():
            for hour in range(self.periods):
                heat_rate = row['Heat_rate_MMBTU_per_MWh']
                fuel_cost = row['Cost_per_MMBtu']
                VarOM = row['Var_OM_cost_per_MWh']
                operating_cost.update((heat_rate * fuel_cost + VarOM) * self.gen[generator, hour])

        # cost for varying generators
        for generator, row in var_generators.iterrows():
            for hour in range(self.periods):
                VarOM = row['Var_OM_cost_per_MWh']
                operating_cost.update(VarOM * self.gen[generator, hour])

        if self.start_shut == True:

            # startup cost for thermal generators
            for generator, row in thermal_generators.iterrows():
                for hour in range(self.periods):
                    existing_cap = row['Existing_Cap_MW']
                    start_cost = row['Start_cost_per_MW']
                    operating_cost.update(existing_cap * start_cost * self.start[generator, hour])

        model.set_objective(operating_cost)

    def define_energy_constraints(self, model) ->  None:
        # Supply must = demand in all time periods
        for hour in range(self.periods):
            sum_energies = QuadraticModel()
            for generator, row in self.generators.iterrows():
                sum_energies += self.gen[generator, hour]
            model.add_constraint(sum_energies == self.demand[hour], label = f'energy demand hour {hour}')

    def define_capacity_constraints(self, model, thermal_generators, non_thermal_non_var_generators, var_generators) ->  None:
        # energy bounds for thermal generators
        for hour in range(self.periods):
            for generator, row in thermal_generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                min_power = row['Min_power']
                
                model.add_constraint(self.gen[generator, hour] - self.commit[generator, hour] * existing_cap * min_power  >= 0,
                                label = f'energy lower bound thermal generator {generator} at {hour}')
                
                model.add_constraint(self.gen[generator, hour] - self.commit[generator, hour] * existing_cap <= 0,
                                label = f'energy upper bound thermal generator {generator} at {hour}')
                
        # energy bounds for non-variable generation not requiring commitment
        for hour in range(self.periods):
            for generator, row in non_thermal_non_var_generators.iterrows():
                existing_cap = row['Existing_Cap_MW_x']

                model.add_constraint(self.gen[generator, hour] - existing_cap <= 0,
                label = f'energy lower bound non variable generator {generator} at {hour}')

        # energy bounds for variable generation, accounting for hourly capacity factor
        for hour in range(self.periods):
            for generator, row in var_generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                name = str(row['region']) + '_' + str(row['Resource']) + '_1.0'
                variability = self.variability.loc[(self.variability['generator'] == name) & 
                (self.variability['Hour'] == hour +1), 'Variability'].values[0]

                model.add_constraint(self.gen[generator, hour] - existing_cap * variability <= 0,
                label = f'energy lower bound variable generator {generator} at {hour}')

    def unit_commitment_constraints(self, model, thermal_generators) -> None:
        # minimum up and down time
        for generator, row in thermal_generators.iterrows():
            for hour in range(self.periods):
                
                if hour >= row['Up_time']:
                    model.add_constraint(self.commit[generator, hour] - sum(self.start[generator, t] for t in range(hour - row['Up_time'], hour)) >= 0,
                                    label = f'start time of generator {generator} at {hour}')
                if hour >= row['Down_time']:
                    model.add_constraint(1 - self.commit[generator, hour] - sum(self.shut[generator, t] for t in range(hour - row['Down_time'], hour)) >= 0,
                                    label = f'shut time of generator {generator} at {hour}')
                    
        # Commmitment state
        for hour in range(1, self.periods):
            for generator, row in thermal_generators.iterrows():
                model.add_constraint(self.commit[generator, hour]  - self.commit[generator, hour - 1] - self.start[generator, hour] + self.shut[generator, hour] == 0,
                                label = f'commitment state generator {generator} at {hour}')
    
    def sample_problem(self) -> dict:

        model = ConstrainedQuadraticModel()
        thermal, non_thermal, non_var, var, non_thermal_non_var = self.classify_dataset()

        self.define_variables(thermal)
        self.define_objective(model, non_var, var, thermal)
        self.define_energy_constraints(model)
        self.define_capacity_constraints(model, thermal, non_thermal_non_var, var)

        if self.start_shut:
            self.unit_commitment_constraints(model, thermal)

        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(model)
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(feasible_sampleset)

        self.qpu_access_time = raw_sampleset.info['qpu_access_time']

        if num_feasible > 0:
            best_samples = feasible_sampleset.truncate(min(10, num_feasible))
        else:
            best_samples = raw_sampleset.truncate(10)

        return best_samples
    
    def classical_implementation(self):

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
        
    def get_results(self) -> pd.DataFrame:

        quantum_sample = self.sample_problem()
        classical_solution, classical_cost, classical_time = self.classical_implementation()

        ## 1. Unpack quantum results

        best_sample = quantum_sample.first.sample
        
        ## 2. Separate variables into dictionaries
        commit_results = {}
        gen_results = {}
        start_results = {}
        shut_results = {}

        for key, value in best_sample.items():
            if key.startswith('commit'):
                commit_results[key] = value
            elif key.startswith('gen'):
                gen_results[key] = value

            if self.start_shut:
                if key.startswith('start'):
                    start_results[key] = value
                elif key.startswith('shut'):
                    shut_results[key] = value

        # Prepare the data
        gens = []
        periods = []
        energies = []
        resource = []
        cost = quantum_sample.first.energy

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
            'Number of variables': len(quantum_sample.variables),
            'Generated energy annealer': energies,
            'Commitment annealer': None,
            'Start-up annealer': None,
            'Shut-down annealer': None,
            'Operational cost annealer': cost,
            'QPU access time': quantum_sample.info['qpu_access_time']/1000000,
            'QPU charge time': quantum_sample.info['charge_time'],
            'QPU run time': quantum_sample.info['run_time']
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


        ## 3. Unpack classical results
        data_df['Generated energy classical'] = [classical_solution[1].get((row['Generators'], row['Periods']), None) for index, row in data_df.iterrows()]
        data_df['Commitment classical'] = [classical_solution[0].get((row['Generators'], row['Periods']), None) for index, row in data_df.iterrows()]

        if self.start_shut:
            data_df['Start-up classical'] = [classical_solution[2].get((row['Generators'], row['Periods']), None) for index, row in data_df.iterrows()]
            data_df['Shut classical'] = [classical_solution[3].get((row['Generators'], row['Periods']), None) for index, row in data_df.iterrows()]
        data_df['Operational cost classical'] = classical_cost
        data_df['CPU time'] = classical_time

        return data_df


