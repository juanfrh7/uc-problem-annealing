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
import neal
import os
import pandas as pd
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
from dimod import QuadraticModel, ConstrainedQuadraticModel, Binary, Integer
from neal import SimulatedAnnealingSampler
import numpy as np

from pulp import *
import timeit

from pathlib import Path
import sys
#path_root = Path(/Users/juanfrancisco/Desktop/uc/uc-problem-annealing/uc_annealing.py).parents[2]
sys.path.append('/Users/juanfrancisco/Desktop/uc/uc-problem-annealing')
sys.path.append('/Users/juanfrancisco/Desktop/uc/uc-problem-annealing/util')

class UC_Annealing:

    def __init__(self, demand, fuel_data, generators) -> None:

        self.demand = demand
        self.fuel_data = fuel_data
        self.generators = generators

        self.num_generators = len(generators)
        self.periods = len(demand)

        self.size = self.num_generators * self.periods

        self.cqm = None
        self.x = None
        self.y = None

    def get_objective(self):
        return self.cqm.objective

    def get_energy_constraints(self) -> str:
        return self.cqm.constraints['energy_demand'].to_polystring()
    
    def define_cqm(self) -> None:
        self.cqm = ConstrainedQuadraticModel()

    def define_variable(self) -> None:
        #define the commitment variable
        self.x = {(n, t): Binary('x_{}_{}'.format(n, t)) for n, rows in self.generators.iterrows() for t in range(self.periods)}

        #define the generation variable 
        self.y = {(n, t): Integer('y_{}_{}'.format(n, t)) for n, rows in self.generators.iterrows() for t in range(self.periods)}

    def define_objective(self) ->  None:
        objective = QuadraticModel()
        for generator, row in self.generators.iterrows():
            for hour in range(self.periods):
                heat_rate = self.generators['Heat_rate_MMBTU_per_MWh'][generator]
                fuel = self.generators['Fuel'][generator]
                fuel_cost = self.fuel_data[self.fuel_data['Fuel'] == fuel]['Cost_per_MMBtu'].values[0]
                VarOM = self.generators['Var_OM_cost_per_MWh'][generator]

                objective.update((heat_rate * fuel_cost + VarOM) * self.y[generator, hour])
                
        self.cqm.set_objective(objective)

    def define_energy_constraint(self) ->  None:
        for hour in range(self.periods):
            sum_energies = QuadraticModel()
            for generator, row in self.generators.iterrows():
                sum_energies += self.y[generator, hour]

            self.cqm.add_constraint(sum_energies == self.demand[hour], label = f'energy demand hour {hour}')

    def define_energy_bounds(self) ->  None:
        for hour in range(self.periods):
            for generator, row in self.generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                min_power = row['Min_power']

                
                self.cqm.add_constraint(self.x[generator, hour] * existing_cap * min_power - self.y[generator, hour] <= 0,
                                label = f'energy lower bound generator {generator} at {hour}')
                
                self.cqm.add_constraint(self.x[generator, hour] * existing_cap - self.y[generator, hour] >= 0,
                                label = f'energy upper bound generator {generator} at {hour}')

    def sample_problem(self) -> dict:

        self.define_cqm()
        self.define_variable()

        self.define_objective()
        self.define_energy_constraint()
        self.define_energy_bounds()

        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(self.cqm)
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(feasible_sampleset)

        self.qpu_access_time = raw_sampleset.info['qpu_access_time']

        if num_feasible > 0:
            best_samples = feasible_sampleset.truncate(min(10, num_feasible))
        else:
            best_samples = raw_sampleset.truncate(10)

        return best_samples
    
    def classical_implementation(self):
        # Create a MILP problem
        prob = LpProblem("Unit_Commitment_Problem", LpMinimize)

        # Decision variables
        x = LpVariable.dicts("x", [(generator, t) for generator, rows in self.generators.iterrows() for t in range(self.periods)], cat="Binary")
        y = LpVariable.dicts("y", [(generator, t) for generator, rows in self.generators.iterrows() for t in range(self.periods)], cat="Integer")
        
        
        prob += lpSum([(row['Heat_rate_MMBTU_per_MWh'] * 
                self.fuel_data[self.fuel_data['Fuel'] == self.generators['Fuel'][generator]]['Cost_per_MMBtu'].values[0]
                + row['Var_OM_cost_per_MWh']) * y[generator, t]
                for generator, row in self.generators.iterrows() for t in range(self.periods)])
        
        #define the energy demand contraint
        for hour in range(self.periods):
            prob += lpSum([y[generator, hour] for generator, row in self.generators.iterrows()]) == self.demand[hour]

        #define the energy demand contraint
        for hour in range(self.periods):
            for generator, row in self.generators.iterrows():
                existing_cap = row['Existing_Cap_MW']
                min_power = row['Min_power']

                prob += lpSum([x[generator, hour] * existing_cap * min_power - y[generator, hour]]) <= 0
                prob += lpSum([x[generator, hour] * existing_cap - y[generator, hour]]) >= 0
                
        # Solve the problem
        def solve_problem():
            prob.solve()

        # Measure the execution time
        execution_time = timeit.timeit(solve_problem, number=1)

        # Check the solution status
        if LpStatus[prob.status] == 'Optimal':
            # Retrieve the optimal solution
                # Retrieve the optimal solution
            x_result = {(generator, t): value(x[generator, t]) for generator, row in self.generators.iterrows() for t in range(self.periods)}
            y_result = {(generator, t): value(y[generator, t]) for generator, row in self.generators.iterrows() for t in range(self.periods)}
            total_cost = value(prob.objective)

            solution = [x_result, y_result]

            total_cost = value(prob.objective)
            return solution, total_cost, execution_time
        else:
            print("No feasible solution found.")
        
    def get_results(self) -> pd.DataFrame:

        quantum_sample = self.sample_problem()
        classical_solution, classical_cost, classical_time = self.classical_implementation()

        ## 1. Unpack quantum results

        best_sample = quantum_sample.first.sample
        # Separate x and y variables into two dictionaries
        x_results_quantum = {}
        y_results_quantum = {}

        for key, value in best_sample.items():
            if key.startswith('x'):
                x_results_quantum[key] = value
            elif key.startswith('y'):
                y_results_quantum[key] = value

        # Prepare the data
        generators = []
        times = []
        resource = []
        energy_annealer = []
        status_annealer = []
        quantum_cost = quantum_sample.first.energy

        for key, value in y_results_quantum.items():
            key_parts = key.split('_')
            _, generator, time = key_parts
            generators.append(int(generator))
            times.append(int(time))
            energy_annealer.append(value)
            resource.append(self.generators['Resource'][int(generator)])

        for key, value in x_results_quantum.items():
            status_annealer.append(value)

        ## 2. Create dataframe

        # Create a DataFrame
        results = pd.DataFrame({
            'Generator': generators,
            'Resource' : resource,
            'Time': times,
            'Size': self.size,
            'Variables': len(quantum_sample.variables),
            'Generated Energy annealer': energy_annealer,
            'Status annealer': status_annealer,
            'Cost annealer': quantum_cost,
            'QPU access time [s]': quantum_sample.info['qpu_access_time']/1000000,
            'QPU charge time [s]': quantum_sample.info['charge_time']/1000000,
            'QPU run time [s]': quantum_sample.info['run_time']/1000000

        })

        ## 3. Unpack classical results

        results['Generated Energy classical'] = [classical_solution[1].get((row['Generator'], row['Time']), 0.0) for index, row in results.iterrows()]
        results['Generator status classical'] = [classical_solution[0].get((row['Generator'], row['Time']), 0.0) for index, row in results.iterrows()]
        results['Classical cost'] = classical_cost
        results['Classical execution time [s]'] = classical_time


        return results


