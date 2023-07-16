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
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, Integer
from neal import SimulatedAnnealingSampler
import numpy as np

from pathlib import Path
import sys
#path_root = Path(/Users/juanfrancisco/Desktop/uc/uc-problem-annealing/uc_annealing.py).parents[2]
sys.path.append('/Users/juanfrancisco/Desktop/uc/uc-problem-annealing')
sys.path.append('/Users/juanfrancisco/Desktop/uc/uc-problem-annealing/util')
from util.utils import get_index, get_generator_and_day
from util.plots import plot_schedule

class UC_Annealing:

    def __init__(self, generators, periods, C, E, D, plot = False) -> None:
        self.generators = generators
        self.periods = periods
        self.cost = C 
        self.efficiency = E 
        self.demand = D 
        self.size = generators * periods

        self.cqm = None
        self.x = None

        self.qpu_access_time = None
        self.sample = self.sample_problem()

        if plot:
            self.plot_results(self.sample)

    def get_objective(self):
        return self.cqm.objective

    def get_energy_constraints(self) -> str:
        return self.cqm.constraints['energy_demand'].to_polystring()
    
    def define_cqm(self) -> None:
        self.cqm = ConstrainedQuadraticModel()

    def define_variable(self) -> None:
        self.x = {(n, t): Binary('x{}_{}'.format(n, t)) for n in range(self.generators) for t in range(self.periods)}

    def define_objective(self) ->  None:
        objective = BinaryQuadraticModel('BINARY')
        for generator in range(self.generators):
            for hour in range(self.periods):
                index = get_index(generator, hour, self.periods)
                objective.update(self.cost[index] * self.x[generator, hour])
        self.cqm.set_objective(objective)

    def define_constraints(self) ->  None:
        sum =  BinaryQuadraticModel('BINARY')
        for generator in range(self.generators):
            for hour in range(self.periods):
                index = get_index(generator, hour, self.periods)
                sum += self.efficiency[index] * self.x[generator, hour]
        self.cqm.add_constraint(sum >= self.demand, label='energy_demand')

    def sample_problem(self) -> dict:

        self.define_cqm()
        self.define_variable()

        self.define_objective()
        self.define_constraints()

        sampler = LeapHybridCQMSampler()
        raw_sampleset = sampler.sample_cqm(self.cqm)
        feasible_sampleset = raw_sampleset.filter(lambda d: d.is_feasible)
        num_feasible = len(feasible_sampleset)

        self.qpu_access_time = raw_sampleset.info['qpu_access_time']

        if num_feasible > 0:
            best_samples = feasible_sampleset.truncate(min(10, num_feasible))
        else:
            best_samples = raw_sampleset.truncate(10)

        best_sample = best_samples.first.sample

        return best_sample

    def plot_results(self, sample) -> None:
        filtered_keys = [key for key, value in sample.items() if value == 1]
        completed_list = [item + '0' if len(item) == 2 else item for item in filtered_keys]
        quantum_solution = [(int(item.split('_')[0][1:]), int(item.split('_')[1])) for item in completed_list]
        plot_schedule(quantum_solution, self.periods, self.generators, save_image=False, image_path=None)


