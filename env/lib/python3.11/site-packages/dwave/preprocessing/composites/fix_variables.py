# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warnings

from dimod.binary.binary_quadratic_model import as_bqm
from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet, append_variables

from dwave.preprocessing.lower_bounds import roof_duality

__all__ = ['FixVariablesComposite']

class FixVariablesComposite(ComposedSampler):
    """Composite to fix variables of a problem to provided.

    Fixes variables of a binary quadratic model (BQM) and modifies linear and
    quadratic terms accordingly. Returned samples include the fixed variable.

    Args:
        child_sampler (:class:`dimod.Sampler`):
            A dimod sampler
        
        algorithm (str, optional, default='explicit'):
            Determines how ``fixed_variables`` are found. 

            'explicit': ``fixed_variables`` should be passed in a call to 
            `.sample()`. If not, no fixing occurs and the problem is directly 
            passed to the child sampler.

            'roof_duality': Roof duality algorithm is used to find ``fixed_variables``. 
            ``strict`` may be passed in a call to `.sample()` to determine what
            variables the algorithm will fix. For details, see 
            :func:`~dwave.preprocessing.lower_bounds.roof_duality`.

    Examples:
       This example uses the :class:`.FixVariablesComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases accordingly.

       >>> from dimod import ExactSolver
       >>> from dwave.preprocessing.composites import FixVariablesComposite
       >>> h = {1: -1.3, 4: -0.5}
       >>> J = {(1, 4): -0.6}
       >>> sampler = FixVariablesComposite(ExactSolver())
       >>> sampleset = sampler.sample_ising(h, J, fixed_variables={1: -1})

       This next example involves the same problem but calculates ``fixed_variables``
       using the 'roof_duality' ``algorithm``.

       >>> sampler = FixVariablesComposite(ExactSolver(), algorithm='roof_duality')
       >>> sampleset = sampler.sample_ising(h, J, strict=False)

    """

    def __init__(self, child_sampler, *, algorithm='explicit'):
        self._children = [child_sampler]
        self.algorithm = algorithm

        self._parameters = self.child.parameters.copy()

        if self.algorithm == 'explicit':
            self._parameters['fixed_variables'] = []
        elif self.algorithm == 'roof_duality':
            self._parameters['strict'] = []
        else:
            raise ValueError("Unknown algorithm: {}".format(algorithm))

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self._parameters

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:class:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            fixed_variables (dict, optional, default=None):
                A dictionary of variable assignments used when ``self.algorithm`` 
                is 'explicit'.

            strict (bool, optional, default=True):
                Only used if ``self.algorithm`` is 'roof_duality'. If True, only 
                fixes variables for which assignments are true for all minimizing 
                points (strong persistency). If False, also fixes variables for 
                which the assignments are true for some but not all minimizing 
                points (weak persistency).

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.SampleSet`

        """

        if self.algorithm == 'explicit':
            fixed_variables = parameters.pop('fixed_variables', None)
            if fixed_variables is None:
                msg = ("No fixed_variables passed in when algorithm is 'explicit'. "
                       "Passing problem to child sampler without fixing.")
                warnings.warn(msg)
                return self.child.sample(bqm, **parameters)
        elif self.algorithm == 'roof_duality':
            _, fixed_variables = roof_duality(bqm, strict=parameters.pop('strict', True))

        # make sure that we're shapeable and that we have a BQM we can mutate
        bqm_copy = as_bqm(bqm, copy=True)

        bqm_copy.fix_variables(fixed_variables)

        sampleset = self.child.sample(bqm_copy, **parameters)

        def _hook(sampleset):
            # make RoofDualityComposite non-blocking

            if sampleset.variables:
                if len(sampleset):
                    return append_variables(sampleset, fixed_variables)
                else:
                    return sampleset.from_samples_bqm((np.empty((0, len(bqm))),
                                                       bqm.variables), bqm=bqm)

            # there are only fixed variables, make sure that the correct number
            # of samples are returned
            samples = [fixed_variables]*max(len(sampleset), 1)

            return sampleset.from_samples_bqm(samples, bqm=bqm)

        return SampleSet.from_future(sampleset, _hook)
