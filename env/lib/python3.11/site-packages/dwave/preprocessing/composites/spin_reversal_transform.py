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

from random import random

from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.sampleset import SampleSet, concatenate
from dimod.vartypes import Vartype

__all__ = ['SpinReversalTransformComposite']

class SpinReversalTransformComposite(Sampler, Composite):
    """Composite for applying spin reversal transform preprocessing.

    Spin reversal transforms (or "gauge transformations") are applied
    by flipping the spin of variables in the Ising problem. After
    sampling the transformed Ising problem, the same bits are flipped in the
    resulting sample [#km]_.

    Args:
        sampler: A `dimod` sampler object.

    Examples:
        This example composes a dimod ExactSolver sampler with spin transforms then
        uses it to sample an Ising problem.

        >>> from dimod import ExactSolver
        >>> from dwave.preprocessing.composites import SpinReversalTransformComposite
        >>> base_sampler = ExactSolver()
        >>> composed_sampler = SpinReversalTransformComposite(base_sampler)
        ... # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.first.sample
        {'a': -1, 'b': -1}

    References
    ----------
    .. [#km] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """
    children = None
    parameters = None
    properties = None

    def __init__(self, child):
        self.children = [child]

        self.parameters = parameters = {'spin_reversal_variables': []}
        parameters.update(child.parameters)

        self.properties = {'child_properties': child.properties}

    def sample(self, bqm, *, num_spin_reversal_transforms=1, **kwargs):
        """Sample from the binary quadratic model.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            num_spin_reversal_transforms (integer, optional, default=1):
                Number of spin reversal transform runs.

        Returns:
            :class:`.SampleSet`
        
        Examples:
            This example runs 100 spin reversals applied to one variable of a QUBO problem.

            >>> from dimod import ExactSolver
            >>> from dwave.preprocessing.composites import SpinReversalTransformComposite
            >>> base_sampler = ExactSolver()
            >>> composed_sampler = SpinReversalTransformComposite(base_sampler)
            ...
            >>> Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
            >>> response = composed_sampler.sample_qubo(Q,
            ...               num_spin_reversal_transforms=100)
            >>> len(response)
            400
        """
        if not bqm.num_variables:
            return self.child.sample(bqm, **kwargs)

        # make a main response
        responses = []

        flipped_bqm = bqm.copy()
        transform = {v: False for v in bqm.variables}
        
        for ii in range(num_spin_reversal_transforms):
            # flip each variable with a 50% chance
            for v in bqm.variables:
                if random() > .5:
                    transform[v] = not transform[v]
                    flipped_bqm.flip_variable(v)

            flipped_response = self.child.sample(flipped_bqm, **kwargs)

            tf_idxs = [flipped_response.variables.index(v)
                       for v, flip in transform.items() if flip]

            if bqm.vartype is Vartype.SPIN:
                flipped_response.record.sample[:, tf_idxs] = -1 * flipped_response.record.sample[:, tf_idxs]
            else:
                flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

            if num_spin_reversal_transforms == 1:
                #Case of a single sampleset call, avoid
                #stripping of info field and other
                #information. 
                return(flipped_response)
            else:
                responses.append(flipped_response)

        return concatenate(responses)
