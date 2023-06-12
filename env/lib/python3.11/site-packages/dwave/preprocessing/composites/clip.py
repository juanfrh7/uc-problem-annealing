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

from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet

__all__ = ['ClipComposite']

class ClipComposite(ComposedSampler):
    """Composite to clip variables of a problem.

    Clips the variables of a binary quadratic model (BQM) and modifies linear
    and quadratic terms accordingly.

    Args:
       sampler (:class:`dimod.Sampler`):
            A dimod sampler.

    Examples:
       This example uses :class:`.ClipComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler clips linear and quadratic biases as
       indicated by options.

       >>> from dimod import ExactSolver
       >>> from dwave.preprocessing.composites import ClipComposite
       >>> h = {'a': -4.0, 'b': -4.0}
       >>> J = {('a', 'b'): 3.2}
       >>> sampler = ClipComposite(ExactSolver())
       >>> response = sampler.sample_ising(h, J, lower_bound=-2.0, upper_bound=2.0)

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'lower_bound': [], 'upper_bound': []})
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, *, lower_bound=None, upper_bound=None, **parameters):
        """Clip and sample from the provided binary quadratic model.

        If lower_bound and upper_bound are given variables with value above or below are clipped.

        Args:
            bqm (:class:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            lower_bound (number):
                Value by which to clip the variables from below.

            upper_bound (number):
                Value by which to clip the variables from above.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.SampleSet`

        """
        child = self.child
        bqm_copy = _clip_bqm(bqm, lower_bound, upper_bound)
        response = child.sample(bqm_copy, **parameters)

        return SampleSet.from_samples_bqm(response, bqm, info=response.info)


def _clip_bqm(bqm, lower_bound, upper_bound):
    """Helper function for clipping a bqm."""

    bqm_copy = bqm.copy()
    if lower_bound is not None:
        linear = bqm_copy.linear
        for k, v in linear.items():
            if v < lower_bound:
                linear[k] = lower_bound
        quadratic = bqm_copy.quadratic
        for k, v in quadratic.items():
            if v < lower_bound:
                quadratic[k] = lower_bound

    if upper_bound is not None:
        linear = bqm_copy.linear
        for k, v in linear.items():
            if v > upper_bound:
                linear[k] = upper_bound
        quadratic = bqm_copy.quadratic
        for k, v in quadratic.items():
            if v > upper_bound:
                quadratic[k] = upper_bound
    return bqm_copy
