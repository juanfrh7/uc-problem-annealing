# Copyright 2022 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

cimport cython

from libcpp.vector cimport vector
from libcpp.utility cimport move as cppmove

import numpy as np

import dimod

from dimod.libcpp cimport ConstrainedQuadraticModel as cppConstrainedQuadraticModel
from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel, make_cqm
from dimod.cyutilities cimport ConstNumeric


cdef class cyPresolver:
    def __init__(self, cyConstrainedQuadraticModel cqm, *, bint move = False):
        self._original_variables = cqm.variables.copy()

        if move:
            self.cpppresolver = cppPresolver[bias_type, index_type, double](cppmove(cqm.cppcqm))

            # todo: replace with cqm.clear()
            cqm.variables._clear()
            cqm.constraint_labels._clear()
        else:
            self.cpppresolver = cppPresolver[bias_type, index_type, double](cqm.cppcqm)

        # we need this because we may detach the model later
        self._model_num_variables = self.cpppresolver.model().num_variables()

    def apply(self):
        """Apply any loaded presolve techniques to the held constrained quadratic model."""
        self.cpppresolver.apply()
        self._model_num_variables = self.cpppresolver.model().num_variables()

    def clear_model(self):
        """Clear the held constrained quadratic model. This is useful to save memory."""
        self.cpppresolver.detach_model()

    def copy_model(self):
        """Return a copy of the held constrained quadratic model."""
        cdef cppConstrainedQuadraticModel[bias_type, index_type] tmp = self.cpppresolver.model()  # copy
        return make_cqm(cppmove(tmp))  # then move

    def detach_model(self):
        """Create a :class:`dimod.ConstrainedQuadraticModel` from the held model.

        Subsequent attempts to access the held model raise a :exc:`RuntimeError`.
        """
        return make_cqm(cppmove(self.cpppresolver.detach_model()))

    def load_default_presolvers(self):
        """Load the default presolvers."""
        self.cpppresolver.load_default_presolvers()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _restore_samples(self, ConstNumeric[:, ::1] samples):
        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        cdef double[:, ::1] original_samples = np.empty((num_samples, self._original_variables.size()), dtype=np.double)

        cdef vector[double] original
        cdef vector[double] reduced
        for i in range(num_samples):
            reduced.clear()
            for vi in range(num_variables):
                reduced.push_back(samples[i, vi])

            original = self.cpppresolver.postsolver().apply(reduced)

            if original.size() != original_samples.shape[1]:
                raise RuntimeError("unexpected reduced variables size")

            for vi in range(original.size()):
                original_samples[i, vi] = original[vi]

        return original_samples

    def restore_samples(self, samples_like):
        """Restore the original variable labels to a set of reduced samples.

        Args:
            samples_like: A :class:`dimod.types.SamplesLike`. The samples must
                be index-labeled.

        Returns:
            Tuple:
                A 2-tuple where the first entry is the restored samples and the second
                is the original labels.

        """
        samples, labels = dimod.as_samples(samples_like, labels_type=dimod.variables.Variables)

        if not labels.is_range:
            raise ValueError("expected samples to be integer labelled")

        if samples.shape[1] != self._model_num_variables:
            raise ValueError(f"sample(s) must have {self._model_num_variables} variables, "
                             f"given sample(s) have {samples.shape[1]}")

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        restored = self._restore_samples(samples)

        return np.asarray(restored), self._original_variables
