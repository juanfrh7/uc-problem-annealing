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

__version__ = '0.5.4'

from dwave.preprocessing import *
import dwave.preprocessing.composites
from dwave.preprocessing.composites import *
import dwave.preprocessing.lower_bounds
from dwave.preprocessing.lower_bounds import *

from dwave.preprocessing.presolve import *


def get_include() -> str:
    """Return the directory with dwave-preprocessing's header files."""
    import os.path
    return os.path.join(os.path.dirname(__file__), 'include')
