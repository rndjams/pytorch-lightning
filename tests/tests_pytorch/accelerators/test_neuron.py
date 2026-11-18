# Copyright The Lightning AI team.
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
from unittest.mock import Mock

import pytest
import torch

from lightning.fabric.accelerators.neuron import NeuronAccelerator as _FabricNeuronAccelerator
from lightning.pytorch.accelerators.neuron import NeuronAccelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


def test_name():
    assert NeuronAccelerator.name() == "neuron"


def test_get_device_stats_empty():
    assert NeuronAccelerator().get_device_stats(Mock()) == {}


def test_setup_device_with_wrong_device_type():
    with pytest.raises(MisconfigurationException, match="Neuron device"):
        NeuronAccelerator().setup_device(torch.device("cpu"))


def test_is_available_and_device_count(monkeypatch):
    monkeypatch.setattr(torch._C, "_get_privateuse1_backend_name", lambda: "neuron")
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 8)
    _FabricNeuronAccelerator.is_available.cache_clear()
    try:
        assert NeuronAccelerator.is_available() is True
        assert NeuronAccelerator.auto_device_count() == 8
    finally:
        _FabricNeuronAccelerator.is_available.cache_clear()


@pytest.mark.parametrize(("devices", "expected"), [(2, 2), ("0,1", [0, 1]), ([0, 1], [0, 1])])
def test_parse_devices(devices, expected):
    assert NeuronAccelerator.parse_devices(devices) == expected
