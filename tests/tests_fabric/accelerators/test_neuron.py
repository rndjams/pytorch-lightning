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
import pytest
import torch

from lightning.fabric.accelerators.neuron import NeuronAccelerator


def _neuron_devtype_constructible() -> bool:
    try:
        torch.device("neuron", 0)
        return True
    except Exception:
        return False


requires_neuron_devtype = pytest.mark.skipif(
    not _neuron_devtype_constructible(), reason="requires the 'neuron' device type (torch-neuronx installed)"
)


def test_name():
    assert NeuronAccelerator.name() == "neuron"


def test_is_available_false_without_backend(monkeypatch):
    monkeypatch.setattr(torch._C, "_get_privateuse1_backend_name", lambda: "cpu")
    NeuronAccelerator.is_available.cache_clear()
    try:
        assert NeuronAccelerator.is_available() is False
        assert NeuronAccelerator.auto_device_count() == 0
    finally:
        NeuronAccelerator.is_available.cache_clear()


def test_is_available_and_device_count(monkeypatch):
    monkeypatch.setattr(torch._C, "_get_privateuse1_backend_name", lambda: "neuron")
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 4)
    NeuronAccelerator.is_available.cache_clear()
    try:
        assert NeuronAccelerator.is_available() is True
        assert NeuronAccelerator.auto_device_count() == 4
    finally:
        NeuronAccelerator.is_available.cache_clear()


@pytest.mark.parametrize(("devices", "expected"), [(2, 2), ("2", 2), ("0,1", [0, 1]), ([0, 2], [0, 2])])
def test_parse_devices(devices, expected):
    assert NeuronAccelerator.parse_devices(devices) == expected


def test_setup_device_with_wrong_device_type():
    with pytest.raises(ValueError, match="Neuron device"):
        NeuronAccelerator().setup_device(torch.device("cpu"))


@requires_neuron_devtype
@pytest.mark.parametrize(("devices", "indices"), [(1, [0]), (3, [0, 1, 2]), ([0, 2], [0, 2])])
def test_get_parallel_devices(devices, indices):
    assert NeuronAccelerator.get_parallel_devices(devices) == [torch.device("neuron", i) for i in indices]
