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
from typing import Any, Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.neuron import NeuronAccelerator as _FabricNeuronAccelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry
from lightning.fabric.utilities.types import _DEVICE
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.utilities.exceptions import MisconfigurationException


class NeuronAccelerator(Accelerator):
    """Accelerator for AWS Neuron (Trainium / Inferentia) via the native TorchNeuron backend.

    Targets the eager ``torch.accelerator`` / ``PrivateUse1`` ``"neuron"`` device from
    ``torch-neuronx`` (Neuron SDK 2.33+), reusing Lightning's standard eager strategies
    (single-device, DDP with the native ``"neuron"`` ``torch.distributed`` backend, FSDP).

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            MisconfigurationException:
                If the selected device is not a Neuron device.
        """
        if device.type != "neuron":
            raise MisconfigurationException(f"Device should be a Neuron device, got {device} instead.")
        try:
            torch.accelerator.set_device_index(device.index or 0)
        except Exception:
            pass

    @override
    def get_device_stats(self, device: _DEVICE) -> dict[str, Any]:
        """Neuron device stats are not exposed through ``torch.accelerator`` yet."""
        return {}

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Union[int, list[int]]:
        """Accelerator device parsing logic."""
        return _FabricNeuronAccelerator.parse_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        return _FabricNeuronAccelerator.get_parallel_devices(devices)

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the number of available NeuronCores."""
        return _FabricNeuronAccelerator.auto_device_count()

    @staticmethod
    @override
    def is_available() -> bool:
        return _FabricNeuronAccelerator.is_available()

    @staticmethod
    @override
    def name() -> str:
        return "neuron"

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register(
            cls.name(),
            cls,
            description=cls.__name__,
        )
