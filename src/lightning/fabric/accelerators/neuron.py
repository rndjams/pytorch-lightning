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
from functools import lru_cache
from typing import Union

import torch
from typing_extensions import override

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.registry import _AcceleratorRegistry


class NeuronAccelerator(Accelerator):
    """Accelerator for AWS Neuron (Trainium / Inferentia) via the native TorchNeuron backend.

    This targets the eager ``torch.accelerator`` / ``PrivateUse1`` ``"neuron"`` device exposed by
    ``torch-neuronx`` (Neuron SDK 2.33+), as opposed to the lazy ``torch_xla`` path handled by
    :class:`~lightning.fabric.accelerators.xla.XLAAccelerator`. It reuses the standard eager
    strategies (single-device, DDP, FSDP) and the native ``"neuron"`` ``torch.distributed`` backend.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    @override
    def setup_device(self, device: torch.device) -> None:
        """
        Raises:
            ValueError:
                If the selected device is not a Neuron device.
        """
        if device.type != "neuron":
            raise ValueError(f"Device should be a Neuron device, got {device} instead.")
        # Bind this process to the requested NeuronCore without eagerly creating a device context.
        try:
            torch.accelerator.set_device_index(device.index or 0)
        except Exception:
            pass

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, list[int]]) -> Union[int, list[int]]:
        """Accelerator device parsing logic."""
        if isinstance(devices, list):
            return devices
        if isinstance(devices, str):
            if devices == "-1":
                return list(range(NeuronAccelerator.auto_device_count()))
            if "," in devices:
                return [int(x) for x in devices.split(",") if x != ""]
            return int(devices)
        return devices

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, list[int]]) -> list[torch.device]:
        """Gets parallel devices for the Accelerator."""
        if isinstance(devices, int):
            devices = list(range(devices))
        return [torch.device("neuron", i) for i in devices]

    @staticmethod
    @override
    def auto_device_count() -> int:
        """Get the number of available NeuronCores.

        Uses ``torch.accelerator.device_count()``, which does not initialize the runtime and so is
        safe to call in the launcher parent process before ``spawn``/``torchrun`` fan-out.
        """
        if not NeuronAccelerator.is_available():
            return 0
        try:
            return torch.accelerator.device_count()
        except Exception:
            return 0

    @staticmethod
    @override
    @lru_cache(1)
    def is_available() -> bool:
        """Neuron is available when ``torch-neuronx`` has registered the ``"neuron"`` PrivateUse1 backend."""
        try:
            return torch._C._get_privateuse1_backend_name() == "neuron"
        except Exception:
            return False

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
