from typing import NamedTuple, Union
from torch import Tensor


class RadiationFieldChannel(NamedTuple):
    spectrum: Tensor
    fluence: Tensor
    error: Tensor = None


class RadiationField(NamedTuple):
    scatter_field: RadiationFieldChannel
    xray_beam: RadiationFieldChannel


class DirectionalInput(NamedTuple):
    direction: Tensor
    spectrum: Tensor


class PositionalInput(DirectionalInput):
    direction: Tensor
    spectrum: Tensor
    position: Tensor


class TrainingInputData(NamedTuple):
    input: Union[DirectionalInput, PositionalInput]
    ground_truth: Union[RadiationField, RadiationFieldChannel]


class ChannelMetrics(NamedTuple):
    fluence_loss: Tensor
    spectrum_loss: Tensor
    fluence_accuracy: Tensor
    spectrum_accuracy: Tensor


class TrainingMetrics(NamedTuple):
    scatter_field: ChannelMetrics
    xray_beam: ChannelMetrics
