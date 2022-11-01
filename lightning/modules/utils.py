import torch
import numpy as np

from pytorch_lightning.utilities import LightningEnum


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class SpkEmbType(LightningEnum):
    """Enum to represent speaker embedding type
    """
    TABLE = "table"
    SHARED = "shared"
    ENCODER = "encoder"
    DVEC = "dvec"
    SCRATCH = "scratch_encoder"

    @staticmethod
    def supported_type(val: str) -> bool:
        return any(x.value == val for x in SpkEmbType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in SpkEmbType]


class FeatureLevel(LightningEnum):
    """Feature level of phoneme and energy
    """
    PHONEME = "phoneme_level"
    FRAME = "frame_level"

    @staticmethod
    def supported_level(val: str) -> bool:
        return any(x.value == val for x in FeatureLevel)

    @staticmethod
    def supported_levels() -> list[str]:
        return [x.value for x in FeatureLevel]


class QuantizationType(LightningEnum):
    """Quantization type for phoneme and energy
    """
    LINEAR = "linear"
    LOG = "log"

    @staticmethod
    def supported_type(val: str) -> bool:
        return any(x.value == val for x in QuantizationType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in QuantizationType]

