"""The grain-id stack shared by all layers of the extractor."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Field:
    """Integer grain ids as ids[k, y, x]; k is a z cell layer (3D field) or a frame (slice stack)."""
    ids: np.ndarray                             # int32 [nk, ny, nx]
    cell: float                                 # in-plane cell size (um)
    dz: float                                   # spacing between consecutive k (um)
    periodic: tuple = (True, True)              # (x, y)
    source: str = ''
    theta: dict = field(default_factory=dict)   # grain id -> (theta_x, theta_z) radians, graph_datastruct convention
    z0: float = 0.0                             # height of layer 0 (um)

    @property
    def nk(self):
        """Number of layers (or frames)."""
        return self.ids.shape[0]

    def height(self, k):
        """Height of layer k in um."""
        return self.z0 + k * self.dz
