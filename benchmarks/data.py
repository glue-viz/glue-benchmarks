import numpy as np
from glue.core import Data


class TimeData:
    """
    Benchmarks related to data
    """

    def setup(self):
        self.data = Data(x=np.random.random((64, 64, 64)))
        self.bounds = [(10., 54., 128)] * 3

    def time_3d_frb(self):
        self.data.compute_fixed_resolution_buffer(bounds=self.bounds,
                                                  target_cid=self.data.id['x'])
