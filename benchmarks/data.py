import numpy as np
from glue.core import Data

from glue.core.subset import RoiSubsetState
from glue.core.roi import PolygonalROI


class TimeDataFRB:
    """
    Benchmarks related to data FRB calculations
    """

    def setup(self):
        self.data = Data(x=np.random.random((64, 64, 64)))
        self.bounds = [(10., 54., 128)] * 3

    def time_3d_frb(self):
        self.data.compute_fixed_resolution_buffer(bounds=self.bounds,
                                                  target_cid=self.data.id['x'])


class TimeDataStats:
    """
    Benchmarks related to compute_statistics
    """

    def setup(self):

        # Set up main 3-d dataset
        self.data = Data(values=np.random.random((512, 512, 512)))

        # Set up subset state which is a polygon defined in pixel space and
        # only covering a small fraction of the whole cube.
        self.polygonal_roi = PolygonalROI([440, 460, 450], [400, 455, 500])
        self.polygonal_subset_state = RoiSubsetState(xatt=self.data.pixel_component_ids[0],
                                                     yatt=self.data.pixel_component_ids[2],
                                                     roi=self.polygonal_roi)

    def time_compute_statistic_subset_state(self):
        # Check efficiency of compute_statistic when collapsing along dimensions
        # for which the ROI is defined
        self.data.compute_statistic('median', cid=self.data.id['values'],
                                    axis=(0, 2), subset_state=self.polygonal_subset_state)
