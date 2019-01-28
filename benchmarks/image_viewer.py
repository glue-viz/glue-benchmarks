import os
import numpy as np

from glue.core import Session, Data
from glue.viewers.matplotlib.viewer import MatplotlibViewerMixin
from glue.viewers.image.viewer import MatplotlibImageMixin
from glue.viewers.common.viewer import Viewer
from glue.viewers.matplotlib.mpl_axes import init_mpl
from glue.viewers.image.state import ImageViewerState
from glue.utils import defer_draw, decorate_all_methods
from glue.core.coordinates import coordinates_from_wcs
from glue.plugins.coordinate_helpers.link_helpers import Galactic_to_FK5
from glue.core.roi import CircularROI
from glue.core.subset import RoiSubsetState

from astropy.wcs import WCS
from astropy.io import fits

ROOT = os.path.dirname(__file__)
EQUATORIAL_WCS = WCS(fits.Header.fromtextfile(os.path.join(ROOT, 'equatorial.hdr')))
GALACTIC_WCS = WCS(fits.Header.fromtextfile(os.path.join(ROOT, 'galactic.hdr')))


@decorate_all_methods(defer_draw)
class HeadlessImageViewer(MatplotlibImageMixin, MatplotlibViewerMixin, Viewer):

    LABEL = '2D Image'
    _state_cls = ImageViewerState

    def __init__(self, session, parent=None, wcs=None, state=None):
        super(HeadlessImageViewer, self).__init__(session, state=state)
        self.figure, self.axes = init_mpl(wcs=True)
        MatplotlibViewerMixin.setup_callbacks(self)
        MatplotlibImageMixin.setup_callbacks(self)


def get_data_collection_and_viewer():

    session = Session()
    data_collection = session.data_collection

    viewer = HeadlessImageViewer(session=session)

    return data_collection, viewer


def load_wcs_data(data_collection):

    data_equ = Data(coords=coordinates_from_wcs(EQUATORIAL_WCS))
    data_equ['primary'] = np.random.random((512, 512))
    data_collection.append(data_equ)

    data_gal = Data(coords=coordinates_from_wcs(GALACTIC_WCS))
    data_gal['primary'] = np.random.random((512, 512))
    data_collection.append(data_gal)

    wy1, wx1 = data_gal.world_component_ids
    wy2, wx2 = data_equ.world_component_ids

    link = Galactic_to_FK5(wx1, wy1, wx2, wy2)

    data_collection.add_link(link)


SHAPES = {'small_2d': (16, 16),
          'medium_2d': (1024, 1024),
          'large_2d': (16384, 16384),
          'small_3d': (16, 16, 16),
          'medium_3d': (256, 256, 256)}


class TestBasic:

    params = ['small_2d', 'medium_2d', 'large_2d', 'small_3d', 'medium_3d']
    param_names = ['shape']

    def setup(self, shape):
        self.data_collection, self.viewer = get_data_collection_and_viewer()
        self.data = Data(cube=np.random.random(SHAPES[shape]))
        self.data_collection.append(self.data)

    def time_add_data(self, shape):
        self.viewer.add_data(self.data)


class TestSlicing:

    def setup(self):
        self.data_collection, self.viewer = get_data_collection_and_viewer()
        self.data = Data(cube=np.random.random((256, 256, 256)))
        self.data_collection.append(self.data)
        self.viewer.add_data(self.data)

    def time_change_slice(self):
        for i in range(0, 256, 64):
            self.viewer.state.slices = (i, 0, 0)


class TestReprojection:
    """
    Benchmark that exercises WCS reprojection.
    """

    def setup(self):
        self.data_collection, self.viewer = get_data_collection_and_viewer()
        load_wcs_data(self.data_collection)
        self.viewer.add_data(self.data_collection[0])

    def time_reprojection(self):
        self.viewer.add_data(self.data_collection[1])


class TestSelection:
    """
    Benchmark that exercises WCS reprojection.
    """

    def setup(self):
        self.data_collection, self.viewer = get_data_collection_and_viewer()
        load_wcs_data(self.data_collection)
        self.viewer.add_data(self.data_collection[0])
        self.viewer.add_data(self.data_collection[1])
        roi = CircularROI(256, 256, 120)
        py1, px1 = self.data_collection[0].pixel_component_ids
        self.subset_state = RoiSubsetState(px1, py1, roi)
        self.viewer.register_to_hub(self.data_collection.hub)

    def time_selection(self):
        self.data_collection.new_subset_group(label='Subset', subset_state=self.subset_state)
