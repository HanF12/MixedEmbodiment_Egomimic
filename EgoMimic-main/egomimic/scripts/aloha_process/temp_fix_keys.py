import h5py
from mimicplay_data_process import replace_key_names
import egomimic.utils.egomimicUtils as egomimicUtils

h5py_file = h5py.File(
    "/coc/flash7/datasets/egoplay/handStackingPublic/handStackingMimicplay.hdf5", "r+"
)

egomimicUtils.nds(h5py_file)

key_dict = {
    "obs/front_image_1": "obs/front_img_1",
    "obs/front_image_2": "obs/front_img_2",
}

replace_key_names(h5py_file, key_dict)

egomimicUtils.nds(h5py_file)
