import os
import random
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils

from robomimic.utils.dataset import SequenceDataset
import time

from cProfile import Profile
from pstats import SortKey, Stats
from egomimic.utils.egomimicUtils import nds


class PlaydataSequenceDataset(SequenceDataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        goal_obs_gap,
        type,
        ac_key,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
        prestacked_actions=False,
        hdf5_normalize_actions=False
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            load_next_obs (bool): whether to load next_obs from the dataset

            prestacked_actions (bool): whether actions were stacked already in the dataset
        """
        super().__init__(
            hdf5_path=hdf5_path,
            obs_keys=obs_keys,
            dataset_keys=dataset_keys,
            frame_stack=frame_stack,
            seq_length=seq_length,
            pad_frame_stack=pad_frame_stack,
            pad_seq_length=pad_seq_length,
            get_pad_mask=get_pad_mask,
            goal_mode=goal_mode,
            hdf5_cache_mode=hdf5_cache_mode,
            hdf5_use_swmr=hdf5_use_swmr,
            hdf5_normalize_obs=hdf5_normalize_obs,
            filter_by_attribute=filter_by_attribute,
            load_next_obs=load_next_obs,
            ac_key=ac_key,
            prestacked_actions=prestacked_actions,
            hdf5_normalize_actions=hdf5_normalize_actions
        )

        self.type = type

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["nstep"]

        self.goal_obs_gap = goal_obs_gap

        self.rgb_keys = [
            k for k in self.obs_keys if ObsUtils.key_is_obs_modality(k, "rgb")
        ]


        self.close_and_delete_hdf5_handle()

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            num_frames_to_stack=self.n_frame_stack
            - 1,  # note: need to decrement self.n_frame_stack by one
            seq_length=self.seq_length,
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "nstep":
            goal_index = (
                min(
                    index_in_demo
                    + random.randint(self.goal_obs_gap[0], self.goal_obs_gap[1]),
                    demo_length,
                )
                - 1
            )

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs",
            dont_load_fut=self.rgb_keys,
        )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs",
                dont_load_fut=self.rgb_keys,
            )

        if goal_index is not None:
            meta["goal_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="obs",
                dont_load_fut=self.rgb_keys,
            )
        
        if self.type == "robot":
            meta["type"] = 0
        elif self.type == "hand":
            meta["type"] = 1
        return meta