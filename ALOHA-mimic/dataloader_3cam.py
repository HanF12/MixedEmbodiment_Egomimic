import torch
from torch.utils.data import Dataset, DataLoader
import cv2 # OpenCV for video processing
import numpy as np
import os
import time # For timing the pre-loading
import h5py # For HDF5 file processing
from torchvision import transforms
import pandas as pd
class PreloadedMultiVideoJointDatasetNEW(Dataset):
    """
    Custom Dataset for loading multiple paired video frames and joint data.
    All video frames and joint data are pre-loaded into RAM during initialization.
    """
    
    def __init__(self, 
                bird_vids_dir, left_arm_vids_dir, right_arm_vids_dir, 
                joint_data_dir, sync_csv_dir, 
                num_queries=100, 
                pad=True, 
                transform='resnet_normalization',
                max_demos=None):
        
        super().__init__()
        self.num_queries = num_queries             
        self.pad = pad  # Whether to pad output action sequence that after demonstration end
        self.demo_lengths = []
        self.transform = transform # any transformation that the data must undergo
        self.temp_cut = 10 # number of frames to cut (from beginning)
        
        normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225])       # preset ResNet normalization values
        base_normalization = transforms.Compose([transforms.ToTensor(), normalize_transform])                   # first apply 0-1 normalization, then apply ResNet normalization
        augment_pipeline_1 = transforms.Compose([                                                               # first random image augmentation
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            normalize_transform])
        augment_pipeline_2 = transforms.Compose([                                                               # second random image augmentation         
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                normalize_transform])
        
        # if we are using pretrained ResNet weights in our backbone, we define the augmentation pipelines.
        if 'resnet_normalization' in self.transform:
            self.image_transforms = [base_normalization]
        else:
            self.image_transforms = [transforms.ToTensor()]
        if 'augment' in self.transform:
            self.image_transforms.append(augment_pipeline_1)
            self.image_transforms.append(augment_pipeline_2)
        

        # Build absolute directory paths
        self.bird_vids_path      = os.path.join(os.getcwd(), bird_vids_dir)
        self.left_arm_vids_path  = os.path.join(os.getcwd(), left_arm_vids_dir)
        self.right_arm_vids_path = os.path.join(os.getcwd(), right_arm_vids_dir)
        self.joint_data_path     = os.path.join(os.getcwd(), joint_data_dir)
        self.sync_csv_path       = os.path.join(os.getcwd(), sync_csv_dir)

        # 1) List all files up front
        bird_vids_all       = sorted([os.path.join(self.bird_vids_path, f) for f in os.listdir(self.bird_vids_path) if f.lower().endswith(".mp4")])
        left_arm_vids_all   = sorted([os.path.join(self.left_arm_vids_path, f) for f in os.listdir(self.left_arm_vids_path) if f.lower().endswith(".mp4")])
        right_arm_vids_all  = sorted([os.path.join(self.right_arm_vids_path, f) for f in os.listdir(self.right_arm_vids_path) if f.lower().endswith(".mp4")])
        joint_npy_all       = sorted([os.path.join(self.joint_data_path, f) for f in os.listdir(self.joint_data_path) if f.lower().endswith(".npy")])
        sync_csv_all        = sorted([os.path.join(self.sync_csv_path, f) for f in os.listdir(self.sync_csv_path) if f.lower().endswith(".csv")])
        if max_demos is not None and max_demos > 0:
            print(f"INFO: Using only the first {max_demos} demonstrations.")
            bird_vids_all      = bird_vids_all[:max_demos]
            left_arm_vids_all  = left_arm_vids_all[:max_demos]
            right_arm_vids_all = right_arm_vids_all[:max_demos]
            joint_npy_all      = joint_npy_all[:max_demos]
            sync_csv_all       = sync_csv_all[:max_demos]
            
        if not sync_csv_all:
            raise FileNotFoundError(f"No .csv files found in {self.sync_csv_path}")
        if not joint_npy_all:
            raise FileNotFoundError(f"No .npy files found in {self.joint_data_path}")
        
        # Store the number of original demonstrations to use in __len__ and __getitem__
        self.num_demos = len(sync_csv_all)

        # Prepare accumulators
        self.bird_frames_list      = []
        self.left_arm_frames_list  = []
        self.right_arm_frames_list = []
        self.joint_data            = []

        total_samples = 0

        print("Loading and synchronizing all recordings...")

        # 2) Since all files share the same base ordering, iterate by index instead of matching rec_id
        for idx in range(self.num_demos):
            bird_vid_path      = bird_vids_all[idx]
            left_arm_vid_path  = left_arm_vids_all[idx]
            right_arm_vid_path  = right_arm_vids_all[idx]
            joint_npy_path     = joint_npy_all[idx]
            csv_path           = sync_csv_all[idx]

            rec_id = os.path.splitext(os.path.basename(csv_path))[0]
            print(f"Processing Recording. Ensure the following files are synced:")
            print(f"      Bird video:      {os.path.basename(bird_vid_path)}")
            print(f"      Left-arm video:  {os.path.basename(left_arm_vid_path)}")
            print(f"      Right-arm video: {os.path.basename(right_arm_vid_path)}")
            print(f"      Joint NPY:       {os.path.basename(joint_npy_path)}")
            print(f"      Sync CSV:        {os.path.basename(csv_path)}")

            # 3) Preload frames for this recording
            bird_frames      = self._load_video_frames([bird_vid_path], resize_factor=1, label=f"Bird View ({rec_id})")
            left_arm_frames  = self._load_video_frames([left_arm_vid_path], resize_factor=1, label=f"Left Arm ({rec_id})")
            right_arm_frames = self._load_video_frames([right_arm_vid_path], resize_factor=1, label=f"Right Arm ({rec_id})")

            # 3.1) forcefully cut x frames from beginning of each 
            bird_frames      = bird_frames[self.temp_cut:]
            left_arm_frames  = left_arm_frames[self.temp_cut:]
            right_arm_frames = right_arm_frames[self.temp_cut:]

            # 4) Load joint data for this recording
            joint_arr = np.load(joint_npy_path)
            T_i, J_total = joint_arr.shape
            take = min(J_total, 7)
            raw_joint_list = [torch.from_numpy(joint_arr[t, :take].astype(np.float32)) for t in range(T_i)]
            
            # 4.1) forcefully cut x joint data from beginning of joint data
            raw_joint_list = raw_joint_list[self.temp_cut:]

            # 5) Read sync CSV and extract indices
            df_sync = pd.read_csv(csv_path)
            
            # ─── Drop all rows that refer to frames we cut away ───
            mask = (
                (df_sync['bird_index']  >= self.temp_cut) &
                (df_sync['left_index']  >= self.temp_cut) &
                (df_sync['right_index'] >= self.temp_cut) &
                (df_sync['joint_index'] >= self.temp_cut)
            )
            df_sync = df_sync[mask].reset_index(drop=True)
            
            # ─── Now rebase to zero and extract ───
            for col in ["bird_index", "left_index", "right_index", "joint_index"]:
                df_sync[col] -= self.temp_cut
            
            required_cols = {"joint_index", "left_index", "right_index","bird_index"}
            if not required_cols.issubset(df_sync.columns):
                raise KeyError(f"Sync CSV '{csv_path}' missing columns: {required_cols - set(df_sync.columns)}")

            bird_idxs, left_arm_idxs, right_index, joint_idxs = df_sync["bird_index"].to_numpy(dtype=np.int64), df_sync["left_index"].to_numpy(dtype=np.int64), df_sync["right_index"].to_numpy(dtype=np.int64),df_sync["joint_index"].to_numpy(dtype=np.int64)
            N_i = len(df_sync)
            
            self.demo_lengths.append(N_i) # store length of each demo for an episodic loader

            # 7) Append the synchronized samples from this recording to global lists
            for j in range(N_i):
                self.bird_frames_list.append(bird_frames[bird_idxs[j]])
                self.left_arm_frames_list.append(left_arm_frames[left_arm_idxs[j]])
                self.right_arm_frames_list.append(right_arm_frames[right_index[j]])
                self.joint_data.append(raw_joint_list[joint_idxs[j]])
                
            total_samples += N_i
            print(f"    → Added {N_i} synced samples (total so far: {total_samples})\n")

        self.num_samples = total_samples
                
        # normalize joints
        all_joints = torch.stack(self.joint_data, dim=0)
        self.joint_mean = all_joints.mean(dim=0)
        self.joint_std  = all_joints.std(dim=0).clamp(min=1e-2)
        
        print(f"Finished initializing. Original demos: {self.num_demos}")

    def __len__(self):
        return self.num_demos

    def __getitem__(self, idx):
        # Determine which view to use (0=base, 1=aug1, 2=aug2) and which original episode to sample from
        view_type = idx // self.num_demos
        original_episode_idx = idx % self.num_demos
        
        ep = original_episode_idx
        ep_start = sum(self.demo_lengths[:ep])
        ep_len = self.demo_lengths[ep]
        # Sample a random start time within this episode
        start_ts_in_ep = np.random.randint(0, ep_len)
        sample_idx = ep_start + start_ts_in_ep # This is the global index for the start frame

        # 1) grab current frames as numpy arrays
        bird_np  = self.bird_frames_list[sample_idx]
        left_np  = self.left_arm_frames_list[sample_idx]
        right_np = self.right_arm_frames_list[sample_idx]
        
        # Get the correct image transformation pipeline
        transform_pipeline = self.image_transforms[view_type]
        
        # Apply the chosen transformation
        bird_frame = transform_pipeline(bird_np)
        left_arm_frame = transform_pipeline(left_np)
        right_arm_frame = transform_pipeline(right_np)
        
        # if True, model will be able to see its own current joint configuration
        joint_raw  = self.joint_data[sample_idx]
        joint_data = (joint_raw - self.joint_mean) / self.joint_std
        
        demo_end_idx = ep_start + ep_len
        slice_end = min(demo_end_idx, sample_idx - 1 + self.num_queries)
        future_list = self.joint_data[max(0, sample_idx - 1) : slice_end]
        
        raw_length = len(future_list)
        pad_length = self.num_queries - raw_length
        
        if self.pad:
            pad_tensor = torch.zeros_like(self.joint_data[0])
            future_list.extend([pad_tensor] * pad_length)

            is_pad = torch.zeros(self.num_queries, dtype=torch.bool)
            if pad_length > 0:
                is_pad[-pad_length:] = True
        else:
            raise RuntimeError('Please enable padding for dataset loading')
    
        # 3) stack into one Tensor of shape [k, 7] and normalize
        prediction_joint_data = torch.stack(
            [ ((torch.as_tensor(x, dtype=torch.float32) - self.joint_mean) / self.joint_std) for x in future_list ],
            dim=0
        )
        stacked_images = torch.stack([left_arm_frame, right_arm_frame, bird_frame], dim=0)
        
        # 4) pack into your output structure
        data = [
            joint_data, 
            stacked_images, 
            prediction_joint_data,
            is_pad
        ]

        return data
    
    def _load_video_frames(self, video_paths, resize_factor=1.0, label="Video"):
        """
        Reads every frame from each .mp4 in `video_paths`, converts BGR→RGB, optionally resizes,
        and returns a single flat list of uint8 arrays [H, W, 3].
        """
        all_frames = []
        for vp in video_paths:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {vp}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            read_count = 0

            while True:
                ret, frame_bgr = cap.read()
                if not ret: break
                read_count += 1

                if resize_factor != 1.0:
                    h, w = frame_bgr.shape[:2]
                    frame_bgr = cv2.resize(frame_bgr, (int(w * resize_factor), int(h * resize_factor)), interpolation=cv2.INTER_AREA)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)

            cap.release()
            print(f"  • {label} '{os.path.basename(vp)}' → read {read_count}/{total} frames")

        return all_frames
    
def sorted_files_in(dir_path, extension):
    full_dir = os.path.join(os.getcwd(), dir_path)
    if not os.path.isdir(full_dir):
        raise FileNotFoundError(f"Directory not found: {full_dir}")
    return sorted([
        os.path.join(full_dir, f)
        for f in os.listdir(full_dir)
        if f.lower().endswith(extension) and os.path.isfile(os.path.join(full_dir, f))
    ])
    
def batch_sanity_check(train_loader: DataLoader, vram_report=True):
    """
    Checks if the batch returned by the DataLoader is valid (hence, a sanity check)
    Also outputs some useful information about the batch, such as shapes and types and estimated VRAM usage
    """
    # Get a single batch from the DataLoader
    batch = next(iter(train_loader))
    print(f"Type of the entire batch: {type(batch)}")
    print(f"Number of items in the batch list: {len(batch)}")
    current_joints_batch = batch[0]
    print(f'Sanity Check: There are currently {current_joints_batch.shape[0]} samples in the batch')
    print("-" * 50)

    # unpack the batch
    current_joints_batch, image_batch, future_joints_batch = batch[:3]

    # batch shape checks
    print("--- BATCH-LEVEL CHECKS ---")
    print(f"Type of current_joints_batch:           {type(current_joints_batch)}")
    print(f"Shape of current_joints_batch:          {current_joints_batch.shape}") # Should be [batch_size, 7], e.g., [8, 7]

    print(f"\nType of image_batch:                  {type(image_batch)}") # This is a list
    print(f"Number of image tensors in image_batch: {len(image_batch)}") # Should be 2 (left_arm, bird)

    left_arm_frames_batch = image_batch[0]
    bird_frames_batch     = image_batch[1]
    print(f"Shape of left_arm_frames_batch:         {left_arm_frames_batch.shape}") # e.g., [8, 3, H, W]
    print(f"Shape of bird_frames_batch:             {bird_frames_batch.shape}")     # e.g., [8, 3, H, W]

    print(f"\nType of future_joints_batch:          {type(future_joints_batch)}")
    print(f"Shape of future_joints_batch:           {future_joints_batch.shape}") # Should be [batch_size, k, 7], e.g., [8, 10, 7]
    print("-" * 50)

    # 4. (Optional) Check the shapes of a SINGLE SAMPLE *within* the batch
    print("--- SINGLE SAMPLE CHECKS (from the first item in the batch) ---")
    first_sample_current_joint = current_joints_batch[0]
    first_sample_left_arm_frame = left_arm_frames_batch[0]
    first_sample_bird_frame = bird_frames_batch[0]
    first_sample_future_joints = future_joints_batch[0]

    print(f"Shape of a single current_joint:            {first_sample_current_joint.shape}") # Should be [7]
    print(f"Shape of a single left_arm_frame:           {first_sample_left_arm_frame.shape}") # Should be [3, H, W]
    print(f"Shape of a single bird_frame:               {first_sample_bird_frame.shape}") # Should be [3, H, W]
    print(f"Shape of a single future_joints sequence:   {first_sample_future_joints.shape}") # Should be [k, 7], e.g., [10, 7]

    current_joints, image_batch, future_joints, pad_batch = batch

    if vram_report:
        # helper to account for nested lists
        def tensor_size_bytes(x):
            if isinstance(x, torch.Tensor):
                return x.element_size() * x.numel()
            elif isinstance(x, (list, tuple)):
                return sum(tensor_size_bytes(y) for y in x)
            else:
                return 0

        total_bytes = 0
        total_bytes += tensor_size_bytes(current_joints)
        total_bytes += tensor_size_bytes(image_batch)
        total_bytes += tensor_size_bytes(future_joints)
        total_bytes += tensor_size_bytes(pad_batch)
        print("-" * 50)
        print(f"VRAM usage for current joint:             {tensor_size_bytes(current_joints)/1e6:.2f} MB")
        print(f"VRAM usage for images in this batch:      {tensor_size_bytes(image_batch)/1e6:.2f} MB")
        print(f"VRAM usage for future joint predictions:  {tensor_size_bytes(future_joints)/1e6:.2f} MB")
        print(f"VRAM usage for padding (this will vary!): {tensor_size_bytes(pad_batch)/1e6:.2f} MB")
        print("-" * 50)
        print(f"Approx. VRAM for this batch:              {total_bytes/1e6:.2f} MB")
        