## Patch Notes
Dirty laundry
- Color jitter is manually implemented for ACT in _robomimic_to_act_data rather than using the ObsUtils color jitter
- hardcoded extrinsics in val_utils.py
- Added ac_key under base Algo in robomimic, I suppose this could just access the model.global_config
- I haven't tested whether aloha_to_robomimic_v2 works with highlevelGMMPretrain

- Remember: type == 0 is robot, type==1 is hand
- Now that our hdf5's have either hand or robot data, we can just specify this from the config.  So simply set config.train.data_type and config.train.data2_type to hand or robot.
- Make sure to set train.prestacked_actions = True when dataset actions are prestacked.  Hand data is prestacked and of shape (N, T, 3) bc coordinate frame changes each step.  Set seq_length to load should be 1 for this case


Dataloader should output batch with following format.  Not currently using dones or rewards
```
dict with keys:  dict_keys(['actions', 'rewards', 'dones', 'pad_mask', 'obs', 'type'])
actions: (1, 30)
rewards: (1, 1)
dones: (1, 1)
pad_mask: (1, 1)
obs: dict with keys:  dict_keys(['ee_pose', 'front_img_1', 'pad_mask'])
        ee_pose: (1, 3)
        front_img_1: (1, 480, 640, 3)
        pad_mask: (1, 1)
type: int
```