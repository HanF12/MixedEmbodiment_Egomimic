import os
from collections import OrderedDict

import numpy as np
import psutil
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import torch
from pytorch_lightning import LightningModule
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo.algo import PolicyAlgo
import egomimic.utils.val_utils as ValUtils
from egomimic.utils.egomimicUtils import nds
from egomimic.pl_utils.pl_data_utils import DualDataModuleWrapper, json_to_config
from egomimic.algo import algo_factory
import robomimic.utils.file_utils as FileUtils
from egomimic.configs import config_factory
import json


class ModelWrapper(LightningModule):
    """
    Wrapper class around robomimic models to ensure compatibility with Pytorch Lightning.
    """

    def __init__(self, config_json, shape_meta, datamodule):
        """
        Args:
            model (PolicyAlgo): robomimic model to wrap.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule"])

        config = json_to_config(config_json)
        model = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            device="cuda",  # default to cpu, pl will move to gpu
        )
        self.model = model
        self.nets = (
            self.model.nets
        )  # to ensure the lightning module has access to the model's parameters
        try:
            self.params = self.model.nets["policy"].params
        except:
            pass
        self.step_log_all_train = []
        self.step_log_all_valid = []

        self.datamodule = datamodule
        self.dual_dl = isinstance(datamodule, DualDataModuleWrapper)

        # TODO __init__ should take the config, and init the model here.  Then save_hyperparameters will just save the config rather than the model

    def training_step(self, batch, batch_idx):
        DUAL_DL = isinstance(batch, list)
        # plt.imsave("debug/front_img_1.png", batch[0]["obs"]["front_img_1"][0, 0].cpu().numpy())

        # full_batch = batch
        # batch = full_batch[0]
        self.train()
        loss_dicts = []
        if not DUAL_DL:
            batch = [batch]
        ac_keys = (
            [
                self.model.global_config.train.ac_key,
                self.model.global_config.train.ac_key_hand,
            ]
            if self.dual_dl
            else [self.model.global_config.train.ac_key]
        )
        norm_dicts = (
            [
                self.datamodule.train_dataset1.get_obs_normalization_stats(),
                self.datamodule.train_dataset2.get_obs_normalization_stats()
            ]
            if self.dual_dl
            else [self.datamodule.train_dataset.get_obs_normalization_stats()]
        )
        for batch, ac_key, norm_dict in zip(batch, ac_keys, norm_dicts):
            # batch["obs"] = ObsUtils.process_obs_dict(batch["obs"])
            info = PolicyAlgo.train_on_batch(
                self.model, batch, self.current_epoch, validate=False
            )
            batch = self.model.process_batch_for_training(batch, ac_key)
            batch = self.model.postprocess_batch_for_training(batch, norm_dict, normalize_actions=self.model.global_config.train.hdf5_normalize_actions)
            predictions = self.model._forward_training(batch)
            losses = self.model._compute_losses(predictions, batch)
            loss_dicts.append(losses)

        # Average over both the hand and robot batch if applicable
        losses = OrderedDict()
        for key in loss_dicts[0].keys():
            losses[key] = torch.mean(
                torch.stack([loss_dict[key] for loss_dict in loss_dicts])
            )

        info["losses"] = TensorUtils.detach(losses)
        self.step_log_all_train.append(self.model.log_info(info))

        # count=0
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         count += 1
        #         # print(name)
        # print("Unused params: ", count)
        # print(self.global_step, self.model.global_config.experiment.epoch_every_n_steps)
        # breakpoint()
        return losses["action_loss"]

    def configure_optimizers(self):
        if self.model.lr_schedulers["policy"]:
            lr_scheduler_dict = {
                "scheduler": self.model.lr_schedulers["policy"],
                "interval": "step",
                "frequency": 1,
            }
            return {
                "optimizer": self.model.optimizers["policy"],
                "lr_scheduler": lr_scheduler_dict,
            }
        else:
            return {
                "optimizer": self.model.optimizers["policy"],
            }

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(optimizer_idx)

    def custom_eval(self, video_dir):
        self.eval()
        self.zero_grad()
        with torch.no_grad():
            valid_step_log = ValUtils.evaluate_high_level_policy(
                self.model,
                self.datamodule.val_dataloader_1(),
                video_dir,
                ac_key=self.model.global_config.train.ac_key,
            )  # save vid only once every video_freq epochs

        return valid_step_log

    def on_train_epoch_start(self):
        # flatten and take the mean of the metrics
        log = {}
        for i in range(len(self.step_log_all_train)):
            for k in self.step_log_all_train[i]:
                if k not in log:
                    log[k] = []
                log[k].append(self.step_log_all_train[i][k])
        log_all = dict((k, float(np.mean(v))) for k, v in log.items())
        for k in self.model.optimizers:
            for i, param_group in enumerate(self.model.optimizers[k].param_groups):
                log_all["Optimizer/{}{}_lr".format(k, i)] = param_group["lr"]
        for k, v in log_all.items():
            self.log("Train/" + k, v, sync_dist=True)
        self.step_log_all_train = []

        val_freq = self.model.global_config.experiment.validation_freq
        video_freq = self.model.global_config.experiment.save.video_freq

        if self.current_epoch % val_freq == 0 and self.current_epoch != 0:
            if self.global_rank == 0:
                # Perform custom validation

                with torch.no_grad():
                    self.eval()
                    self.zero_grad()
                    pass_vid = (
                        os.path.join(self.trainer.default_root_dir, "videos")
                        if self.current_epoch % video_freq == 0
                        else None
                    )
                    valid_step_log = ValUtils.evaluate_high_level_policy(
                        self.model,
                        self.datamodule.val_dataloader_1(),
                        pass_vid,
                        max_samples=self.model.global_config.experiment.validation_max_samples,
                        ac_key=self.model.global_config.train.ac_key,
                        type="robot",
                    )  # save vid only once every video_freq epochs

                    if self.dual_dl:
                        valid_step_log_2 = ValUtils.evaluate_high_level_policy(
                            self.model,
                            self.datamodule.val_dataloader_2(),
                            pass_vid,
                            max_samples=self.model.global_config.experiment.validation_max_samples,
                            ac_key=self.model.global_config.train.ac_key_hand,
                            type="hand",
                        )  # save vid only once every video_freq epochs
                    self.train()
                for k, v in valid_step_log.items():
                    self.log("Valid/" + k, v, sync_dist=False)
                if self.dual_dl:
                    for k, v in valid_step_log_2.items():
                        self.log("Valid/" + k, v, sync_dist=False)

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / int(1e9)
        print("\nEpoch {} Memory Usage: {} GB\n".format(self.current_epoch, mem_usage))
        # self.log('epoch', self.trainer.current_epoch)
        self.log("System/RAM Usage (GB)", mem_usage, sync_dist=True)

        return super().on_train_epoch_start()

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        if False and self.model.lr_warmup:
            # lr warmup schedule taken from Gato paper
            # update params
            initial_lr = 0
            target_lr = self.model.optim_params["policy"]["learning_rate"]["initial"]
            # manually warm up lr without a scheduler
            schedule_iterations = 10000
            if self.global_step < schedule_iterations:
                for pg in self.optimizers().param_groups:
                    pg["lr"] = (
                        initial_lr
                        + (target_lr - initial_lr)
                        * self.global_step
                        / schedule_iterations
                    )
            else:
                scheduler.step(self.global_step - schedule_iterations)
        else:
            scheduler.step(self.global_step)
