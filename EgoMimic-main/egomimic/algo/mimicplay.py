"""
Implementation of MimicPlay and PlayLMP baselines (formalized as BC-RNN (robomimic) and BC-trans)
"""

from collections import OrderedDict

import copy
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import egomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import egomimic.utils.file_utils as FileUtils
from egomimic.algo import register_algo_factory_func, PolicyAlgo
from egomimic.algo.GPT import GPT_wrapper, GPT_wrapper_scratch
from robomimic.algo.bc import BC_Gaussian, BC_RNN

from egomimic.utils.obs_utils import keep_keys
import time


@register_algo_factory_func("mimicplay")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MimicPlay algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    if algo_config.highlevel.enabled:
        if algo_config.lowlevel.enabled:
            return Lowlevel_GPT_mimicplay, {}
        else:
            if (
                algo_config.gmm.kl == True
                and algo_config.gmm.domain_discriminator == False
            ):
                return KLDiv_Highlevel_GMM_pretrain, {}
            elif (
                algo_config.gmm.kl == False
                and algo_config.gmm.domain_discriminator == True
            ):
                return DomainDiscriminator_Highlevel_GMM_pretrain, {}
            elif (
                algo_config.gmm.kl == False
                and algo_config.gmm.domain_discriminator == False
            ):
                return Highlevel_GMM_pretrain, {}
            else:
                Exception("Invalid config for highlevel training")
            return Highlevel_GMM_pretrain, {}  # Highlevel_GMM_pretrain, {}
    else:
        if algo_config.lowlevel.enabled:
            return Baseline_GPT_from_scratch, {}
        else:
            return BC_RNN_GMM, {}


class Domain_Discriminator(nn.Module):
    def __init__(self, in_features=67):
        super(Domain_Discriminator, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(
            nn.Linear(67, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Highlevel_GMM_pretrain(BC_Gaussian):
    """
    MimicPlay highlevel latent planner, trained to generate 3D trajectory based on observation and goal image.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        # del self.obs_shapes['robot0_eef_pos_future_traj']
        # Here obs_shapes should be the rgb images and current hand pos
        self.obs_shapes = keep_keys(
            self.obs_shapes, self.global_config.policy_inputs.high_level
        )
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
        )

        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

        self.both_human_robot = False

    def process_batch_for_training(self, batch):
        assert False, "Must pass in ac_key for this class"

    def process_batch_for_training(self, batch, ac_key):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        input_batch = dict()
        input_batch["obs"] = {
            k: batch["obs"][k][:, 0, :]
            for k in batch["obs"]
            if k != "pad_mask" and k != "type"
        }
        input_batch["obs"]["pad_mask"] = batch["obs"]["pad_mask"]
        input_batch["goal_obs"] = batch.get(
            "goal_obs", None
        )  # goals may not be present
        if ac_key in batch:
            input_batch[ac_key] = batch[ac_key]

        if "type" in batch:
            input_batch["type"] = batch["type"]

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

        # call parent class method
        # return super().process_batch_for_training(batch)
        # return self.process_batch_for_training(batch)
    
    def postprocess_batch_for_training(self, batch, normalization_stats, normalize_actions=True):
        batch = super().postprocess_batch_for_training(batch, normalization_stats, normalize_actions)

        B, T, A = batch[self.ac_key].shape
        self.orig_shape = batch[self.ac_key].shape
        batch[self.ac_key] = batch[self.ac_key].view(B, -1)

        return batch

    def _get_latent_plan(self, obs, goal):
        # assert 'agentview_image' in obs.keys() # only visual inputs can generate latent plans
        # Todo: generalize this to take inputs from cfg.  For us, it needs to take two input images
        # bs = self.global_config.train.batch_size
        # seq = self.global_config.train.seq_length

        if len(obs[self.global_config.observation.modalities.obs.rgb[0]].shape) == 5:
            bs, seq, C, H, W = obs[
                self.global_config.observation.modalities.obs.rgb[0]
            ].shape
            for k in obs.keys():
                # assert obs[k].size()[0] == bs, "batch size doesn't match"
                assert (
                    obs[k].size()[1] == 10
                ), "seq len doesn't match, if changed seq len can rm this assert"
                # merge the first two dimensions
                obs[k] = obs[k].view(-1, *obs[k].size()[2:])
                goal[k] = goal[k].view(-1, *goal[k].size()[2:])

            # bs, seq, c, h, w = obs['agentview_image'].size()

            # for item in ['agentview_image']:
            #     obs[item] = obs[item].view(bs * seq, c, h, w)
            #     goal[item] = goal[item].view(bs * seq, c, h, w)

            # obs['robot0_eef_pos'] = obs['robot0_eef_pos'].view(bs * seq, 3)

            # TODO: DO NOT COMMIT, the obs here needs hand_loc, but we only have ee_pose, and these are not in the same space, shape etc.  Stubbing hand_loc for now.  Ideally ee_pose would be named the same as hand_loc, and would also be in the same space
            breakpoint()  # fix
            obs["hand_loc"] = torch.ones((320, 4)).to(obs["front_img_1"].device)
            dists, enc_out, mlp_out = self.nets["policy"].forward_train(
                obs_dict=obs, goal_dict=goal, return_latent=True
            )
            del obs["hand_loc"]

            act_out_all = dists.mean
            act_out = act_out_all

            # unmerge the first two dimensions
            for k in obs.keys():
                obs[k] = obs[k].view(bs, seq, *obs[k].size()[1:])
                goal[k] = goal[k].view(bs, seq, *goal[k].size()[1:])

            # for item in ['agentview_image']:
            #     obs[item] = obs[item].view(bs, seq, c, h, w)
            #     goal[item] = goal[item].view(bs, seq, c, h, w)

            # obs['robot0_eef_pos'] = obs['robot0_eef_pos'].view(bs, seq, 3)

            enc_out_feature_size = enc_out.size()[1]
            mlp_out_feature_size = mlp_out.size()[1]

            mlp_out = mlp_out.view(bs, seq, mlp_out_feature_size)
        else:
            dists, enc_out, mlp_out = self.nets["policy"].forward_train(
                obs_dict=obs, goal_dict=goal, return_latent=True
            )

            act_out_all = dists.mean
            act_out = act_out_all

        return act_out, mlp_out

    def forward_eval(self, batch, unnorm_stats=None):
        """
        returns outdict of form {self.ac_key: (B, ac_dim)}
        """
        with torch.no_grad():
            dists, latent, _ = self.nets["policy"].forward_train(
                obs_dict=batch["obs"], goal_dict=batch["goal_obs"], return_latent=True
            )

            dists = dists.mean
            out_dict = {
                self.ac_key: dists
            }
            if self.global_config.train.prestacked_actions:
                out_dict[self.ac_key] = out_dict[self.ac_key].view(self.orig_shape)
            if unnorm_stats:
                out_dict = ObsUtils.unnormalize_batch(out_dict, normalization_stats=unnorm_stats)
            
            return out_dict     

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        ## Check if data from single data source (Robot or Human)
        if isinstance(batch, dict):
            self.both_human_robot = False
        elif isinstance(batch, list):
            self.both_human_robot = True

        dists, enc_out, mlp_out = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], goal_dict=batch["goal_obs"], return_latent=True
        )

        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch[self.ac_key])

        predictions = OrderedDict(
            log_probs=log_probs,
            enc_out=enc_out,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class KLDiv_Highlevel_GMM_pretrain(Highlevel_GMM_pretrain):
    def _create_networks(self):
        super()._create_networks()
        assert (
            self.algo_config.gmm.kl == True
        ), "KL must be enabled in config to call KL_Div_Highlevel_GMM_pretrain class"
        self.kl_weight = self.algo_config.gmm.kl_weight

    def _compute_losses(self, predictions, batch):
        assert (
            self.algo_config.gmm.kl == True
        ), "KL must be enabled in config to call KL_Div_Highlevel_GMM_pretrain class"
        assert (
            self.both_human_robot == True
        ), "Batched data from dual source (robot and human) not provided"
        base_losses = super()._compute_losses(predictions, batch)

        input_kl = F.log_softmax(predictions["enc_out_2"], dim=1)  # robot
        target_kl = F.softmax(predictions["enc_out"], dim=1)  # human
        kl_div_loss = self.kl_weight * torch.nn.KLDivLoss(reduction="batchmean")(
            input_kl, target_kl
        )
        base_losses["kl_div_loss"] = kl_div_loss

        action_loss = -predictions["log_probs"].mean() + kl_div_loss.mean()
        base_losses["log_probs"] = -action_loss
        base_losses["action_loss"] = action_loss
        return base_losses

    def log_info(self, info):
        base_logs = super().log_info(info)
        base_logs["kl_div_loss"] = info["losses"]["kl_div_loss"].item()
        return base_logs


class DomainDiscriminator_Highlevel_GMM_pretrain(Highlevel_GMM_pretrain):
    def _create_networks(self):
        super()._create_networks()
        assert (
            self.algo_config.gmm.domain_discriminator == True
        ), "Domain Discriminator must be enabled in config to call DomainDiscriminator_Highlevel_GMM_pretrain class"

        self.discriminator = Domain_Discriminator()
        self.discriminator = self.discriminator.float().to(self.device)
        # Parameters for the optimizer
        learning_rate = 0.0002  # Common starting learning rate for Adam in GANs
        betas = (
            0.5,
            0.999,
        )  # Betas used typically in GANs to control the moving averages
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.1 * learning_rate, betas=betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.nets["policy"]._modules["nets"]["encoder"].parameters(),
            lr=10 * learning_rate,
            betas=betas,
        )
        self.training_step = 0

    def _compute_losses(self, predictions, batch):
        assert (
            self.algo_config.gmm.domain_discriminator == True
        ), "Domain Discriminator must be enabled in config to call DomainDiscriminator_Highlevel_GMM_pretrain class"
        assert (
            self.both_human_robot == True
        ), "Batched data from dual source (robot and human) not provided"

        base_losses = super()._compute_losses(predictions, batch)
        human_latent, robot_latent = predictions["enc_out"], predictions["enc_out_2"]
        real_labels = torch.ones(human_latent.size(0), 1, device=self.device)
        fake_labels = torch.zeros(robot_latent.size(0), 1, device=self.device)

        real_loss = F.binary_cross_entropy(
            self.discriminator(human_latent.detach()), real_labels
        )
        fake_loss = F.binary_cross_entropy(
            self.discriminator(robot_latent.detach()), fake_labels
        )
        discriminator_loss = (real_loss + fake_loss) / 2
        if self.nets.training:
            self.training_step += 1
            if self.training_step % 100 == 0:
                print("Backproping discriminator")
                self.discriminator_optimizer.zero_grad()  # Reset gradients
                discriminator_loss.backward()  # Compute gradients
                self.discriminator_optimizer.step()  # Update weights

            generator_loss = 10 * F.binary_cross_entropy(
                self.discriminator(robot_latent), real_labels
            )
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

        base_losses["discriminator_loss"] = discriminator_loss
        base_losses["generator_loss"] = generator_loss
        return base_losses

    def log_info(self, info):
        base_logs = super().log_info(info)
        base_logs["discriminator_loss"] = info["losses"]["discriminator_loss"].item()
        base_logs["generator_loss"] = info["losses"]["generator_loss"].item()
        return base_logs


class Lowlevel_GPT_mimicplay(BC_RNN):
    """
    MimicPlay lowlevel plan-guided robot controller, trained to output 6-DoF robot end-effector actions conditioned on generated highlevel latent plans
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        self.human_nets, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=self.algo_config.lowlevel.trained_highlevel_planner,
            device=self.device,
            verbose=False,
            update_obs_dict=False,
        )

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        # del self.obs_shapes['agentview_image']
        # del self.obs_shapes["front_img_1"]
        # del self.obs_shapes["front_image_2"]
        self.obs_shapes = keep_keys(
            self.obs_shapes, self.global_config.policy_inputs.low_level
        )
        self.obs_shapes["latent_plan"] = [self.algo_config.highlevel.latent_plan_dim]

        self.nets = nn.ModuleDict()

        self.nets["policy"] = GPT_wrapper(
            self.algo_config.lowlevel.feat_dim,
            self.algo_config.lowlevel.n_layer,
            self.algo_config.lowlevel.n_head,
            self.algo_config.lowlevel.block_size,
            self.algo_config.lowlevel.gmm_modes,
            self.algo_config.lowlevel.action_dim,
            self.algo_config.lowlevel.proprio_dim,
            self.algo_config.lowlevel.spatial_softmax_num_kp,
            self.algo_config.lowlevel.gmm_min_std,
            self.algo_config.lowlevel.dropout,
            self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_height,
            self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_width,
        )

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(
            self.goal_ee_traj[current_id : (current_id + self.eval_goal_img_window)]
            - ee_pos,
            dim=1,
        )
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "agentview_image"
        ][:]
        self.goal_ee_traj = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "robot0_eef_pos"
        ][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.0
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]

        input_batch[self.ac_key] = batch[self.ac_key]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        with torch.no_grad():
            _, mlp_feature = self.human_nets.policy._get_latent_plan(
                batch["obs"], batch["goal_obs"]
            )
            batch["obs"]["latent_plan"] = mlp_feature.detach()

        dists = self.nets["policy"].forward_train(batch["obs"])

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch[self.ac_key])

        predictions = OrderedDict(
            log_probs=log_probs,
        )

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        obs_to_use = obs_dict

        with torch.no_grad():
            self.goal_id = min(
                self.current_id + self.algo_config.playdata.eval_goal_gap,
                self.goal_image_length - 1,
            )
            goal_img = {
                "agentview_image": self.goal_image[self.goal_id : (self.goal_id + 1)]
            }
            action, mlp_feature = self.human_nets.policy._get_latent_plan(
                obs_to_use, goal_img
            )
            obs_to_use["latent_plan"] = mlp_feature.detach()
            obs_to_use["guidance"] = action.detach()

            self.current_id = self.find_nearest_index(
                obs_to_use["robot0_eef_pos"], self.current_id
            )

        action = self.nets["policy"].forward_step(obs_to_use)

        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.nets["policy"].reset()
        self.human_nets.policy.reset()
        self.current_id = 0


class Baseline_GPT_from_scratch(BC_RNN):
    """
    BC transformer baseline (an end-to-end version of MimicPlay's lowlevel robot controller (no highlevel planner)).
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert not self.algo_config.highlevel.enabled
        assert self.algo_config.lowlevel.enabled

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        self.nets = nn.ModuleDict()

        self.nets["policy"] = GPT_wrapper_scratch(
            self.algo_config.lowlevel.feat_dim,
            self.algo_config.lowlevel.n_layer,
            self.algo_config.lowlevel.n_head,
            self.algo_config.lowlevel.block_size,
            self.algo_config.lowlevel.gmm_modes,
            self.algo_config.lowlevel.action_dim,
            self.algo_config.lowlevel.proprio_dim,
            self.algo_config.lowlevel.spatial_softmax_num_kp,
            self.algo_config.lowlevel.gmm_min_std,
            self.algo_config.lowlevel.dropout,
            self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_height,
            self.obs_config.encoder.rgb.obs_randomizer_kwargs.crop_width,
        )

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(
            self.goal_ee_traj[current_id : (current_id + self.eval_goal_img_window)]
            - ee_pos,
            dim=1,
        )
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "agentview_image"
        ][:]
        self.goal_ee_traj = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "robot0_eef_pos"
        ][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.0
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]

        input_batch[self.ac_key] = batch[self.ac_key]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        dists = self.nets["policy"].forward_train(batch["obs"], batch["goal_obs"])

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch[self.ac_key])

        predictions = OrderedDict(
            log_probs=log_probs,
        )

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        obs_to_use = obs_dict

        self.goal_id = min(
            self.current_id + self.algo_config.playdata.eval_goal_gap,
            self.goal_image_length - 1,
        )
        goal_img = {
            "agentview_image": self.goal_image[self.goal_id : (self.goal_id + 1)]
        }

        action = self.nets["policy"].forward_step(obs_to_use, goal_img)

        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self.nets["policy"].reset()
        self.current_id = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC-RNN baseline (an end-to-end baseline adapted from robomimic)
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert not self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        self.eval_goal_img_window = self.algo_config.lowlevel.eval_goal_img_window
        self.eval_max_goal_img_iter = self.algo_config.lowlevel.eval_max_goal_img_iter

        self.nets = nn.ModuleDict()

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.buffer = []
        self.current_id = 0
        self.save_count = 0
        self.zero_count = 0

        self.nets = self.nets.float().to(self.device)

    def find_nearest_index(self, ee_pos, current_id):
        distances = torch.norm(
            self.goal_ee_traj[current_id : (current_id + self.eval_goal_img_window)]
            - ee_pos,
            dim=1,
        )
        nearest_index = distances.argmin().item()
        if nearest_index == 0:
            self.zero_count += 1
        if self.zero_count > self.eval_max_goal_img_iter:
            nearest_index += 1
            self.zero_count = 0

        return min(nearest_index + current_id, self.goal_image_length - 1)

    def load_eval_video_prompt(self, video_path):
        self.goal_image = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "agentview_image"
        ][:]
        self.goal_ee_traj = h5py.File(video_path, "r")["data"]["demo_1"]["obs"][
            "robot0_eef_pos"
        ][:]
        self.goal_image = torch.from_numpy(self.goal_image).cuda().float()
        self.goal_ee_traj = torch.from_numpy(self.goal_ee_traj).cuda().float()
        self.goal_image = self.goal_image.permute(0, 3, 1, 2)
        self.goal_image = self.goal_image / 255.0
        self.goal_image_length = len(self.goal_image)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]

        key_list = copy.deepcopy(list(input_batch["obs"].keys()))
        for key in key_list:
            input_batch["obs"][key] = input_batch["obs"][key]

        input_batch["goal_obs"] = batch["goal_obs"]
        input_batch[self.ac_key] = batch[self.ac_key]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]
        log_probs = dists.log_prob(batch[self.ac_key])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(
                batch_size=batch_size, device=self.device
            )

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self.goal_id = min(
            self.current_id + self.algo_config.playdata.eval_goal_gap,
            self.goal_image_length - 1,
        )
        goal_dict = {
            "agentview_image": self.goal_image[
                self.goal_id : (self.goal_id + 1)
            ].unsqueeze(0)
        }

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state
        )
        return action

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
