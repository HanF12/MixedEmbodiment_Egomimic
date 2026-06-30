"""
Implementation of Action Chunking with Transformers (ACT).
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import robomimic.utils.tensor_utils as TensorUtils
from egomimic.algo import register_algo_factory_func, PolicyAlgo
from robomimic.algo.bc import BC_VAE

from egomimic.utils.egomimicUtils import nds
import matplotlib.pyplot as plt
import robomimic.utils.obs_utils as ObsUtils

from egomimic.configs import config_factory

from egomimic.models.act_nets import Transformer, StyleEncoder


import robomimic.models.base_nets as BaseNets
import egomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

import json


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """
    algo_class, algo_kwargs = ACT, {}

    return algo_class, algo_kwargs


class ACTModel(nn.Module):
    '''
    ACT Model closely following DETRVAE from ACT but using standard torch.nn components

    backbones : visual backbone per cam input
    transformer : encoder-decoder transformer
    encoder : style encoder
    latent_dim : style var dim
    a_dim : action dim
    state_dim : proprio dim
    num_queries : predicted action dim
    camera_names : list of camera inputs
    '''
    def __init__(
            self,
            backbones,
            transformer,
            encoder,
            latent_dim,
            a_dim,
            state_dim,
            num_queries,
            camera_names,
            num_channels,
    ):
        super(ACTModel, self).__init__()

        self.action_dim = a_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d

        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.num_channels = num_channels

        if backbones is not None:
            self.input_proj = nn.Conv2d(
                self.num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
        else:
            self.backbones = None

        ## ACTSP
        self.cls_embed = nn.Embedding(1, hidden_dim)

        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )

        self.encoder_action_proj = nn.Linear(
                self.action_dim, hidden_dim
            )  # project robot action to embedding
        
        self.encoder_joint_proj = nn.Linear(
                self.state_dim, hidden_dim
            )  # project robot qpos to embedding

        self.transformer_input_proj = nn.Linear(self.state_dim, hidden_dim)

        self.action_head = nn.Linear(hidden_dim, self.action_dim)
    
    def forward(self, qpos, actions, image, is_pad=None, env_state=None):
        return self._forward(
                qpos=qpos,
                actions=actions,
                image=image,
                encoder_action_proj=self.encoder_action_proj,
                encoder_joint_proj=self.encoder_joint_proj,
                transformer_input_proj=self.transformer_input_proj,
                action_head=self.action_head,
                camera_names=self.camera_names,
                is_pad=is_pad,
        )
    
    def _forward(self, qpos, actions, image, encoder_action_proj=None, encoder_joint_proj=None, transformer_input_proj=None, action_head=None, camera_names=None, env_state=None, is_pad=None, aux_action_head=None):
        '''
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim

        '''

        is_training = actions is not None
        batch_size = qpos.size(0)

        if is_training:
            actions_encod = encoder_action_proj(actions)
            qpos_encod = encoder_joint_proj(qpos)
            # Use StyleEncoder to get latent distribution and sample
            dist = self.encoder(qpos_encod, actions_encod)
            mu = dist.mean
            logvar = dist.scale.log() * 2
            latent_sample = dist.rsample()
        else:
            # Inference mode, use zeros for latent vector
            mu = logvar = None
            latent_sample = torch.zeros(batch_size, self.latent_dim, device=qpos.device)

        latent_input = self.latent_out_proj(latent_sample)  # [batch_size, hidden_dim]

        all_cam_features = []
        for cam_id in range(len(camera_names)):
            features = self.backbones[cam_id](image[:, cam_id])
            features = self.input_proj(features)
            all_cam_features.append(features)

        src = torch.cat(all_cam_features, dim=-1)  # [B, hidden_dim, H, W * num_cameras]

        batch_size, hidden_dim, height, width = src.shape
        src = src.flatten(2).permute(0, 2, 1)  # [B, S, hidden_dim], S = H * W * num_cameras]

        proprio_input = transformer_input_proj(qpos).unsqueeze(1)  # [B, 1, hidden_dim]
        latent_input = latent_input.unsqueeze(1)  # [B, 1, hidden_dim]

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, num_queries, hidden_dim]

        tgt = query_embed # tgt = torch.zeros_like(query_embed) + query_embed. ACT passes zeros to decoder
        
        src = torch.cat(
            [latent_input,
            proprio_input,
            src],
            axis=1
        ) # [B, S + 2, hidden_dim]
        
        # Learnable additional pos embed for latent input, proprio input
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1) #[B, 2, hidden_dim]
        src[:, :2, :] += additional_pos_embed 

        hs_queries = self.transformer(src, tgt) # [B, tgt, hidden_dim]

        action_pred = action_head(hs_queries)  # [B, num_queries, action_dim]
        is_pad_pred = self.is_pad_head(hs_queries)  # [B, num_queries, 1]

        # aux action head for 2 head output
        if aux_action_head:
            action_pred_aux = aux_action_head(hs_queries)
            return (action_pred, action_pred_aux), is_pad_pred, [mu, logvar]

        return action_pred, is_pad_pred, [mu, logvar]

        
class ACT(BC_VAE):
    """
    BC training with a VAE policy.
    """

    def build_model_opt(self, policy_config):
        """
        Builds networks and optimizers for BC algo.
        """
        # return build_ACT_model_and_optimizer(policy_config)

        backbones = []
        if len(policy_config["camera_names"]) > 0:
            for cam_name in policy_config["camera_names"]:
                backbone_class_name = policy_config["backbone_class_name"]
                backbone_kwargs = policy_config["backbone_kwargs"]

                try:
                    backbone_class = getattr(BaseNets, backbone_class_name)
                except AttributeError:
                    raise ValueError(f"Unsupported backbone class: {backbone_class_name}")
                
                backbone = backbone_class(**backbone_kwargs)
                backbones.append(backbone)
        else:
            backbones = None

            # assume camera input shape is same for all TODO dynamic size
        
        assert len(backbones) > 0 and backbones is not None , "There must be atleast one vision backbone"

        cam_name = policy_config["camera_names"][0]  
        input_shape = self.obs_key_shapes[cam_name]  # (C, H, W)
        num_channels = backbones[0].output_shape(input_shape)[0]

        
        transformer = Transformer(
            d=policy_config["hidden_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["dec_layers"],
            dropout=policy_config["dropout"],
        )

        style_encoder = StyleEncoder(
            act_len=policy_config["action_length"],
            hidden_dim=policy_config["hidden_dim"],
            latent_dim=policy_config["latent_dim"],
            h=policy_config["nheads"],
            d_ff=policy_config["dim_feedforward"],
            num_layers=policy_config["enc_layers"],
            dropout=policy_config["dropout"],
        )

        model = ACTModel(
            backbones=backbones,
            transformer=transformer,
            encoder=style_encoder,
            latent_dim=policy_config["latent_dim"],
            a_dim=policy_config["a_dim"],
            state_dim=policy_config["state_dim"],
            num_queries=policy_config["num_queries"],
            camera_names=policy_config["camera_names"],
            num_channels=num_channels,
        )

        model.cuda()

        return model


    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.nets = nn.ModuleDict()
        self.chunk_size = self.global_config["train"]["seq_length"]
        self.camera_keys = self.obs_config["modalities"]["obs"]["rgb"].copy()
        self.proprio_keys = self.obs_config["modalities"]["obs"]["low_dim"].copy()
        self.obs_keys = self.proprio_keys + self.camera_keys

        self.proprio_dim = 0
        for k in self.proprio_keys:
            self.proprio_dim += self.obs_key_shapes[k][0]

        backbone_class_name = self.obs_config["encoder"]["rgb"]["core_kwargs"][
            "backbone_class"
        ]
        backbone_kwargs = self.obs_config["encoder"]["rgb"]["core_kwargs"][
            "backbone_kwargs"
        ]

        policy_config = {
            "num_queries": self.global_config.train.seq_length,
            "hidden_dim": self.algo_config.act.hidden_dim,
            "dim_feedforward": self.algo_config.act.dim_feedforward,
            "backbone": self.algo_config.act.backbone,
            "enc_layers": self.algo_config.act.enc_layers,
            "dec_layers": self.algo_config.act.dec_layers,
            "nheads": self.algo_config.act.nheads,
            "latent_dim": self.algo_config.act.latent_dim,
            "action_length": self.chunk_size,
            "a_dim": self.ac_dim,
            "ac_key": self.ac_key,
            "state_dim": self.proprio_dim,
            "camera_names": self.camera_keys,
            "backbone_class_name": backbone_class_name,
            "backbone_kwargs": backbone_kwargs,
            "dropout": self.algo_config.act.get("dropout", 0.1)
        }

        self.kl_weight = self.algo_config.act.kl_weight
        model = self.build_model_opt(policy_config)    

        self.nets["policy"] = model
        self.nets = self.nets.float().to(self.device)

        self.temporal_agg = False
        self.query_frequency = self.chunk_size  # TODO maybe tune

        self._step_counter = 0
        self.a_hat_store = None

        rand_kwargs = self.global_config.observation.encoder.rgb.obs_randomizer_kwargs
        self.color_jitter = transforms.ColorJitter(
            brightness=(rand_kwargs.brightness_min, rand_kwargs.brightness_max), 
            contrast=(rand_kwargs.contrast_min, rand_kwargs.contrast_max), 
            saturation=(rand_kwargs.saturation_min, rand_kwargs.saturation_max), 
            hue=(rand_kwargs.hue_min, rand_kwargs.hue_max)
        )

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

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorcal VAEs.
        """

        return super().train_on_batch(batch, epoch, validate=validate)

    def _modality_check(self, batch):
        """
        Helper function to check if the batch is robot or hand data.
        """
        if (batch["type"] == 0).all():
            modality = "robot"
        elif (batch["type"] == 1).all():
            modality = "hand"
        else:
            raise ValueError(
                "Got mixed modalities, current implementation expects either robot or hand data only."
            )
        return modality

    def _robomimic_to_act_data(self, batch, cam_keys, proprio_keys):
        qpos = [batch["obs"][k] for k in proprio_keys]
        qpos = torch.cat(qpos, axis=1)

        images = []

        if len(cam_keys) > 0:
            for cam_name in cam_keys:
                image = batch["obs"][cam_name]
                if self.nets.training:
                    image = self.color_jitter(image)
                image = image.unsqueeze(axis=1)
                images.append(image)
            images = torch.cat(images, axis=1)
        else:
            images = None

        env_state = torch.zeros([qpos.shape[0], 10]).cuda()  # this is not used

        actions = batch[self.ac_key] if self.ac_key in batch else None
        is_pad = batch["obs"]["pad_mask"] == 0  # from 1.0 or 0 to False and True
        is_pad = is_pad.squeeze(dim=-1)
        B, T = is_pad.shape
        assert T == self.chunk_size

        return qpos, images, env_state, actions, is_pad

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

        qpos, images, env_state, actions, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos=qpos, image=images, env_state=env_state, actions=actions, is_pad=is_pad
        )
        total_kld, dim_wise_kld, mean_kld = self.kl_divergence(mu, logvar)
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict["l1"] = l1
        loss_dict["kl"] = total_kld[0]

        predictions = OrderedDict(
            actions=actions,
            kl_loss=loss_dict["kl"],
            reconstruction_loss=loss_dict["l1"],
        )

        return predictions

    def forward_eval(self, batch, unnorm_stats):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            predictions (dict): dictionary containing network outputs
        """

        qpos, images, env_state, _, is_pad = self._robomimic_to_act_data(
            batch, self.camera_keys, self.proprio_keys
        )
        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](
            qpos=qpos, image=images, env_state=env_state, actions=None, is_pad=is_pad
        )

        predictions = OrderedDict()
        predictions[self.ac_key] = a_hat

        if unnorm_stats:
            predictions = ObsUtils.unnormalize_batch(predictions, unnorm_stats)

        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        raise NotImplementedError("Not in use, but reference forward_eval if you want to use this.")

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._step_counter = 0

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

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
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
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def kl_divergence(self, mu, logvar):
        batch_size = mu.size(0)
        assert batch_size != 0
        if mu.data.ndimension() == 4:
            mu = mu.view(mu.size(0), mu.size(1))
        if logvar.data.ndimension() == 4:
            logvar = logvar.view(logvar.size(0), logvar.size(1))

        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        dimension_wise_kld = klds.mean(0)
        mean_kld = klds.mean(1).mean(0, True)

        return total_kld, dimension_wise_kld, mean_kld


class TestModel:
    def __init__(self, config_path):
        ext_cfg = self.load_config(config_path)
        config = config_factory(ext_cfg["algo_name"])

        with config.values_unlocked():
            config.update(ext_cfg)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ac_dim = 7  # Action dimension
        obs_key_shapes = {
            'joint_positions': (7,),
            'front_img_1': (3, 480, 640),
            'right_wrist_img': (3, 480, 640),
        }

        self.act_algo = ACT(
            algo_config=config.algo,  # Use the appropriate section from the config object
            obs_config=config.observation,
            global_config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=ac_dim,
            device=device,
        )

        self.act_algo.train_config = config.train

        # Create networks
        self.act_algo._create_networks()

    def load_config(self, config_path):
        """Load the config from a JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def run_test(self):
        batch_size = 2
        seq_length = self.act_algo.train_config.seq_length  # Access via the config object

        # Create a dummy batch
        dummy_batch = {
            'obs': {
                'joint_positions': torch.randn(batch_size, seq_length, *self.act_algo.obs_key_shapes['joint_positions']),
                'front_img_1': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['front_img_1']), dtype=torch.uint8),
                'right_wrist_img': torch.randint(0, 256, (batch_size, seq_length, *self.act_algo.obs_key_shapes['right_wrist_img']), dtype=torch.uint8),
                'pad_mask': torch.ones(batch_size, seq_length, 1),
            },
            'actions_joints_act': torch.randn(batch_size, seq_length, self.act_algo.ac_dim),
        }

        # Process the batch for training
        batch = self.act_algo.process_batch_for_training(dummy_batch, 'actions_joints_act')

        print("Processed Batch:", batch)

        # Move batch to device
        batch = self.to_device(batch, self.act_algo.device)

        # Ensure the model is in training mode
        self.act_algo.nets['policy'].train()

        # Perform forward pass for training
        predictions = self.act_algo._forward_training(batch)

        # Compute losses
        losses = self.act_algo._compute_losses(predictions, batch)

        print("Predictions:")
        for key, value in predictions.items():
            print(f"{key}: {value}")

        print("\nLosses:")
        for key, value in losses.items():
            print(f"{key}: {value.item() if isinstance(value, torch.Tensor) else value}")

    def to_device(self, batch, device):
        """Utility function to move data to the specified device."""
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v, device) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        else:
            return batch
