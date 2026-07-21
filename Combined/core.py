"""
Mixed human+robot DETR-VAE / ACT model.

Shared:
  - bird backbone (cam0)
  - front backbone (cam1)
  - ACT transformer + CVAE encoder

Robot-only:
  - left_wrist backbone (cam2)
  - right_wrist backbone (cam3)
  - state encoder 14 -> H
  - action head H -> 14
  - CVAE action/state projs for 14D

Human-only:
  - state encoder 20 -> H
  - action head H -> 20
  - CVAE action/state projs for 20D

Embodiment token is added into the proprio embedding (keeps ALOHA transformer API intact).
Camera slots are always [bird, front, left_wrist, right_wrist]; masked cams are zeroed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from Combined.config import (
    EMBODIMENT_HUMAN,
    EMBODIMENT_ROBOT,
    HUMAN_STATE_DIM,
    MODEL_CAMERA_NAMES,
    NUM_CAMERAS,
    ROBOT_STATE_DIM,
    validate_camera_names,
)

ALOHA_DIR = (Path(__file__).resolve().parents[1] / "ALOHA-mimic").resolve()
if str(ALOHA_DIR) not in sys.path:
    sys.path.insert(0, str(ALOHA_DIR))

from model import (  # type: ignore  # noqa: E402
    TransformerEncoder,
    TransformerEncoderLayer,
    build_backbone,
    build_transformer,
)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class MixedDETRVAE(nn.Module):
    def __init__(self, backbones, transformer, encoder, num_queries, camera_names):
        super().__init__()
        self.num_queries = int(num_queries)
        self.camera_names = list(camera_names)
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        # Shared spatial proj for all camera feature maps: [B,C,H,W] -> [B,Hdim,H,W]
        self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        # backbones[0]=bird, [1]=front (shared across embodiments)
        # backbones[2]=left_wrist, [3]=right_wrist (robot-only weights; unused for human via mask)
        self.backbones = nn.ModuleList(backbones)

        # Embodiment-specific state encoders
        self.robot_state_proj = nn.Linear(ROBOT_STATE_DIM, hidden_dim)  # [B,14] -> [B,H]
        self.human_state_proj = nn.Linear(HUMAN_STATE_DIM, hidden_dim)  # [B,20] -> [B,H]

        # Embodiment / domain token (2 embeddings: robot, human)
        self.embodiment_embed = nn.Embedding(2, hidden_dim)

        # Embodiment-specific action heads
        self.robot_action_head = nn.Linear(hidden_dim, ROBOT_STATE_DIM)  # [B,K,H] -> [B,K,14]
        self.human_action_head = nn.Linear(hidden_dim, HUMAN_STATE_DIM)  # [B,K,H] -> [B,K,20]
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # CVAE encoder pieces (separate for each action dim)
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.robot_encoder_action_proj = nn.Linear(ROBOT_STATE_DIM, hidden_dim)
        self.human_encoder_action_proj = nn.Linear(HUMAN_STATE_DIM, hidden_dim)
        self.robot_encoder_state_proj = nn.Linear(ROBOT_STATE_DIM, hidden_dim)
        self.human_encoder_state_proj = nn.Linear(HUMAN_STATE_DIM, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        # ALOHA transformer expects exactly 2 additional tokens: latent + proprio
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def _encode_latent(self, state, actions, is_pad, embodiment: int):
        """
        CVAE prior encoder over [CLS, state, action_chunk].

        state:   [B, D]
        actions: [B, K, D]
        is_pad:  [B, K]
        returns: latent_input [B,H], mu [B,L], logvar [B,L]
        """
        bs = state.shape[0]
        if embodiment == EMBODIMENT_ROBOT:
            action_embed = self.robot_encoder_action_proj(actions)  # [B,K,H]
            state_embed = self.robot_encoder_state_proj(state).unsqueeze(1)  # [B,1,H]
        else:
            action_embed = self.human_encoder_action_proj(actions)
            state_embed = self.human_encoder_state_proj(state).unsqueeze(1)

        cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # [B,1,H]
        # encoder_input: [seq=1+1+K, B, H]
        encoder_input = torch.cat([cls_embed, state_embed, action_embed], dim=1).permute(1, 0, 2)
        cls_state_pad = torch.full((bs, 2), False, device=state.device)
        is_pad_full = torch.cat([cls_state_pad, is_pad], dim=1)  # [B, 2+K]
        pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)  # [2+K, 1, H]
        encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad_full)
        encoder_output = encoder_output[0]  # CLS token [B,H]
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)  # [B,H]
        return latent_input, mu, logvar

    def forward(
        self,
        state,
        image,
        embodiment,
        camera_mask=None,
        env_state=None,
        actions=None,
        is_pad=None,
    ):
        """
        Args:
          state:        [B, 14] robot or [B, 20] human (homogeneous batch)
          image:        [B, 4, 3, H, W]
          embodiment:   int or LongTensor[B] (all equal); 0=robot, 1=human
          camera_mask:  [B, 4] or [4] float {0,1}; optional
          actions:      [B, K, D] if training else None
          is_pad:       [B, K] if training else None

        Returns:
          a_hat:     [B, K, D]
          is_pad_hat:[B, K, 1]
          [mu, logvar]
        """
        is_training = actions is not None
        bs = state.shape[0]

        if isinstance(embodiment, torch.Tensor):
            emb_ids = embodiment.to(dtype=torch.long, device=state.device).reshape(-1)
            if emb_ids.numel() == 1:
                emb_ids = emb_ids.repeat(bs)
            if not torch.all(emb_ids == emb_ids[0]):
                raise ValueError("MixedDETRVAE expects a homogeneous embodiment per batch")
            embodiment_id = int(emb_ids[0].item())
        else:
            embodiment_id = int(embodiment)
            emb_ids = torch.full((bs,), embodiment_id, dtype=torch.long, device=state.device)

        if embodiment_id == EMBODIMENT_ROBOT and state.shape[-1] != ROBOT_STATE_DIM:
            raise ValueError(f"Robot state must be [B,{ROBOT_STATE_DIM}], got {tuple(state.shape)}")
        if embodiment_id == EMBODIMENT_HUMAN and state.shape[-1] != HUMAN_STATE_DIM:
            raise ValueError(f"Human state must be [B,{HUMAN_STATE_DIM}], got {tuple(state.shape)}")

        if is_training:
            latent_input, mu, logvar = self._encode_latent(state, actions, is_pad, embodiment_id)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=state.device)
            latent_input = self.latent_out_proj(latent_sample)

        # ---- vision: encode each camera slot ----
        # image[:, cam] is [B,3,H,W]
        if camera_mask is None:
            camera_mask = torch.ones(bs, NUM_CAMERAS, device=state.device, dtype=torch.float32)
        else:
            camera_mask = camera_mask.to(device=state.device, dtype=torch.float32)
            if camera_mask.ndim == 1:
                camera_mask = camera_mask.unsqueeze(0).expand(bs, -1)
            if camera_mask.shape != (bs, NUM_CAMERAS):
                raise ValueError(f"camera_mask must be [B,4], got {tuple(camera_mask.shape)}")

        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(NUM_CAMERAS):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # [B, C, h, w]
            pos = pos[0]  # [B, Cpos, h, w] or similar
            feat = self.input_proj(features)  # [B, H, h, w]
            # Zero masked cameras (human wrists / any disabled slot)
            m = camera_mask[:, cam_id].view(bs, 1, 1, 1)
            feat = feat * m
            pos = pos * m
            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        # Concatenate along width: [B, H, h, 4*w]
        src = torch.cat(all_cam_features, dim=3)
        pos = torch.cat(all_cam_pos, dim=3)

        # ---- proprio + embodiment ----
        if embodiment_id == EMBODIMENT_ROBOT:
            proprio_input = self.robot_state_proj(state)  # [B,H]
        else:
            proprio_input = self.human_state_proj(state)  # [B,H]
        proprio_input = proprio_input + self.embodiment_embed(emb_ids)  # domain token

        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
        )[0]  # [B, K, H]

        if embodiment_id == EMBODIMENT_ROBOT:
            a_hat = self.robot_action_head(hs)  # [B,K,14]
        else:
            a_hat = self.human_action_head(hs)  # [B,K,20]
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(args):
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)


def build(args):
    validate_camera_names(args.camera_names)
    # Four slot backbones: bird/front shared conceptually; wrists robot-only.
    backbones = [build_backbone(args) for _ in range(NUM_CAMERAS)]
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = MixedDETRVAE(
        backbones,
        transformer,
        encoder,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))
    return model


def kl_divergence(mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld
