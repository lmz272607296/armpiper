import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dynamixel_interfaces.srv import SetDataToDxl

from interface.position_record import (
    DEFAULT_HAND_INITIAL_POSE_DEG,
    DEFAULT_HAND_POSITION_BIAS_DEG,
    DEFAULT_DXL_DATA_SERVICE_NAME,
    DEFAULT_DXL_DATA_SERVICE_TIMEOUT_SEC,
    DEFAULT_FALLBACK_DXL_DATA_SERVICE_NAMES,
    DEFAULT_SOFT_START_DURATION_SEC,
    DEFAULT_SOFT_START_RATE_HZ,
    INITIAL_POSITION_OFFSET_DEG,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = ROOT / "weights"
DEFAULT_BEST_CHECKPOINT = "seed_expert_best.pth"
DEFAULT_FINAL_CHECKPOINT = "seed_expert.pth"
DEFAULT_STATS_FILE = "normalizer_stats.npy"
DEFAULT_CONTROL_HZ = 10.0
DEFAULT_ROLLOUT_STEPS = 73
DEFAULT_ENSEMBLE_DECAY = 0.7
DEFAULT_MAX_DELTA_RAD = 0.025
DEFAULT_LOWPASS_ALPHA = 0.35
DEFAULT_INTERPOLATION_FACTOR = 2
DEFAULT_INFERENCE_BLOCKS_PER_BATCH = 5
STARTUP_FREE_DXL_IDS = tuple(range(8))
ALL_DXL_IDS = tuple(range(16))
STARTUP_HELD_DXL_IDS = tuple(range(8, 16))
ROS_NODE_NAME = "hand_inference_controller"
ROS_HAND_NODE_NAME = "hand_inference_hand_controller"


def _list_weight_run_dirs(weights_dir: Path) -> list[Path]:
    if not weights_dir.exists():
        return []
    return sorted((path for path in weights_dir.iterdir() if path.is_dir()), key=lambda path: path.name)


def resolve_latest_weight_dir(weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Optional[Path]:
    run_dirs = _list_weight_run_dirs(weights_dir)
    if not run_dirs:
        return None
    return run_dirs[-1]


def resolve_root_weight_files(weights_dir: Path = DEFAULT_WEIGHTS_DIR) -> Tuple[Optional[Path], Optional[Path]]:
    best_path = weights_dir / DEFAULT_BEST_CHECKPOINT
    final_path = weights_dir / DEFAULT_FINAL_CHECKPOINT
    stats_path = weights_dir / DEFAULT_STATS_FILE
    checkpoint = best_path if best_path.exists() else final_path if final_path.exists() else None
    stats = stats_path if stats_path.exists() else None
    return checkpoint, stats


def resolve_artifact_paths(
    checkpoint_path: Optional[str],
    normalizer_stats_path: Optional[str],
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
) -> Tuple[Path, Path]:
    explicit_checkpoint = checkpoint_path is not None
    checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    stats = Path(normalizer_stats_path).expanduser().resolve() if normalizer_stats_path else None

    if checkpoint is not None and checkpoint.is_dir():
        best_path = checkpoint / DEFAULT_BEST_CHECKPOINT
        final_path = checkpoint / DEFAULT_FINAL_CHECKPOINT
        checkpoint = best_path if best_path.exists() else final_path

    if checkpoint is not None and stats is None:
        sibling_stats = checkpoint.parent / DEFAULT_STATS_FILE
        if sibling_stats.exists():
            stats = sibling_stats

    root_checkpoint, root_stats = resolve_root_weight_files(weights_dir)
    if checkpoint is None and root_checkpoint is not None:
        checkpoint = root_checkpoint.resolve()
    if stats is None and not explicit_checkpoint and root_stats is not None:
        stats = root_stats.resolve()

    latest_dir = resolve_latest_weight_dir(weights_dir)
    if checkpoint is None and latest_dir is not None:
        best_path = latest_dir / DEFAULT_BEST_CHECKPOINT
        final_path = latest_dir / DEFAULT_FINAL_CHECKPOINT
        checkpoint = best_path if best_path.exists() else final_path

    if stats is None and not explicit_checkpoint and latest_dir is not None:
        stats = latest_dir / DEFAULT_STATS_FILE

    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Pass --checkpoint explicitly or place a checkpoint under weights/ or weights/<run_dir>/."
        )
    if stats is None or not stats.exists():
        raise FileNotFoundError(
            "normalizer_stats.npy not found. This model needs obs_mean/obs_std/act_min/act_max for inference. "
            "Pass --stats explicitly or place normalizer_stats.npy next to the checkpoint run directory."
        )

    return checkpoint, stats


def _count_transformer_layers(state_dict: dict, prefix: str) -> int:
    indices: set[int] = set()
    for key in state_dict:
        parts = key.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "layers" and idx > 0 and ".".join(parts[: idx + 1]) == prefix:
                try:
                    indices.add(int(parts[idx + 1]))
                except ValueError:
                    pass
                break
    return max(indices) + 1 if indices else 4


def infer_model_kwargs_from_checkpoint(ckpt: object) -> dict:
    ckpt_dict = ckpt if isinstance(ckpt, dict) else {}
    state_dict = ckpt_dict.get("model", ckpt_dict)
    if not isinstance(state_dict, dict):
        raise TypeError("Unsupported checkpoint format.")

    cfg_model = ckpt_dict.get("cfg", {}).get("model", {}) if isinstance(ckpt_dict.get("cfg", {}), dict) else {}
    inferred = {
        "obs_dim": state_dict["cvae_encoder.obs_proj.weight"].shape[1],
        "action_dim": state_dict["cvae_encoder.action_proj.weight"].shape[1],
        "chunk_size": state_dict["decoder.query_embed.weight"].shape[0],
        "latent_dim": state_dict["cvae_encoder.mu_proj.weight"].shape[0],
        "d_model": state_dict["cvae_encoder.action_proj.weight"].shape[0],
        "nhead": 8,
        "num_encoder_layers": _count_transformer_layers(state_dict, "cvae_encoder.transformer_encoder.layers"),
        "num_decoder_layers": _count_transformer_layers(state_dict, "decoder.transformer_decoder.layers"),
        "dropout": 0.1,
        "kl_weight": 10.0,
    }
    inferred.update(cfg_model)
    return inferred


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CVAEEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        latent_dim: int,
        chunk_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_proj = nn.Linear(action_dim, d_model)
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = SinusoidalPositionEncoding(d_model, max_len=chunk_size + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.mu_proj = nn.Linear(d_model, latent_dim)
        self.logvar_proj = nn.Linear(d_model, latent_dim)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = obs.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        act_emb = self.action_proj(actions)
        obs_emb = self.obs_proj(obs).unsqueeze(1)
        seq = torch.cat([cls + obs_emb, act_emb], dim=1)
        encoded = self.transformer_encoder(self.pos_enc(seq))
        cls_out = encoded[:, 0]
        return self.mu_proj(cls_out), self.logvar_proj(cls_out)


class ACTTransformerDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        chunk_size: int,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.query_embed = nn.Embedding(chunk_size, d_model)
        self.pos_enc_memory = SinusoidalPositionEncoding(d_model, max_len=16, dropout=dropout)
        self.pos_enc_query = SinusoidalPositionEncoding(d_model, max_len=chunk_size, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = obs.size(0)
        obs_token = self.obs_proj(obs).unsqueeze(1)
        z_token = self.z_proj(z).unsqueeze(1)
        memory = self.pos_enc_memory(torch.cat([obs_token, z_token], dim=1))
        indices = torch.arange(self.chunk_size, device=obs.device)
        queries = self.query_embed(indices).unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.transformer_decoder(self.pos_enc_query(queries), memory)
        return self.action_head(decoded)


class ACTPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        latent_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.register_buffer("obs_mean", torch.empty(0), persistent=False)
        self.register_buffer("obs_std", torch.empty(0), persistent=False)
        self.register_buffer("act_min", torch.empty(0), persistent=False)
        self.register_buffer("act_max", torch.empty(0), persistent=False)
        self.cvae_encoder = CVAEEncoder(
            action_dim=action_dim,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
        )
        self.decoder = ACTTransformerDecoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
        )

    def set_normalizer_stats(
        self,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_min: np.ndarray,
        act_max: np.ndarray,
    ) -> None:
        self.obs_mean = self.obs_mean.new_tensor(obs_mean, dtype=torch.float32)
        self.obs_std = self.obs_std.new_tensor(obs_std, dtype=torch.float32)
        self.act_min = self.act_min.new_tensor(act_min, dtype=torch.float32)
        self.act_max = self.act_max.new_tensor(act_max, dtype=torch.float32)

    def normalize_obs(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.obs_mean.numel() == 0 or self.obs_std.numel() == 0:
            raise RuntimeError("Observation normalization stats are missing.")
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        return (obs_tensor - self.obs_mean.to(obs_tensor.device)) / self.obs_std.to(obs_tensor.device)

    def denormalize_actions(self, actions: torch.Tensor | np.ndarray) -> torch.Tensor:
        if self.act_min.numel() == 0 or self.act_max.numel() == 0:
            raise RuntimeError("Action de-normalization stats are missing.")
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        act_min = self.act_min.to(action_tensor.device)
        act_max = self.act_max.to(action_tensor.device)
        return 0.5 * (action_tensor + 1.0) * (act_max - act_min) + act_min

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.zeros(obs.size(0), self.latent_dim, device=obs.device)
        return self.decoder(obs, z)

    @torch.no_grad()
    def predict_denormalized(self, raw_obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        self.eval()
        obs = self.normalize_obs(raw_obs)
        pred_actions = self.forward(obs)
        return self.denormalize_actions(pred_actions)


def load_policy(checkpoint_path: Path, stats_path: Path, device: str) -> ACTPolicy:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = infer_model_kwargs_from_checkpoint(ckpt)
    model = ACTPolicy(**model_kwargs).to(device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    stats = np.load(stats_path, allow_pickle=True).item()
    model.set_normalizer_stats(
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        act_min=stats["act_min"],
        act_max=stats["act_max"],
    )
    model.eval()
    return model


def temporal_ensemble(
    predicted_chunks: list[np.ndarray],
    chunk_start_indices: list[int],
    target_index: int,
    decay: float,
) -> Tuple[np.ndarray, int]:
    candidates: list[np.ndarray] = []
    weights: list[float] = []

    for chunk, start_idx in zip(reversed(predicted_chunks), reversed(chunk_start_indices)):
        offset = target_index - start_idx
        if 0 <= offset < len(chunk):
            candidates.append(chunk[offset])
            weights.append(decay ** offset)

    if not candidates:
        raise RuntimeError(f"No predicted chunk covers target index {target_index}.")

    stacked = np.stack(candidates, axis=0)
    weight_array = np.asarray(weights, dtype=np.float32)
    weight_array /= np.sum(weight_array)
    return np.sum(stacked * weight_array[:, None], axis=0), len(candidates)


def rollout_policy(
    model: ACTPolicy,
    initial_pose: np.ndarray,
    rollout_steps: int,
    ensemble_decay: float,
    lowpass_alpha: float,
    max_delta_rad: float,
) -> dict:
    current_pose = initial_pose.astype(np.float32).copy()
    action_lower = model.act_min.detach().cpu().numpy().astype(np.float32)
    action_upper = model.act_max.detach().cpu().numpy().astype(np.float32)

    trajectory = [current_pose.copy()]
    predicted_chunks: list[np.ndarray] = []
    chunk_start_indices: list[int] = []
    overlap_counts: list[int] = []
    raw_ensemble_targets: list[np.ndarray] = []
    applied_deltas: list[np.ndarray] = []
    first_chunk: Optional[np.ndarray] = None

    device = str(model.obs_mean.device)
    for step_idx in range(rollout_steps):
        obs_tensor = torch.from_numpy(current_pose).unsqueeze(0).to(device)
        pred_chunk = model.predict_denormalized(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)
        if first_chunk is None:
            first_chunk = pred_chunk.copy()

        predicted_chunks.append(pred_chunk)
        chunk_start_indices.append(step_idx + 1)

        target_pose, overlap_count = temporal_ensemble(
            predicted_chunks=predicted_chunks,
            chunk_start_indices=chunk_start_indices,
            target_index=step_idx + 1,
            decay=ensemble_decay,
        )

        delta = lowpass_alpha * (target_pose - current_pose)
        delta = np.clip(delta, -max_delta_rad, max_delta_rad)
        next_pose = np.clip(current_pose + delta, action_lower, action_upper)

        overlap_counts.append(overlap_count)
        raw_ensemble_targets.append(target_pose.astype(np.float32))
        applied_deltas.append((next_pose - current_pose).astype(np.float32))
        trajectory.append(next_pose.astype(np.float32))
        current_pose = next_pose.astype(np.float32)

    if first_chunk is None:
        obs_tensor = torch.from_numpy(current_pose).unsqueeze(0).to(device)
        first_chunk = model.predict_denormalized(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)

    trajectory_array = np.stack(trajectory, axis=0)
    delta_array = np.stack(applied_deltas, axis=0) if applied_deltas else np.zeros((0, initial_pose.shape[0]), dtype=np.float32)

    return {
        "first_chunk": first_chunk,
        "trajectory": trajectory_array,
        "raw_ensemble_targets": np.stack(raw_ensemble_targets, axis=0) if raw_ensemble_targets else np.zeros((0, initial_pose.shape[0]), dtype=np.float32),
        "applied_deltas": delta_array,
        "overlap_counts": overlap_counts,
        "max_abs_applied_delta_rad": float(np.max(np.abs(delta_array))) if delta_array.size else 0.0,
        "mean_l2_applied_delta_rad": float(np.mean(np.linalg.norm(delta_array, axis=-1))) if delta_array.size else 0.0,
    }


def build_short_horizon_trajectory(
    model: ACTPolicy,
    initial_pose: np.ndarray,
    horizon_steps: int,
    ensemble_decay: float,
    lowpass_alpha: float,
    max_delta_rad: float,
) -> np.ndarray:
    current_pose = np.asarray(initial_pose, dtype=np.float32).reshape(-1).copy()
    action_lower = model.act_min.detach().cpu().numpy().astype(np.float32)
    action_upper = model.act_max.detach().cpu().numpy().astype(np.float32)
    if current_pose.shape != action_lower.shape:
        raise ValueError(f"Expected pose shape {action_lower.shape}, got {current_pose.shape}.")

    predicted_chunks: list[np.ndarray] = []
    chunk_start_indices: list[int] = []
    trajectory: list[np.ndarray] = []
    device = str(model.obs_mean.device)

    for step_idx in range(horizon_steps):
        obs_tensor = torch.from_numpy(current_pose).unsqueeze(0).to(device)
        pred_chunk = model.predict_denormalized(obs_tensor)[0].detach().cpu().numpy().astype(np.float32)
        predicted_chunks.append(pred_chunk)
        chunk_start_indices.append(step_idx + 1)

        target_pose, _ = temporal_ensemble(
            predicted_chunks=predicted_chunks,
            chunk_start_indices=chunk_start_indices,
            target_index=step_idx + 1,
            decay=ensemble_decay,
        )
        delta = lowpass_alpha * (target_pose - current_pose)
        delta = np.clip(delta, -max_delta_rad, max_delta_rad)
        current_pose = np.clip(current_pose + delta, action_lower, action_upper).astype(np.float32)
        trajectory.append(current_pose.copy())

    return np.stack(trajectory, axis=0) if trajectory else np.zeros((0, current_pose.shape[0]), dtype=np.float32)


def build_startup_pose() -> np.ndarray:
    initial_pose_deg = np.asarray(DEFAULT_HAND_INITIAL_POSE_DEG, dtype=np.float64)
    position_bias_deg = np.asarray(DEFAULT_HAND_POSITION_BIAS_DEG, dtype=np.float64)
    return np.deg2rad(initial_pose_deg + position_bias_deg - INITIAL_POSITION_OFFSET_DEG)


def interpolate_trajectory(trajectory: np.ndarray, interpolation_factor: int) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)
    if interpolation_factor <= 1 or trajectory.shape[0] <= 1:
        return trajectory.copy()

    point_count = (trajectory.shape[0] - 1) * interpolation_factor + 1
    interpolated = np.empty((point_count, trajectory.shape[1]), dtype=np.float64)
    alpha = np.linspace(0.0, 1.0, interpolation_factor + 1, dtype=np.float64)[:-1]
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    write_index = 0
    for start_sample, end_sample in zip(trajectory[:-1], trajectory[1:]):
        segment = start_sample + (end_sample - start_sample) * alpha[:, np.newaxis]
        interpolated[write_index:write_index + interpolation_factor] = segment
        write_index += interpolation_factor

    interpolated[write_index] = trajectory[-1]
    return interpolated


def get_model_input_pose(hand_controller) -> np.ndarray:
    real_positions = getattr(hand_controller, "real_raw_positions", None)
    if real_positions is None:
        real_positions = getattr(hand_controller, "raw_positions", None)
    if real_positions is None:
        raise RuntimeError("Hand joint state is not available.")
    return np.asarray(real_positions, dtype=np.float32).reshape(-1)


def real_trajectory_to_controller_order(trajectory: np.ndarray, hand_controller) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        return np.asarray(hand_controller.real_to_sim(trajectory), dtype=np.float64)
    return np.stack(
        [np.asarray(hand_controller.real_to_sim(sample), dtype=np.float64) for sample in trajectory],
        axis=0,
    )


def format_prediction_log(trajectory: np.ndarray, initial_pose: Optional[np.ndarray] = None) -> str:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)
    deg_plus_180 = np.rad2deg(trajectory) + 180.0
    lines = ["predicted joint0-joint15 deg+180:"]
    if initial_pose is not None:
        initial_pose = np.asarray(initial_pose, dtype=np.float64).reshape(-1)
        values = ", ".join(f"{value:.2f}" for value in (np.rad2deg(initial_pose) + 180.0))
        lines.append(f"initial: [{values}]")
    for index, sample in enumerate(deg_plus_180):
        values = ", ".join(f"{value:.2f}" for value in sample)
        lines.append(f"step{index}: [{values}]")
    return "\n".join(lines)


class HandInferenceController:
    def __init__(self, node, hand_controller, model: ACTPolicy):
        self.node = node
        self.hand_controller = hand_controller
        self.model = model
        self.control_hz = DEFAULT_CONTROL_HZ
        self.interpolation_factor = DEFAULT_INTERPOLATION_FACTOR
        self.seconds_per_point = 1.0 / (self.control_hz * self.interpolation_factor)
        self.horizon_steps = DEFAULT_ROLLOUT_STEPS
        self.ensemble_decay = DEFAULT_ENSEMBLE_DECAY
        self.lowpass_alpha = DEFAULT_LOWPASS_ALPHA
        self.max_delta_rad = DEFAULT_MAX_DELTA_RAD
        self.expected_obs_dim = int(model.obs_mean.numel())
        self.waiting_for_state_logged = False
        self.soft_start_target = build_startup_pose().astype(np.float32)
        self.soft_start_duration_sec = DEFAULT_SOFT_START_DURATION_SEC
        self.soft_start_rate_hz = DEFAULT_SOFT_START_RATE_HZ
        self.soft_start_active = False
        self.soft_start_complete = False
        self.soft_start_finish_time = 0.0
        self.command_execution_finish_time = 0.0
        self.inference_blocks_remaining = DEFAULT_INFERENCE_BLOCKS_PER_BATCH
        self.waiting_for_keyboard_start = False
        self.startup_torque_prepared = False
        self.startup_torque_restored = False
        self.startup_waiting_for_space = False
        self.startup_torque_in_progress = False
        self.startup_torque_thread = None
        self.dxl_data_service_timeout_sec = DEFAULT_DXL_DATA_SERVICE_TIMEOUT_SEC
        self.dxl_data_service_names = self.build_dxl_data_service_names(DEFAULT_DXL_DATA_SERVICE_NAME)
        self.dxl_data_clients = {
            service_name: node.create_client(SetDataToDxl, service_name)
            for service_name in self.dxl_data_service_names
        }
        self.keyboard_stop_event = threading.Event()
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.keyboard_thread.start()
        self.timer = node.create_timer(1.0 / self.control_hz, self.publish_inference_trajectory)

    def now(self) -> float:
        return time.monotonic()

    def command_is_active(self) -> bool:
        return self.now() < self.command_execution_finish_time

    def mark_command_active(self, trajectory: np.ndarray, seconds_per_point: float) -> None:
        trajectory = np.asarray(trajectory)
        point_count = trajectory.shape[0] if trajectory.ndim > 1 else 1
        self.command_execution_finish_time = self.now() + (point_count * float(seconds_per_point))

    def ensure_batch_state(self) -> None:
        if not hasattr(self, "inference_blocks_remaining"):
            self.inference_blocks_remaining = DEFAULT_INFERENCE_BLOCKS_PER_BATCH
        if not hasattr(self, "waiting_for_keyboard_start"):
            self.waiting_for_keyboard_start = False
        if not hasattr(self, "startup_torque_prepared"):
            self.startup_torque_prepared = True
        if not hasattr(self, "startup_torque_restored"):
            self.startup_torque_restored = True
        if not hasattr(self, "startup_waiting_for_space"):
            self.startup_waiting_for_space = False
        if not hasattr(self, "startup_torque_in_progress"):
            self.startup_torque_in_progress = False

    def build_dxl_data_service_names(self, preferred_name: str) -> list[str]:
        service_names: list[str] = []
        for service_name in (preferred_name, *DEFAULT_FALLBACK_DXL_DATA_SERVICE_NAMES):
            normalized_name = str(service_name).strip()
            if normalized_name and normalized_name not in service_names:
                service_names.append(normalized_name)
        return service_names

    def resolve_dxl_data_client(self):
        deadline = self.now() + self.dxl_data_service_timeout_sec
        while self.now() < deadline:
            for service_name in self.dxl_data_service_names:
                client = self.dxl_data_clients[service_name]
                if client.wait_for_service(timeout_sec=0.2):
                    return service_name, client
            time.sleep(0.05)
        return None

    def call_dxl_data_service(self, client, request):
        future = client.call_async(request)
        deadline = self.now() + self.dxl_data_service_timeout_sec
        while self.now() < deadline:
            if future.done():
                return future.result()
            time.sleep(0.05)
        return future.result() if future.done() else None

    def set_joint_torque_state(self, dxl_ids, enable: bool) -> bool:
        resolved = self.resolve_dxl_data_client()
        if resolved is None:
            self.node.get_logger().error(
                f"No set_dxl_data service became available within {self.dxl_data_service_timeout_sec:.1f}s. "
                f"Tried: {self.dxl_data_service_names}"
            )
            return False

        service_name, client = resolved
        target_value = 1 if enable else 0
        action = "enable" if enable else "disable"
        success = True
        for dxl_id in dxl_ids:
            request = SetDataToDxl.Request()
            request.id = int(dxl_id)
            request.item_name = "Torque Enable"
            request.item_data = target_value
            response = self.call_dxl_data_service(client, request)
            if response is None or not response.result:
                self.node.get_logger().error(f"Failed to {action} torque for dxl{dxl_id} through {service_name}.")
                success = False
                continue

        target_group = [f"dxl{dxl_id}" for dxl_id in dxl_ids]
        self.node.get_logger().info(f"{action.capitalize()}d torque command sent for {target_group} through {service_name}.")
        return success

    def prepare_startup_torque_state(self) -> bool:
        if self.startup_torque_prepared or self.startup_torque_in_progress:
            return True
        self.startup_waiting_for_space = True
        self.startup_torque_in_progress = True
        self.startup_torque_thread = threading.Thread(target=self.run_startup_torque_worker, daemon=True)
        self.startup_torque_thread.start()
        self.node.get_logger().info("Disabling torque for joint0-joint7 once. Wait for the prompt, adjust them, then press SPACE.")
        return True

    def run_startup_torque_worker(self) -> bool:
        if self.startup_torque_prepared:
            return True
        self.startup_torque_prepared = True
        torque_disabled = self.set_joint_torque_state(STARTUP_FREE_DXL_IDS, False)
        if not torque_disabled:
            self.node.get_logger().warn(
                "Tried to disable torque for joint0-joint7 once, but the service reported failure. "
                "Not retrying automatically; press SPACE to continue after checking the hand state."
            )
        self.startup_waiting_for_space = True
        self.startup_torque_in_progress = False
        self.node.get_logger().info("Disabled torque for joint0-joint7. Adjust them, then press SPACE to enable all torque and start inference.")
        return torque_disabled

    def restore_all_torque(self) -> bool:
        if not self.set_joint_torque_state(ALL_DXL_IDS, True):
            return False
        self.startup_torque_restored = True
        self.startup_waiting_for_space = False
        self.node.get_logger().info("Enabled torque for joint0-joint15.")
        return True

    def publish_inference_trajectory(self) -> None:
        self.ensure_batch_state()
        if self.command_is_active():
            return

        if self.inference_blocks_remaining <= 0:
            self.prompt_for_next_batch()
            return

        if getattr(self.hand_controller, "real_raw_positions", None) is None:
            if not self.waiting_for_state_logged:
                self.node.get_logger().info("Waiting for hand joint state before running inference.")
                self.waiting_for_state_logged = True
            return

        current_pose = get_model_input_pose(self.hand_controller)
        if current_pose.shape != (self.expected_obs_dim,):
            self.node.get_logger().error(
                f"Expected {self.expected_obs_dim}-D hand position state, got shape {current_pose.shape}."
            )
            return

        if not self.soft_start_complete:
            self.publish_soft_start_if_needed(current_pose)
            return

        if not self.startup_torque_prepared:
            self.prepare_startup_torque_state()
            return
        if self.startup_waiting_for_space or not self.startup_torque_restored:
            return

        try:
            trajectory = build_short_horizon_trajectory(
                model=self.model,
                initial_pose=current_pose,
                horizon_steps=self.horizon_steps,
                ensemble_decay=self.ensemble_decay,
                lowpass_alpha=self.lowpass_alpha,
                max_delta_rad=self.max_delta_rad,
            )
            self.node.get_logger().info(format_prediction_log(trajectory, initial_pose=current_pose))
            trajectory = interpolate_trajectory(trajectory, self.interpolation_factor)
            controller_trajectory = real_trajectory_to_controller_order(trajectory, self.hand_controller)
            self.hand_controller.command_joint_position(controller_trajectory, self.seconds_per_point)
            self.mark_command_active(controller_trajectory, self.seconds_per_point)
            self.inference_blocks_remaining -= 1
        except Exception as exc:
            self.node.get_logger().error(f"Hand inference publish failed: {repr(exc)}")

    def prompt_for_next_batch(self) -> None:
        if self.waiting_for_keyboard_start:
            return
        self.waiting_for_keyboard_start = True
        self.node.get_logger().info("Five inference blocks complete. Press SPACE to run next inference batch from current hand pose, or q to quit.")

    def handle_keyboard_key(self, key: str) -> None:
        if key == " ":
            if self.command_is_active():
                self.node.get_logger().info("Current command is still executing; wait for it to finish before starting the next batch.")
                return
            self.ensure_batch_state()
            if self.startup_torque_in_progress:
                self.node.get_logger().info("Torque disable is still in progress; wait for it to finish before starting inference.")
                return
            if self.startup_waiting_for_space and not self.startup_torque_restored:
                if not self.restore_all_torque():
                    return
            self.inference_blocks_remaining = DEFAULT_INFERENCE_BLOCKS_PER_BATCH
            self.waiting_for_keyboard_start = False
            self.node.get_logger().info("Starting next inference batch from current hand pose.")
            return
        if key.lower() == "q" or key in ("\x03", "\x04"):
            if getattr(self, "startup_torque_prepared", False) and not getattr(self, "startup_torque_restored", True):
                self.restore_all_torque()
            self.keyboard_stop_event.set()
            self.node.get_logger().info("Exit requested from keyboard.")
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def keyboard_loop(self) -> None:
        if not sys.stdin.isatty():
            self.node.get_logger().warn("Standard input is not a TTY; keyboard control is unavailable.")
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.keyboard_stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    key = sys.stdin.read(1)
                    if key:
                        self.handle_keyboard_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def publish_soft_start_if_needed(self, current_pose: np.ndarray) -> None:
        if self.soft_start_active:
            if time.monotonic() >= self.soft_start_finish_time:
                self.soft_start_active = False
                self.soft_start_complete = True
                self.node.get_logger().info("Hand soft-start complete; beginning policy inference.")
            return

        delta = np.abs(self.soft_start_target - current_pose)
        if np.all(delta < 1e-4):
            self.soft_start_complete = True
            self.node.get_logger().info("Hand already matches startup pose; beginning policy inference.")
            return

        steps = max(2, int(np.ceil(self.soft_start_duration_sec * self.soft_start_rate_hz)))
        alpha = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        trajectory = current_pose + (self.soft_start_target - current_pose) * alpha[:, np.newaxis]
        seconds_per_point = self.soft_start_duration_sec / steps
        controller_trajectory = real_trajectory_to_controller_order(trajectory, self.hand_controller)
        self.hand_controller.command_joint_position(controller_trajectory, seconds_per_point)
        self.mark_command_active(controller_trajectory, seconds_per_point)
        self.soft_start_active = True
        self.soft_start_finish_time = time.monotonic() + self.soft_start_duration_sec
        self.node.get_logger().info(
            f"Published hand soft-start over {self.soft_start_duration_sec:.2f}s with {steps} points."
        )


def main_ros() -> None:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from interface.hand_controller import LeapHand

    rclpy.init()
    hand_controller = None
    node = None
    executor = None
    try:
        checkpoint_path, stats_path = resolve_artifact_paths(None, None)
        model = load_policy(checkpoint_path, stats_path, "cpu")
        hand_controller = LeapHand(ROS_HAND_NODE_NAME)
        node = Node(ROS_NODE_NAME)
        HandInferenceController(node=node, hand_controller=hand_controller, model=model)
        node.get_logger().info(
            f"Loaded hand inference policy from {checkpoint_path}; publishing {DEFAULT_ROLLOUT_STEPS} points at "
            f"{DEFAULT_CONTROL_HZ:.1f} Hz with interpolation factor {DEFAULT_INTERPOLATION_FACTOR}."
        )
        executor = MultiThreadedExecutor()
        executor.add_node(hand_controller)
        executor.add_node(node)
        executor.spin()
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if hand_controller is not None:
            hand_controller.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-file ACT inference for the current chopstick BC model. "
            "It supports one-step prediction and CPU rollout from an initial 16-D hand joint pose."
        )
    )
    parser.add_argument(
        "--initial-pose",
        "--obs",
        dest="obs",
        type=float,
        nargs="+",
        required=True,
        help="Raw 16-D hand joint positions. Do not pass joint torques here for the current checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth checkpoint. Defaults to the latest run.")
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to normalizer_stats.npy. Defaults to the latest run.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--control-hz", type=float, default=DEFAULT_CONTROL_HZ, help="Output control frequency for the generated trajectory.")
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS, help="How many rollout steps to generate from the initial pose.")
    parser.add_argument("--ensemble-decay", type=float, default=DEFAULT_ENSEMBLE_DECAY, help="Decay for overlapping chunk temporal ensembling. 1.0 is uniform average; smaller values trust newer chunks more.")
    parser.add_argument("--lowpass-alpha", type=float, default=1.0, help="Step gain applied to the ensembled next target. Use values below 1.0 only if you need extra smoothing.")
    parser.add_argument("--max-delta-rad", type=float, default=DEFAULT_MAX_DELTA_RAD, help="Per-step per-joint delta clamp in radians for smoother rollout.")
    parser.add_argument("--full-chunk", action="store_true", help="Also output the full first predicted chunk from the initial pose.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs = np.asarray(args.obs, dtype=np.float32)

    requested_device = args.device
    device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
    checkpoint_path, stats_path = resolve_artifact_paths(args.checkpoint, args.stats)
    model = load_policy(checkpoint_path, stats_path, device)

    expected_obs_dim = int(model.obs_mean.numel())
    if obs.shape != (expected_obs_dim,):
        raise ValueError(
            f"Expected a {expected_obs_dim}-D hand joint-position observation, got shape {obs.shape}. "
            "The current published checkpoint in this directory was trained on positions, not torques."
        )

    if args.rollout_steps < 0:
        raise ValueError("--rollout-steps must be >= 0.")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be > 0.")
    if not (0.0 < args.ensemble_decay <= 1.0):
        raise ValueError("--ensemble-decay must be in (0, 1].")
    if not (0.0 < args.lowpass_alpha <= 1.0):
        raise ValueError("--lowpass-alpha must be in (0, 1].")
    if args.max_delta_rad <= 0:
        raise ValueError("--max-delta-rad must be > 0.")

    rollout = rollout_policy(
        model=model,
        initial_pose=obs,
        rollout_steps=args.rollout_steps,
        ensemble_decay=args.ensemble_decay,
        lowpass_alpha=args.lowpass_alpha,
        max_delta_rad=args.max_delta_rad,
    )

    pred_chunk = rollout["first_chunk"]
    trajectory = rollout["trajectory"]
    timestamps = (np.arange(len(trajectory), dtype=np.float32) / np.float32(args.control_hz)).astype(np.float32)

    payload = {
        "model_input_semantics": f"raw {expected_obs_dim}-dim hand joint positions",
        "model_output_semantics": f"predicted future {int(model.act_min.numel())}-dim hand joint position targets",
        "model_input_unit": "radians",
        "model_output_unit": "radians",
        "obs_dim": expected_obs_dim,
        "action_dim": int(model.act_min.numel()),
        "chunk_size": int(pred_chunk.shape[0]),
        "input_obs": obs.tolist(),
        "predicted_first_action": pred_chunk[0].tolist(),
        "rollout_first_action": trajectory[1].tolist() if len(trajectory) > 1 else pred_chunk[0].tolist(),
        "control_hz": float(args.control_hz),
        "dt_s": float(1.0 / args.control_hz),
        "rollout_steps": int(args.rollout_steps),
        "rollout_duration_s": float(args.rollout_steps / args.control_hz),
        "ensemble_decay": float(args.ensemble_decay),
        "lowpass_alpha": float(args.lowpass_alpha),
        "max_delta_rad": float(args.max_delta_rad),
        "trajectory_timestamps_s": timestamps.tolist(),
        "rollout_joint_positions": trajectory.tolist(),
        "overlap_counts": rollout["overlap_counts"],
        "max_abs_applied_delta_rad": rollout["max_abs_applied_delta_rad"],
        "mean_l2_applied_delta_rad": rollout["mean_l2_applied_delta_rad"],
        "checkpoint": str(checkpoint_path),
        "stats": str(stats_path),
        "device": device,
    }
    if args.full_chunk:
        payload["predicted_chunk"] = pred_chunk.tolist()

    output_text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    print(output_text)


if __name__ == "__main__":
    main()
