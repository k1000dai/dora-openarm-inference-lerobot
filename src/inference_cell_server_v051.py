"""Cell-local policy server for OpenArm inference with LeRobot >= 0.5.1.

This is the lerobot 0.5.1 counterpart of ``inference_cell_server.py`` (which
targets lerobot 0.3.3 + an ACT policy). The wire protocol is identical — it
reads Arrow IPC observations over a Unix socket and replies with a JSON action
chunk — only the policy plumbing changes.

What is different in 0.5.x (the "preprocessor" rework):

* Normalization no longer lives inside the policy. It moved into an external
  ``preprocessor`` / ``postprocessor`` pipeline built by
  ``make_pre_post_processors`` and loaded from the model directory
  (``policy_preprocessor.json`` / ``policy_postprocessor.json``). The state /
  action QUANTILE stats are stored there, NOT in ``model.safetensors``.
* ``policy.predict_action_chunk`` now expects an already-preprocessed batch and
  returns a *normalized* chunk that must be run back through the postprocessor.
* The pi05 policy resizes images internally (resize-with-pad to its model
  resolution), so we no longer resize ourselves — we just hand it float images
  in [0, 1], channel-first.
* pi05 is language-conditioned: every observation must carry a ``task`` string
  that matches the instruction used during training (see ``TASK`` below).
"""

import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference

# Model directory. Resolved relative to the repo root so it works regardless of
# the server's working directory; override with the PRETRAINED_PATH env var
# (a local dir or a Hugging Face Hub repo id, e.g. k1000dai/pi05_policy_spill_merge).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "pi05_lerobot_spill_30k"
PRETRAINED_PATH = os.environ.get("PRETRAINED_PATH", str(DEFAULT_MODEL_PATH))

# Language instruction fed to pi05 on every step. This MUST match the task
# string the model was trained on or the policy will misbehave. Set it via the
# TASK env var, or replace the placeholder below.
TASK = os.environ.get("TASK", None)
if TASK is None:
    TASK = "<SET ME>"
    print("WARNING: TASK is not set. Set the TASK env var to the instruction the model ")

DEFAULT_SOCKET = "/dev/shm/policy-server.socket"

# Every camera the dora observer can publish, mapped to the model feature name.
# At startup this is filtered down to the cameras the loaded policy actually
# expects (the pi05 spill model uses only head_left + both wrists).
ARROW_TO_MODEL = {
    "camera_head_left": "observation.images.head_left",
    "camera_head_right": "observation.images.head_right",
    "camera_wrist_left": "observation.images.wrist_left",
    "camera_wrist_right": "observation.images.wrist_right",
    "camera_ceiling": "observation.images.ceiling",
}

# Filled at startup from policy.config.input_features.
CAMERA_KEY_MAP: dict[str, str] = {}

KNOWN_RESOLUTIONS = {
    600 * 960 * 3: (600, 960),
    720 * 1280 * 3: (720, 1280),
    480 * 640 * 3: (480, 640),
    1080 * 1920 * 3: (1080, 1920),
}

INTERVAL_NS = 33_333_333
CUTOFF_HZ = 15


def detect_resolution(n_bytes):
    if n_bytes in KNOWN_RESOLUTIONS:
        return KNOWN_RESOLUTIONS[n_bytes]
    n_pixels = n_bytes // 3
    for ratio_h, ratio_w in [(3, 4), (9, 16), (3, 5), (2, 3)]:
        h = int((n_pixels * ratio_h / ratio_w) ** 0.5)
        w = n_pixels // h
        if h * w == n_pixels:
            return (h, w)
    raise ValueError(f"Cannot determine resolution for {n_bytes} bytes")


def observation_to_raw(observation):
    """Build the raw (un-batched, numpy) observation dict.

    Images are kept as HWC uint8 and the state as float32 — exactly the format
    ``prepare_observation_for_inference`` expects. We deliberately do NOT resize
    the images: the pi05 policy resize-with-pads them to its own resolution.
    """
    raw = {}

    position = observation["position"].values.to_numpy().astype(np.float32)
    raw["observation.state"] = position

    for arrow_key, model_key in CAMERA_KEY_MAP.items():
        data = observation[arrow_key].values.to_numpy().astype(np.uint8)
        src_h, src_w = detect_resolution(len(data))
        raw[model_key] = data.reshape(src_h, src_w, 3)

    return raw


def infer(policy, preprocessor, postprocessor, observation, device):
    raw = observation_to_raw(observation)
    with torch.inference_mode():
        # Convert to channel-first float32 [0, 1] tensors with a batch dim, on
        # device, and attach the task string (the tokenizer step needs it).
        batch = prepare_observation_for_inference(raw, device, task=TASK)
        batch = preprocessor(batch)
        actions = policy.predict_action_chunk(batch)
        actions = postprocessor(actions)

    # .float() guards against bfloat16 outputs, which numpy cannot convert.
    positions = actions.squeeze(0).float().cpu().numpy().tolist()
    return {
        "interval": INTERVAL_NS,
        "cutoff_hz": CUTOFF_HZ,
        "positions": positions,
    }


def main():
    socket_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOCKET

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if TASK == "<SET ME>":
        raise ValueError(
            "TASK is unset. pi05 is language-conditioned: set the TASK env var "
            "(or edit the TASK constant) to the instruction the model was "
            "trained on before starting the server."
        )

    # The 0.5.x processor stats live next to the weights. Fail early with a clear
    # message if the model dir is missing them (only checked for local dirs).
    model_dir = Path(PRETRAINED_PATH)
    if model_dir.is_dir():
        missing = [
            f
            for f in ("policy_preprocessor.json", "policy_postprocessor.json")
            if not (model_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{model_dir} is missing {missing}. LeRobot 0.5.x stores the "
                "normalization stats in these processor files; copy them in "
                "alongside config.json / model.safetensors (they are saved next "
                "to the weights at training time)."
            )

    print(f"Loading policy from {PRETRAINED_PATH} on {device}...")
    policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=PRETRAINED_PATH)
    policy = get_policy_class(policy_config.type).from_pretrained(
        PRETRAINED_PATH, config=policy_config
    )
    policy.to(device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=PRETRAINED_PATH,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    CAMERA_KEY_MAP.update(
        {a: m for a, m in ARROW_TO_MODEL.items() if m in policy.config.input_features}
    )
    print(f"Policy loaded ({policy_config.type}). Cameras: {list(CAMERA_KEY_MAP)}")
    print(f"Task: {TASK!r}")

    if os.path.exists(socket_path):
        os.remove(socket_path)

    print(f"Listening on {socket_path}")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.bind(socket_path)
        sock.listen()
        try:
            with sock.accept()[0] as conn:
                print("Connected")
                with conn.makefile("rw") as io:
                    for line in io:
                        request = json.loads(line)
                        with pa.OSFile(request["data_path"], "rb") as f:
                            with pa.ipc.open_file(f) as reader:
                                obs = reader.get_batch(0).to_struct_array()[0]
                        actions = infer(policy, preprocessor, postprocessor, obs, device)
                        io.write(json.dumps(actions) + "\n")
                        io.flush()
        finally:
            if os.path.exists(socket_path):
                os.remove(socket_path)


if __name__ == "__main__":
    main()
