"""Docker-based policy server for OpenArm inference with LeRobot."""

import json
import socket
import sys

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from lerobot.policies.factory import get_policy_class
from lerobot.policies.pretrained import PreTrainedConfig

PRETRAINED_PATH = "k1000dai/act_openarm_pick_cube_40k"

CAMERA_KEY_MAP = {
    "camera_head_left": "observation.images.head_left",
    "camera_wrist_left": "observation.images.wrist_left",
    "camera_wrist_right": "observation.images.wrist_right",
}

IMAGE_SIZES: dict[str, tuple[int, int]] = {}

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


def prepare_image(raw_data, target_h, target_w):
    data = raw_data.values.to_numpy().astype(np.uint8)
    src_h, src_w = detect_resolution(len(data))
    img = data.reshape(src_h, src_w, 3)
    if src_h != target_h or src_w != target_w:
        img = np.array(Image.fromarray(img).resize((target_w, target_h)))
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def observation_to_batch(observation, device):
    batch = {}

    position = observation["position"].values.to_numpy().astype(np.float32)
    batch["observation.state"] = torch.from_numpy(position).unsqueeze(0).to(device)

    for arrow_key, model_key in CAMERA_KEY_MAP.items():
        target_h, target_w = IMAGE_SIZES[model_key]
        img = prepare_image(observation[arrow_key], target_h, target_w)
        batch[model_key] = img.unsqueeze(0).to(device)

    return batch


def infer(policy, observation, device):
    batch = observation_to_batch(observation, device)
    actions = policy.predict_action_chunk(batch)
    positions = actions.squeeze(0).cpu().numpy().tolist()
    return {
        "interval": INTERVAL_NS,
        "cutoff_hz": CUTOFF_HZ,
        "positions": positions,
    }


def main():
    socket_path = sys.argv[1]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading policy from {PRETRAINED_PATH} on {device}...")
    policy_config = PreTrainedConfig.from_pretrained(PRETRAINED_PATH)
    policy_config.pretrained_path = PRETRAINED_PATH
    policy = get_policy_class(policy_config.type).from_pretrained(
        config=policy_config,
        pretrained_name_or_path=policy_config.pretrained_path,
    )

    if device != policy.config.device:
        policy.to(device)
    policy.reset()

    for model_key in CAMERA_KEY_MAP.values():
        feature = policy.config.input_features[model_key]
        c, h, w = feature.shape
        IMAGE_SIZES[model_key] = (h, w)
    print(f"Policy loaded. Expected image sizes: {IMAGE_SIZES}")

    print(f"Connecting to {socket_path}...")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(socket_path)
        with sock.makefile("rw") as io:
            for line in io:
                request = json.loads(line)
                with pa.OSFile(request["data_path"], "rb") as f:
                    with pa.ipc.open_file(f) as reader:
                        obs = reader.get_batch(0).to_struct_array()[0]
                actions = infer(policy, obs, device)
                io.write(json.dumps(actions) + "\n")
                io.flush()


if __name__ == "__main__":
    main()
