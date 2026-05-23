import json
import os
import socket
import sys

import numpy as np
import pyarrow as pa
import torch
from PIL import Image

from lerobot.policies.act.modeling_act import ACTPolicy

PRETRAINED_PATH = "k1000dai/act_openarm_pick_cube_40k"
DEFAULT_SOCKET = "/dev/shm/policy-server.socket"

CAMERA_KEY_MAP = {
    "camera_head_left": "observation.images.head_left",
    "camera_wrist_left": "observation.images.wrist_left",
    "camera_wrist_right": "observation.images.wrist_right",
}

# Filled at startup from policy.config.input_features. The dataset's stored
# video resolution is not necessarily the model's input resolution — ACT's
# transformer positional encoding is tied to the feature-map size produced by
# the backbone, so the image must be resized to the shape the policy expects.
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

        print(img.shape, img.dtype, img.is_contiguous(), img.stride())
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
    socket_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOCKET
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading policy from {PRETRAINED_PATH} on {device}...")
    policy = ACTPolicy.from_pretrained(PRETRAINED_PATH)
    if device != policy.config.device:
        policy.to(device)
    policy.reset()

    for model_key in CAMERA_KEY_MAP.values():
        feature = policy.config.input_features[model_key]
        c, h, w = feature.shape
        IMAGE_SIZES[model_key] = (h, w)
    print(f"Policy loaded. Expected image sizes: {IMAGE_SIZES}")

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
                        actions = infer(policy, obs, device)
                        io.write(json.dumps(actions) + "\n")
                        io.flush()
        finally:
            if os.path.exists(socket_path):
                os.remove(socket_path)


if __name__ == "__main__":
    main()
