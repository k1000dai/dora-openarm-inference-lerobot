# dora-openarm-inference-lerobot

Real-time bimanual robot control using a pre-trained ACT (Action Chunking with Transformers) policy from [LeRobot](https://github.com/huggingface/lerobot), orchestrated via [dora-rs](https://github.com/dora-rs/dora) dataflow and simulated in [MuJoCo](https://mujoco.org/).

## Overview

This system runs inference on an [OpenArm](https://openarm.dev/) bimanual robot (dual 7-DOF arms + grippers) using a transformer-based policy trained with LeRobot. The pipeline accepts camera observations and arm state, infers action chunks, and executes them on the robot or its MuJoCo simulation.

## Dataset and model


### Key Components

| Component | Description |
|---|---|
| **inference_server** (`src/inference_server.py`) | Loads ACT policy (`k1000dai/act_openarm_pick_cube_40k`) and serves inference via Unix domain socket |
| **dora-openarm-observer** | Aggregates arm state + camera images into Arrow IPC |
| **dora-openarm-local-policy-server** | Bridges dora node to the external inference server |
| **dora-openarm-actions-executor** | Upsamples action chunks (Hermite spline) and applies low-pass filter (biquad Butterworth, 15 Hz cutoff) |
| **dora-openarm-mujoco** | MuJoCo physics simulation for the OpenArm robot |

### Rate Control

- **30 Hz** observation sampling
- **250 Hz** motor control output
- Action chunks are upsampled with cubic Hermite spline interpolation and smoothed with a second-order Butterworth low-pass filter

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- CUDA-capable GPU (falls back to CPU/MPS)

Two separate virtual environments are used due to CUDA version requirements:

| Environment | Purpose |
|---|---|
| `.venv` | dora dataflow orchestration |
| `.venv_server` | Policy inference server | 

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/k1000dai/dora-openarm-inference-lerobot.git
cd dora-openarm-inference-lerobot

# Install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install dora-rs-cli
deactivate
uv venv .venv_server 
source .venv_server/bin/activate
uv pip install lerobot==0.3.3 pyarrow Pillow
uv pip install torch torchvision --torch-backend auto --upgrade
deactivate
```

## Usage

### policy inference server
```bash
source .venv_server/bin/activate
python src/inference_server.py /dev/shm/policy-server.socket
```

The inference server runs as a separate process, communicating via a Unix domain socket at `/dev/shm/policy-server.socket`.

### dora dataflow
```bash
source .venv/bin/activate
dora build dataflow-inference.yaml --uv
SOCKET=/dev/shm/policy-server.socket dora run dataflow-inference.yaml --uv
```

## Project Structure

```
├── dataflow-inference.yaml        # Dora dataflow graph definition
├── src/
│   └── inference_server.py        # ACT policy inference server
├── nodes/                         # Dora nodes (git submodules)
│   ├── dora-openarm-observer/
│   ├── dora-openarm-local-policy-server/
│   ├── dora-openarm-actions-executor/
│   ├── dora-openarm-mujoco/
│   └── dora-openarm-quitter/
├── pyproject.toml
└── main.py
```
