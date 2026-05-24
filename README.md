# dora-openarm-inference-lerobot

Real-time bimanual robot control using a pre-trained ACT (Action Chunking with Transformers) policy from [LeRobot](https://github.com/huggingface/lerobot), orchestrated via [dora-rs](https://github.com/dora-rs/dora) dataflow and simulated in [MuJoCo](https://mujoco.org/).

## Overview

This system runs inference on an [OpenArm](https://github.com/openarm-org) bimanual robot (dual 7-DOF arms + grippers) using a transformer-based policy trained with LeRobot. The pipeline accepts camera observations and arm state, infers action chunks, and executes them on the robot or its MuJoCo simulation.

## Architecture

The system is built as a dora-rs dataflow graph defined in `dataflow-inference.yaml`:

```
mujoco (simulator)
  ├── arm observations (right/left)
  ├── 5 camera streams (JPEG)
  │
  ├──► observer (batches observations → Arrow IPC)
  │     └──► policy-server (ACT model inference via Unix socket)
  │           └──► actions-executor (upsample + filter → 250 Hz motor commands)
  │                 └──► mujoco (closes the loop)
  │
  └── tick (250 Hz → 30 Hz gating)
```

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

| Environment | Purpose | CUDA |
|---|---|---|
| `.venv` | dora dataflow orchestration | cu126 |
| `.venv_server` | Policy inference server | cu130 |

## Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/k1000dai/dora-openarm-inference-lerobot.git
cd dora-openarm-inference-lerobot

# Install dependencies
uv sync
```

## Usage

```bash
# Build and run the dataflow
uv run dora build dataflow-inference.yaml --uv
uv run dora run dataflow-inference.yaml
```

The inference server runs as a separate process, communicating via a Unix domain socket at `/dev/shm/policy-server.socket`.

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

## Dependencies

- **dora-rs** >= 0.5.0 - Dataflow orchestration
- **lerobot** == 0.3.3 - Pre-trained ACT policy
- **pyarrow** >= 15.0.0 - Arrow IPC serialization
- **Pillow** >= 10.0.0 - Image processing
- **mujoco** >= 3.6.0 - Physics simulation (in mujoco node)
