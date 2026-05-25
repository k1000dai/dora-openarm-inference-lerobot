FROM python:3.12

# use bash
SHELL ["/bin/bash", "-c"]
WORKDIR /project

COPY pyproject.toml .
COPY src/ src/
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

RUN uv venv .venv
RUN uv pip install .
RUN uv pip install torch torchvision --torch-backend=cu128 --upgrade


# docker-policy-server mounts --volume=cache:/cache at runtime.
# Pin all caches there so offline inference works with --network=none.
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch

# Pre-download all model weights at build time.
# 1. LeRobot policy checkpoint (config + safetensors)
RUN source .venv/bin/activate && python -c "\
from lerobot.policies.pretrained import PreTrainedConfig; \
from lerobot.policies.factory import get_policy_class; \
cfg = PreTrainedConfig.from_pretrained('k1000dai/act_openarm_pick_cube_40k'); \
cfg.pretrained_path = 'k1000dai/act_openarm_pick_cube_40k'; \
get_policy_class(cfg.type).from_pretrained(config=cfg, pretrained_name_or_path=cfg.pretrained_path)"
# 2. ResNet18 backbone (used by ACT policy)
RUN source .venv/bin/activate && python -c "import torchvision; torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)"

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && dora-openarm-inference-lerobot \"$@\"", "--"]
