#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! python -c 'import torch' >/dev/null 2>&1; then
    echo "PyTorch is not installed in this Python environment." >&2
    echo "Install a build for this machine first: https://pytorch.org/get-started/locally/" >&2
    exit 1
fi

python -m pip install torch-geometric==2.8.1
python -m pip install -r "${SCRIPT_DIR}/requirements-core.txt"

echo "GrainGNN dependencies installed."
python -c 'import torch, torch_geometric; print(f"PyTorch {torch.__version__}; PyG {torch_geometric.__version__}")'