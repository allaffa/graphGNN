#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../../" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/GrainGNN-Installation-Frontier-ROCm72}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/graingnn_venv_rocm72}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_ROCM_INDEX_URL="${PYTORCH_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"

if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    elif [[ -f /usr/share/Modules/init/bash ]]; then
        source /usr/share/Modules/init/bash
    fi
fi

if ! command -v module >/dev/null 2>&1; then
    echo "Could not initialize the module command; run this script on OLCF Frontier." >&2
    exit 1
fi

module reset
module load cpe/24.07
module load cce/18.0.0
module load rocm/7.2.0
module load amd-mixed/7.2.0
module load craype-accel-amd-gfx90a
module load PrgEnv-gnu
module load miniforge3/23.11.0-0

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found after loading miniforge3." >&2
    exit 1
fi

mkdir -p "${INSTALL_ROOT}"
if [[ ! -d "${VENV_PATH}/conda-meta" ]]; then
    conda create --yes --prefix "${VENV_PATH}" "python=${PYTHON_VERSION}" pip
fi

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${INSTALL_ROOT}/.cache/pip}"
mkdir -p "${PIP_CACHE_DIR}"

conda run --prefix "${VENV_PATH}" python -m pip install \
    --index-url "${PYTORCH_ROCM_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VERSION}"

conda run --prefix "${VENV_PATH}" python - <<'PY'
import torch

assert torch.version.hip is not None, f"Expected a ROCm PyTorch build, got {torch.__version__}"
assert torch.version.hip.startswith("7.2."), f"Expected ROCm 7.2, got HIP {torch.version.hip}"
print(f"PyTorch {torch.__version__}; HIP {torch.version.hip}")
PY

conda run --prefix "${VENV_PATH}" bash "${REPO_ROOT}/install_dependencies.sh"

cat <<EOF
GrainGNN ROCm 7.2 installation complete.
Environment: ${VENV_PATH}

To activate it in a new shell, load the same module stack and run:
  conda activate ${VENV_PATH}
EOF