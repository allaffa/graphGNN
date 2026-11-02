# GrainGNN Installation

Create one Python environment for GrainGNN on each machine. The laptop uses Python's `venv`; TACC uses Conda; Frontier uses the ROCm 7.2 Conda installer included in this repository. Follow only the section for the machine you are using.

## Laptop

From the repository root, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch for the laptop's hardware using the [official PyTorch selector](https://pytorch.org/get-started/locally/). For example, for a CPU-only environment:

```sh
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install GrainGNN's PyTorch Geometric and scientific Python dependencies:

```sh
./install_dependencies.sh
```

The script expects PyTorch to already be installed, so it will not replace a selected CPU, CUDA, or ROCm build. It installs only the core dependencies listed in `requirements-core.txt`; VTK is not included because it is only needed by the optional ExaCA VTK reader. Install it separately with `python -m pip install vtk` if using that reader.

## TACC Vista and Stampede3

The TACC environment guide has the complete installation and sample workflow. It uses the existing Conda installation under `$WORK`. In a login shell, initialize Conda, create the environment once, and activate it:

```sh
source "$WORK/anaconda3/bin/activate"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
conda create -n GNN --override-channels -c conda-forge python=3.12 pip
conda activate GNN
```

Then continue with the PyTorch and GrainGNN dependency installation in [the TACC guide](env_tacc.md). In a new shell or batch job, initialize Conda and activate the existing environment with:

```sh
source "$WORK/anaconda3/bin/activate" GNN
```

## OLCF Frontier

From the repository root on Frontier, run the installer. It creates a Conda environment at `GrainGNN-Installation-Frontier-ROCm72/graingnn_venv_rocm72` by default, installs PyTorch for ROCm 7.2, then installs GrainGNN's core dependencies:

```sh
bash scripts/hpc/olcf/frontier/installation/install-rocm72.sh
```

To activate the environment in a later shell, first change to the repository root (the default environment path is relative to it), then load the same module stack and activate its Conda prefix:

```sh
cd /path/to/graphGNN
module reset
module load cpe/24.07 cce/18.0.0 rocm/7.2.0 amd-mixed/7.2.0
module load craype-accel-amd-gfx90a PrgEnv-gnu miniforge3/23.11.0-0
conda activate "$PWD/GrainGNN-Installation-Frontier-ROCm72/graingnn_venv_rocm72"
```

The installer accepts `INSTALL_ROOT`, `VENV_PATH`, and `PYTHON_VERSION` overrides; see the script header for defaults. It does not install HydraGNN's additional packages or optional PyG compiled extensions.

## Verify

After installation, activate the environment and run the focused tests:

```sh
python -m pip install pytest
python -m pytest tests/
```

For the TACC end-to-end sample workflow, use the validation commands in [docs/env_tacc.md](env_tacc.md).