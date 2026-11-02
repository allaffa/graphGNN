# GrainGNN Installation

GrainGNN supports laptop installations and HPC installations. Install a PyTorch build appropriate for the machine first; the remaining dependencies are installed by the repository script.

## Laptop

Create and activate a virtual environment from the repository root:

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

## HPC

Use the facility-specific installer where one is provided. The current ROCm target is OLCF Frontier with ROCm 7.2:

```sh
bash scripts/hpc/olcf/frontier/installation/install-rocm72.sh
```

This loads the Frontier ROCm 7.2 module stack, creates a Conda environment under `GrainGNN-Installation-Frontier-ROCm72/`, and installs the ROCm 7.2 PyTorch wheel before GrainGNN's core dependencies. It does not install HydraGNN's additional packages or optional PyG compiled extensions. It accepts `INSTALL_ROOT`, `VENV_PATH`, and `PYTHON_VERSION` overrides; see the script header for defaults.

For TACC Vista (GH200) and Stampede3 (H100), use the existing [TACC environment guide](env_tacc.md). That setup is unchanged.

## Verify

After installation, activate the environment and run the focused tests:

```sh
python -m pip install pytest
python -m pytest tests/
```

For the TACC end-to-end sample workflow, use the validation commands in [docs/env_tacc.md](env_tacc.md).