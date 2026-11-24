# 0. Install

ResFlow needs Python 3.12. The pinned versions are torch 2.5.1 and torch_geometric 2.7.0. Training and rollouts also run
on the CPU, but are slow there (see [05_train.md](05_train.md)).

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
# torch 2.5.1: pick one
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU (CUDA 12.4)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu     # CPU only
pip install torch_geometric==2.7.0
pip install -e .
resflow --help                      # lists convert, build, fit-norm, train-state, train-rate, finetune, rollout
```

`pip install -e .` also installs numpy, pandas, scipy, h5py, xarray, tqdm, matplotlib and PyYAML. The tests need
`pytest`: `pip install -e .[dev]`.

Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## Deck builder

The deck builder has its own requirements (see [02_build_decks.md](02_build_decks.md)): `cd deck_builder && pip install -r requirements.txt`.
It can use the same venv.

## OPM Flow (open-source simulator, optional)

To make training data without an IMEX licence ([04_opm_conversion.md](04_opm_conversion.md)), install OPM in a separate venv:

```bash
python3.12 -m venv .venv-opm
source .venv-opm/bin/activate
pip install opm opm-simulators
```

The wheels exist for Linux x86_64 only. On Windows use WSL. The OPM venv runs the simulator (`deck_builder/run_opm_batch.py`);
conversion, training and rollout run in the ResFlow venv.
