# Directed Influence Ising

[![Build Status](https://github.com/joanar18/directed_symptom_networks/actions/workflows/ci.yml/badge.svg)](https://github.com/joanar18/directed_symptom_networks/actions/workflows/ci.yml) [![PyPI version](https://badge.fury.io/py/directed_symptom_networks.svg)](https://badge.fury.io/py/directed_symptom_networks) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Directed Influence Ising** is a light‑weight, research‑ready Python package for estimating *directed* interactions between binary variables (symptoms, behaviours, events) using an extended Ising framework. It wraps reproducible training loops, evaluation metrics and data utilities so you can go from raw cascades to interpretable directed networks in a few lines of code.

---

## Key Features

* **Directed Influence Ising Fit** – learn asymmetric edge weights via penalised likelihood (ℓ₁/ℓ₂/MCP).
* **Modular API** – swap optimiser, scheduler or loss without touching core maths.
* **GPU‑aware trainer** – automatic device selection, mixed‑precision, gradient clipping.
* **Experiment scripts** – ready‑made baselines (IsingFit, PC‑corrected, MRF) and ablation flags.
* **Metrics** – precision\@k, AUROC, early‑influence detection and cascade replay accuracy.
* **Data utils** – clean import for benchmark datasets (PTSD, Social Contagion, etc.).
* **Visualisation** – ICM visualiser & interactive NetworkX export.

---

## Installation

```bash
# Python ≥3.10 with PyTorch ≥2.2 required
pip install directed_symptom_networks        # PyPI (coming soon)
# or dev install
git clone https://github.com/joanar18/directed_symptom_networks.git
cd directed_symptom_networks
pip install -e .[dev]
```

---

## Quick Start

```python
import torch
from directed_symptom_networks.core.models import DirectedInfluenceIsingModel
from directed_symptom_networks.trainer import Trainer

# synthetic demo
n = 25
model = DirectedInfluenceIsingModel(n_nodes=n, l1=0.01)
x = torch.randint(0, 2, (1024, n)).float()

trainer = Trainer(model, lr=1e-2, epochs=500)
trainer.fit(x)

G = model.to_networkx(threshold=0.05)
print(f"Learned {G.number_of_edges()} directed edges")
```

More runnable examples live in [`examples/`](examples/) and Jupyter notebooks in [`notebooks/`](notebooks/).

---

## Repository Layout

```
directed_symptom_networks/
  core/                # models, losses, trainer
  utils/                # metrics, data handling
  experiments/    # reproducible experiment scripts
tests/                # pytest suite
examples/         # quick demos
```

---

## Reproducibility Tips

1. Everywhere you run an experiment, set the seed:

```python
import torch, random, numpy as np
seed = 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
```

2. Save checkpoints with `state_dict()` not whole objects.
3. Log dependencies with `pip freeze > requirements.lock`.

---

## Contributing

Pull requests are welcome! Please run `ruff --fix` and `pytest -q` before pushing. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guidelines.

---

## Citation

If you use this code in your research, please cite:

```
@software{manjarinrocha2025directedising,
  author       = {J.J. Manjarín and J. Rocha},
  title        = {{Directed Influence Ising}: A Python package for fitting directed Ising models},
  month        = may,
  year         = 2025,
  version      = {0.1.0},
  url          = {https://github.com/TU_SITIO/directed_symptom_networks}
}
```

---

## License

Distributed under the MIT License – see the [`LICENSE`](LICENSE) file for details.

---

## Acknowledgements

This project was developed as part of the **Directed Symptom Networks for Causal Analysis in Psychopathology Thesis** at IE University and is inspired by the seminal work of Epskamp *et al.* (2018) and Haslbeck *et al.* (2021).
