CPURP: Relation Prediction in Uncertain Knowledge Graphs with Relational Context and Relation Paths

This repository contains the anonymous supplementary implementation for CPURP, a context- and path-based model for relation prediction in uncertain knowledge graphs (UKGs).

In a UKG, a fact is represented as a quadruple (h, r, t, w), where h and t are entities, r is a relation, and w is a confidence score. Given an entity pair (h, t), CPURP predicts a relation and estimates the confidence of the resulting fact.

Overview

CPURP combines two complementary types of structural evidence:

Relational context around the head and tail entities, modeled with edge-centered message passing and uncertainty-aware neighbor gating.

Uncertain multi-hop relation paths between the entity pair, encoded with path semantics and confidence statistics, then aggregated by candidate-relation-aware attention.

The model is jointly optimized for relation prediction and confidence estimation.

Repository structure

.
├── configs/                 # Experiment configuration files
├── data/                    # Dataset directory (see Data preparation)
├── scripts/                 # Shell scripts for experiments
├── src/                     # Source code
├── ukg/                     # UKG data processing and model utilities
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── run_sensitivity.py       # Sensitivity-analysis entry point
└── plot_sensitivity.py      # Plotting utility for sensitivity results

Requirements

The code was developed with Python 3.9+ and PyTorch. Create a clean environment, then install the packages required by the imported modules in the source code. A typical setup is:

python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install torch numpy scipy pyyaml tqdm scikit-learn

If a module is reported as missing when running the code, install it with pip install <module-name>.

Data preparation

Experiments use the public uncertain knowledge graph benchmarks CN15K, NL27K, and PPI5K. Obtain the datasets from their original public sources, then place them under data/ using the paths expected by configs/default.yaml.

Before training, open configs/default.yaml and verify:

the dataset name;

the dataset directory;

model and training hyperparameters; and

the output directory for checkpoints and logs.

Generated checkpoints, logs, cached files, and result files are intentionally excluded from this repository.

Running experiments

The default configuration is located at configs/default.yaml.

# Train a model
python train.py --config configs/default.yaml

# Evaluate a trained model
python evaluate.py --config configs/default.yaml

# Run sensitivity experiments
python run_sensitivity.py --config configs/default.yaml

# Plot sensitivity results after they have been generated
python plot_sensitivity.py

If your local version of an entry script uses a different command-line argument name, run python <script>.py --help and use the configuration argument shown there.

Reproducibility notes

For each benchmark, use the split files and preprocessing procedure provided with that benchmark. Use the hyperparameters in configs/default.yaml for the main experiments, and run multiple random seeds when reproducing the reported averages.

The supplementary repository is provided for anonymous review. It intentionally contains no author names, affiliations, email addresses, or private experimental artifacts.

License

See LICENSE for licensing information.
