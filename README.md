# BOBR: Bayesian Optimisation of Bin boundaRies
This is Python-based framework that optimizes the bin boundaries to maximize the significance using Bayesian Optimization technique. The framework leverages on **Optuna** package for finding the best bin boundaries. 

This repository provides the official implementation of the BOBR binning approach described in our paper: [Learning to bin: differentiable and Bayesian optimization for multi-dimensional discriminants in high-energy physics](https://arxiv.org/abs/2601.07756).
If you use this code in your research or publications, please cite:
```
@article{Erdmann:2026opi,
    author = "Erdmann, Johannes and Kasaraguppe, Nitish Kumar and Mausolf, Florian",
    title = "{Learning to bin: differentiable and Bayesian optimization for multi-dimensional discriminants in high-energy physics}",
    eprint = "2601.07756",
    archivePrefix = "arXiv",
    primaryClass = "physics.data-an",
    month = "1",
    year = "2026"
}
```

## Installation
```
micromamba create -y -n bobr-env python=3.10 pip
python -m pip install -e ".[dev]"
```