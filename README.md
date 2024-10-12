# Neighbor Graph Maneuver for Human-Centered Visualization (HUCAN)

## Pipeline

1. Generate a wide spectrum of visualisation results
2. Query users for a rating of satisfaction from 1 to 5
3. Build human surrogate: feature engineering + Random Forest
4. Pre-training policy using human surrogate as reward function
5. Fine-tuning policy using real user feedback

## Experiments

1. exp1: a simulation dataset containing 20 isometric Gaussians, i.e., each cluster is a Gaussian and there is a linear dependency between clusters
2. exp2: a real biology dataset which contains the same pattern, i.e., time-dependency between clusters

## Human Surrogate Training

For different experiments, human surrogates are trained in the following files:

- exp1.ipynb
- exp2.ipynb

codes and data path to be modified.

## Pre-train

```setup
    python ppo.py
```

## Fine-tune

```setup
    python inference.py
```

**Note:** Available parameters can be viewed using

```setup
    python ppo.py --help
```
