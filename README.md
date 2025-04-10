# Benchmark Field-Level Inference

[![arXiv](https://img.shields.io/badge/astro--ph.CO-arXiv:2504.XXXX-b31b1b.svg)](https://arxiv.org/abs/2504.XXXX)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsimonfroy/benchmark-field-level/blob/main/examples/infer_model.ipynb) 


## Overview
This repository accompanies the JCAP-submitted paper [**Benchmarking field-level cosmological inference from galaxy surveys**]().

Field-level inference is a powerful approach that allows maximum information extraction from cosmological surveys (e.g. galaxy surveys) by modeling the entire observed field rather than just its summary statistics. This project provides a benchmark suite to evaluate different preconditioning strategies and sampling methods, and to compare their performance on an idealized but standardized galaxy clustering model.


## Install
For standard installation of the package and its dependencies, use:

```bash
pip install git+https://github.com/hsimonfroy/benchmark-field-level.git
```

For development purposes, install in editable mode:

```bash
pip install -e git+https://github.com/hsimonfroy/benchmark-field-level.git#egg=flbench
```

This package relies on [JAX](https://github.com/google/jax) for GPU-accelerated computations. Please note that its installation is left to the user. Follow the [official JAX installation guide](https://github.com/google/jax#installation) to set it up for your specific hardware.

## Model

The benchmarked galaxy clustering model is **fast** (jit-compiled and GPU accelerated) and **differentiable**. It includes:
 * Linear matter field generation
 * Structure formation, selected among
    * Linear growth
    * Lagrangian Perturbation Theory (1LPT or 2LPT) displacement
    * Particle Mesh (PM) N-body displacement, with BullFrog or FastPM solvers
 * Redshift-Space Distortions (RSD)
 * Second order Lagrangian galaxy bias
 * Observational Noise

https://github.com/hsimonfroy/montecosmo/assets/85559558/b64ff962-00fd-4b5f-8ea2-4f476c325cd0

The model also includes different preconditioning strategies for the linear matter field:
* Prior-preconditioning in real space
* Prior-preconditioning in Fourier space
* Static posterior-preconditioning assuming a Kaiser model 
* Dynamic posterior-preconditioning assuming a Kaiser model

## Inference


## Examples
The `examples/` directory contains Jupyter notebooks that demonstrate how to use the benchmark tools:

1. [**`infer_model.ipynb`**](https://github.com/hsimonfroy/benchmark-field-level/blob/main/examples/infer_model.ipynb)  
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsimonfroy/benchmark-field-level/blob/main/examples/infer_model.ipynb)  
    * Experimental setup:
      * Instantiate a field-level cosmological model
      * Generate observation and condition the model on it
    * Perform field-level inference:
      * Warmup phase only inferring the field with MCLMC sampler
      * Warmup phase inferring all parameters jointly with any implemented sampler

2. [**`sample_analysis.ipynb`**](https://github.com/hsimonfroy/benchmark-field-level/blob/main/examples/sample_analysis.ipynb)  
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsimonfroy/benchmark-field-level/blob/main/examples/sample_analysis.ipynb)  
    * Assess convergence:
      * Chain diagnostics
      * Inspection at the field-level
    * Quantify performance
