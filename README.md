# Benchmark Field-Level Inference

[![arXiv](https://img.shields.io/badge/astro--ph.CO-arXiv:2504.20130-b31b1b.svg)](https://arxiv.org/abs/2504.20130)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hsimonfroy/benchmark-field-level/blob/main/examples/infer_model.ipynb) 


## Overview
This repository accompanies the JCAP-submitted paper [**Benchmarking field-level cosmological inference from galaxy surveys**](https://arxiv.org/abs/2504.20130).

Field-level inference is a powerful approach that allows maximum information extraction from cosmological surveys (e.g. galaxy surveys) by modeling the entire observed field rather than just its summary statistics. This project provides a benchmark suite to evaluate different preconditioning strategies and high-dimensional sampling methods, and to compare their performance on an idealized but standardized galaxy clustering model.


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

<div align="center">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/13f56149-8678-422f-8b23-cf110f7a28ee">
     <img alt="model" src="https://github.com/user-attachments/assets/4bb759e3-39c5-4a0c-a038-729ef331b6bc" width="800">
   </picture>
</div>

The model also includes different preconditioning strategies for the linear matter field:
* Prior-preconditioning in real space
* Prior-preconditioning in Fourier space
* Static posterior-preconditioning assuming a Kaiser model 
* Dynamic posterior-preconditioning assuming a Kaiser model

## Inference
Interfaced MCMC samplers include:
* Hamiltonian Monte Carlo (HMC)
* No-U-Turn Sampler (NUTS)
* NUTS within Gibbs (NUTSwG)
* Metropolis-Adjusted Microcanonical Sampler (MAMS, i.e. adjusted MCHMC)
* MicroCanonical Langevin Monte Carlo (MCLMC, unadjusted)

<div align="center">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/cec09893-ed9a-4108-b78c-df273568b4bf">
     <img alt="sampler_triangle" src="https://github.com/user-attachments/assets/d20f77ca-580e-4747-9d67-8b7511de4b2f" width=500>
   </picture>
</div>

The cosmology $(\Omega_m, \sigma_8)$ and bias parameters $(b_1, b_2, b_{s^2}, b_{\nabla^2})$ are sampled jointly with the ($\geq 10^6$-dimensional) initial linear matter field $\delta_L$. From these samples, the universe history can be reconstructed, including the evolved galaxy density field $\delta_g$.

<div align="center">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/0747672f-d550-4967-854a-1dc23281c13c">
     <img alt="chains" src="https://github.com/user-attachments/assets/b161bc1d-fc2c-4f58-80f6-994239442d46">
   </picture>
</div>



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
