# Benchmark Field-Level Inference

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/astro--ph.CO-arXiv:2504.XXXX-b31b1b.svg)](https://arxiv.org/abs/2504.XXXX)


## Overview
This repository accompanies the JCAP-submitted paper [**Benchmarking field-level cosmological inference from galaxy surveys**]().

Field-level inference is a powerful approach that allows us to extract maximum information from cosmological surveys (e.g. galaxy surveys) by modeling the entire observed field rather than just its summary statistics. This project provides a benchmark suite to evaluate different inference methods and compare their performance on an idealized but standardized galaxy clustering model.


## Installation

To install the package, use the following command:

```bash
pip install -e git+https://github.com/hsimonfroy/benchmark-field-level.git#egg=flbench
```

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

## Inference


## Tutorial

The `tuto/` directory contains Jupyter notebooks that demonstrate how to use the benchmark tools:


1. [`infer_model.ipynb`](https://github.com/hsimonfroy/benchmark-field-level/blob/main/tuto/infer_model.ipynb):
    * Experimental setup
        * Instantiate a cosmological model
        * Generate observation and condition the model on it
    * Perform the inference
        * Warmup phase only inferring the field with MCLMC sampler
        * Warmup phase inferring all parameters jointly with any implemented sampler 

2. [`sample_analysis.ipynb`](https://github.com/hsimonfroy/benchmark-field-level/blob/main/tuto/sample_analysis.ipynb):
    * Assess convergence
        * Chain diagnostics
        * Inspection at the field-level
    * Quantify performance

