# Field-Level Cosmological Inference Benchmark

This repository contains tools and notebooks for benchmarking field-level cosmological inference from galaxy surveys. The project aims to provide a standardized framework for evaluating and comparing different inference methods in cosmological field-level analysis.

## Overview

Field-level cosmological inference is a powerful approach that allows us to extract maximum information from galaxy survey data by modeling the entire observed field rather than just summary statistics. This project provides a benchmark suite to evaluate different inference methods and compare their performance on an idealized but standardized galaxy clustering analysis.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hsimonfroy/benchmark-field-level.git
cd benchmark-field-level
```

2. Install the package:
```bash
pip install -e .
```

## Tutorial Notebooks

The `intro/` directory contains Jupyter notebooks that demonstrate how to use the benchmark tools:

1. `sample_analysis.ipynb`: This notebook provides a comprehensive introduction to sample analysis in field-level cosmological inference. It covers:
   - Loading and preprocessing galaxy survey data
   - Basic statistical analysis
   - Visualization techniques
   - Common pitfalls and best practices

2. `infer_model.ipynb`: This notebook focuses on model inference techniques, including:
   - Setting up inference models
   - Parameter estimation
   - Model comparison
   - Performance evaluation
