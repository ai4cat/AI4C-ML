# AI4C-ML: Machine Learning for Atomically Dispersed Catalysts

AI4C-ML provides machine learning workflows for atomically dispersed catalyst (ADC) design, including atomic structure generation, ORR performance prediction, active learning, and reaction barrier modeling.

This repository contains three main modules:

1. `Atom_Generation`: atomic structure generation
2. `GPGB_AL`: descriptor-based half-wave potential prediction and active learning
3. `H2O2_decom_bar`: hydrogen peroxide decomposition barrier prediction

---

## Installation

### Development Environment

- Python 3.13
- Validated on Ubuntu 24.04.4 LTS

Use the following command to create the required environment:

```bash
conda env create -f ai4cml.yml
```

### Setup

To download and set up the codes, run:

```bash
git clone https://github.com/ai4cat/AI4C-ML.git
cd AI4C-ML
```

---

## Repository Structure

```text
AI4C-ML
├── Atom_Generation
│   ├── filename.py
│   ├── generation.py
│   └── atom_combine.py
│
├── GPGB_AL
│   ├──code
│   ├──── main.py
│   ├──── read_data.py
│   ├──── data_sanity_check.py
│   ├── parm
│   └──── model_params.json
│
├── H2O2_decom_bar
│   ├──code
│   ├──── predict.py
│   ├──── training.py
│   └── model
│
├── ai4cml.yml
├── LICENSE
└── README.md
```

---

## 1. Atom_Generation

`Atom_Generation` is used for atomic structure generation. Due to the structural diversity of atomically dispersed catalysts, this repository provides one representative workflow for the batch generation of dual-atom catalyst models. The related CIF files can be found in the [database](http://openadc.com.cn:23345/).

This module requires `.vasp` files as input. The generated structures can be further used for downstream DFT calculations or machine learning workflows.

### Workflow

```text
Input .vasp file
      ↓
Modify filename.py
      ↓
Run generation.py
      ↓
Run atom_combine.py
      ↓
Output processed .vasp files
```

### Step 1: Modify `filename.py`

First, modify `filename.py` to define the naming rule and identify the positional relationship of the target atoms.

This step is used to ensure that the generated structures are named consistently and that the relationship between the selected atoms can be correctly recognized.

### Step 2: Run `generation.py`

Then, run `generation.py` to generate candidate structures.

In this script, users can adjust:

- Metal atom types
- Surrounding coordination atoms
- Local atomic combinations
- Batch generation settings

This step is the core structure generation process.

### Step 3: Run `atom_combine.py`

Finally, use `atom_combine.py` to process the generated `.vasp` files.

This script is used to further organize and standardize the output structures, ensuring that the generated files can be smoothly used in subsequent computational tasks.

### Main Features

- Batch generation of atomically dispersed catalyst structures
- Support for dual-atom catalyst model construction
- Flexible control of metal centers and local coordination environments
- Output `.vasp` files compatible with downstream DFT and ML workflows

---

## 2. GPGB_AL

`GPGB_AL` is a machine learning workflow for predicting ORR half-wave potential based on descriptors. It uses the GPGB framework and supports both model training and prediction.

The module is designed for descriptor-based catalyst performance prediction and active learning-driven candidate selection.

### Main Script

The main script is:
```bash
main.py
```
This script is used for model training and prediction.

### Important Notes

Before running the workflow, please make sure that:

- The dataset path is correctly specified.
- The input dataset strictly follows the format of the example dataset.
- The example dataset is available on the [Hugging Face page](https://huggingface.co/datasets/ai4c/AI4C-ML-Dataset/tree/main/GPGB_AL).
- `main.py` and `read_data.py` are placed in the same directory.
- Model parameters can be modified in [`model_params.json`](https://github.com/ai4cat/AI4C-ML/blob/main/GPGB_AL/param/model_params.json). Please make sure this file is placed in the correct path before running the workflow.

The dataset format should be consistent with the provided example. Otherwise, the program may fail to correctly read descriptors, labels, or prediction data.

### Workflow

```text
Prepare descriptor dataset
      ↓
Check dataset path and format
      ↓
Run main.py
      ↓
Train GPGB model
      ↓
Predict half-wave potential
      ↓
Use active learning for candidate selection
      ↓
Check duplicate data using data_sanity_check.py
```

### Dataset Sanity Check

After model training, prediction, or active learning selection, users can run:
```bash
python data_sanity_check.py
```
This script checks whether duplicate rows exist within or between different output data tables.

It can be used to confirm:

- Whether the prediction set contains samples that have already appeared in the training set
- Whether newly selected active-learning data overlap with the original training dataset
- Whether duplicate rows exist inside each dataset
- Whether the training, prediction, and active-learning datasets are properly separated

### Main Features

- Descriptor-based ORR performance prediction
- Half-wave potential prediction using `main.py`
- GPGB framework for model training and prediction
- Active learning workflow for candidate selection
- Dataset sanity checking for duplicate and overlapping samples
- Compatible with example datasets provided on Hugging Face

---

## 3. H2O2_decom_bar

`H2O2_decom_bar` is used for reaction barrier prediction, with a focus on hydrogen peroxide decomposition barriers.

This module provides supervised learning workflows for estimating reaction energy barriers and can support large-scale screening of catalytic stability.

### Application

This module can be used for:

- Prediction of hydrogen peroxide decomposition barriers
- Energy barrier estimation using supervised learning
- Large-scale screening of catalyst stability
- Evaluation of catalyst stability-related reaction processes

### Dataset

The example dataset is provided on the [Hugging Face page](https://huggingface.co/datasets/ai4c/AI4C-ML-Dataset/tree/main/H2O2_decom_bar).

Please ensure that the input data follow the same format as the example dataset before running the workflow.

### Main Features

- Prediction of hydrogen peroxide decomposition barriers
- Supervised learning workflows for energy barrier estimation
- Support for large-scale screening of catalytic stability
- Useful for evaluating catalyst stability-related reaction processes

---

## Features

AI4C-ML integrates structure generation, performance prediction, active learning, and reaction barrier modeling into a unified workflow for atomically dispersed catalyst discovery.

Main features include:

- End-to-end ML workflow for atomically dispersed catalyst discovery
- Atomic structure generation for candidate ADC models
- Descriptor-based modeling for ORR half-wave potential prediction
- Active learning for efficient candidate selection
- Reaction barrier prediction for catalytic stability screening
- Support for high-throughput and data-driven catalyst discovery

---

## Example Data

Example datasets are provided on the [Hugging Face page](https://huggingface.co/datasets/ai4c/AI4C-ML-Dataset).

Please strictly follow the format of the example datasets when preparing new input data.

Recommended checks before running the workflows:

- Confirm that the dataset path is correct
- Confirm that column names and data formats are consistent with the example files
- Confirm that descriptor columns and target labels are correctly assigned
- Use `data_sanity_check.py` when necessary to avoid data leakage or duplicate samples

---

## Notes

- For `Atom_Generation`, users should prepare valid `.vasp` input files before running the scripts.
- For `GPGB_AL`, `main.py` and `read_data.py` should be placed in the same directory.
- For `GPGB_AL`, the dataset format must strictly follow the example dataset.
- For `H2O2_decom_bar`, the example dataset can be found on the Hugging Face page.
- The generated structures and prediction results can be further used for downstream DFT calculations, model optimization, and catalyst screening.

---

## Contributing
Contributions are welcome! Please follow the standard fork-and-pull request workflow on GitHub.

If you use our code in your research, please cite our paper:
```bash
@article{,
  title={s},
  author={},
  journal={},
  year={},
  volume = {},
  pages = {}
}
```

---

## License

This project is licensed under the [Apache License 2.0](https://github.com/ai4cat/AI4C-ML/blob/main/LICENSE).

Please see the LICENSE file for more details.
