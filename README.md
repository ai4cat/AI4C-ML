# AI4C-ML: Machine Learning for Atomically Dispersed Catalysts

This repository provides machine learning workflows for atomically dispersed catalyst (ADC) design, including structure generation, performance prediction, and reaction barrier modeling.

## Installation
### Development Environment
Python 3.13
Validated on Linux/Windows OS
Use `conda env create -f mltrain.yml` to create the enviornment.
Setup
To set up the codes, run the following commands:

https://github.com/ai4cat/AI4C-ML.git
cd AI4C-ML

## Repository Structure

### 1. `Atom_Generation`
Atomic structure generation module for constructing candidate ADC configurations.

- High-throughput generation of atomically dispersed catalyst structures  
- Flexible control of metal centers and coordination environments  
- Outputs compatible with downstream DFT and ML workflows  

---

### 2. `GPGB_AL`
Machine learning pipeline for ORR performance prediction.

- GPGB (Genetic Programming + Gradient Boosting) framework  
- Training and testing workflows for half-wave potential (E<sub>1/2</sub>)  
- Active learning strategy for candidate selection  
- Descriptor-based modeling and feature optimization  

---

### 3. `H2O2_decom_bar`
Machine learning models for reaction barrier prediction.

- Prediction of hydrogen peroxide decomposition barriers  
- Supervised learning workflows for energy barrier estimation  
- Supports large-scale screening of catalytic stability  

---

## Features

- End-to-end ML workflow for catalyst discovery  
- Integration of structure generation, performance prediction, and stability evaluation  
- Designed for high-throughput and data-driven catalyst screening  

---

## **License**

This project is licensed under the [CC-BY-ND-NC License](https://github.com/ai4cat/AI4C-ML/blob/main/LICENSE). Please see the LICENSE file for more details.
