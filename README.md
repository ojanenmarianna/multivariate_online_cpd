# Multivariate online change point detection

This repository contains implementations of parametric and non-parametric online change point detection (CPD) methods for multivariate data. The methods are evaluated on synthetic and real-world datasets, and the repository provides tools for experimentation and analysis.

## Setting up the environment

The experiments were conducted using an **Anaconda environment**. To set up the environment:

1. Clone the repository:

        git clone git@github.com:ojanenmarianna/mocpd.git
        cd mocpd

3. Create and activate the environment, in the yml-file replace <env_name> with the environment name you want to use (e.g. mocpd) and <path_to_env> with the path to your conda environments. After replacing them, run:

        conda env create -f conda_env.yml
        conda activate <env_name>

## Datasets

### Included datasets

- Synthetic datasets for testing can be created using the following functions:  
  - **generate_multivariate_signal**  
  - **create_s_abrupt**  
  - **create_s_gradual**  
- Real-world datasets are not included in this repository but can be downloaded from the following sources:
  - TODO: Add links or references to the datasets here.


### Adding datasets

Place the datasets into `./src/data` directory.

## Running the Experiments

TODO: run all experiments
TODO: run a specific method with a specific dataset

## Results

- Results are stored in the `results/` directory in CSV format.

## References

TODO: Add link to relevan papers/repositiories
