# Multivariate online change point detection

This repository contains implementations of parametric and non-parametric online change point detection (CPD) methods for multivariate data. The methods are evaluated on synthetic and real-world datasets, and the repository provides tools for experimentation and analysis.

## Setting up the environment

The experiments were conducted using an **Anaconda environment**. To set up the environment:

1. Clone the repository:

        git clone git@github.com:ojanenmarianna/multivariate_online_cpd.git
        cd multivariate_online_cpd

3. Create and activate the environment; in the yml-file replace <env_name> with the environment name you want to use (e.g. mocpd) and <path_to_env> with the path to your conda environments. After replacing them, run:

        conda env create -f conda_env.yml
        conda activate <env_name>

## Datasets

### Included datasets

- Synthetic datasets for testing can be created using the following functions:  
  - **generate_multivariate_signal**
  - **create_s_abrupt**  
  - **create_s_gradual**
- The synthetic anomaly datasets can be downloaded from <https://github.com/Mohamed164/AD-microservice-app/tree/main>
- Real-world datasets are not included in this repository but were obtained in 2024 from the following sources:
  - Electricity Transformer Dataset (ETDataset): <https://github.com/zhouhaoyi/ETDataset> 
  - Electricity Load Diagrams (ELD): <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014>
  - Datasets annotated by the Turing Institute:
    - Bee Waggle: <https://www.cc.gatech.edu/~borg/ijcv_psslds/>
    - Room Occupancy: <https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+>
    - Run Log: <https://github.com/alan-turing-institute/TCPD/tree/master/datasets/run_log>


### Adding datasets

Place the datasets into `./src/data` directory.

## Running the Experiments

TODO: run all experiments

You can run CPD with a specific method for a particular dataset by running the `test_method.py` file. Replace <method_name> with the method you want to use and <dataset_name> with the dataset you want to run the CPD to:

        python3 test_method.py <method_name> <dataset_name>

## Results

- Results are stored in the `results/` directory in CSV format.

## References

TODO: Add links to relevant papers/repositories
