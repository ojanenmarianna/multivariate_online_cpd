import argparse
import sys
import re
import torch.nn as nn
from rpy2 import robjects
from rpy2.robjects import numpy2ri
import numpy as np
from roerich.change_point import OnlineNNClassifier, OnlineNNRuLSIF
from utils.basic_functions import load_data_from_csv, load_data_from_json, load_eld_data, load_anomaly_data
from utils.feature_functions import fourier_feat, generate_frequencies, gauss_kernel
from utils.generate_multivariate_data import generate_multivariate_signal, create_s_abrupt, create_s_gradual
from utils.plots import plot_changes_for_every_feature, plot_changes_for_every_cluster
from algos.bocpd import MultivariateBOCPD
from algos.newma_scanb import NEWMA, ScanB, select_optimal_parameters

sys.path.append('../')


class SimpleNN(nn.Module):
    """Custom neural network for ONNR"""
    def __init__(self, n_inputs=1):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_inputs, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 1))

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, n_inputs).

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        return self.net(x)


def load_dataset(dataset_name):
    """Function to load datasets dynamically."""
    data_path = "./src/data/"
    features = None

    if dataset_name == "ettm":
        df = load_data_from_csv(data_path + "ETTm.csv")
        features = df.iloc[:10000, 1:]  # First 10,000 rows
    elif dataset_name == "etth2":
        df = load_data_from_csv(data_path + "ETTh2.csv")
        features = df.iloc[:, 1:]
    elif dataset_name == "eld":
        features = load_eld_data(data_path + "LD2011_2014.txt").iloc[:20000, :]
    elif dataset_name == "bee":
        features = load_data_from_json(data_path + "bee_waggle_6.json")
    elif dataset_name == "occupancy":
        df = load_data_from_csv(data_path + "occupancy.txt")
        features = df.iloc[:, 1:5].iloc[::16]  # Use first 4 features and sample every 16 rows
    elif dataset_name == "run_log":
        features = load_data_from_csv(data_path + "run_log.csv")
    elif dataset_name == "syn":
        lengths = [200, 50, 200, 50, 100, 300, 200, 300, 200]
        syn_gt = np.cumsum(lengths[:-1]).tolist()
        features = generate_multivariate_signal(lengths, syn_gt)
    elif dataset_name == "13-05":
        load_anomaly_data(data_path, "13-05-2024_metrics.csv")
    elif dataset_name == "20-05":
        load_anomaly_data(data_path, "20-05-2024_metrics.csv")
    elif dataset_name == "s_a":
        features, _ = create_s_abrupt()
    elif dataset_name == "s_g":
        features, _ = create_s_gradual()
    else:
        print(f"Error: Dataset '{dataset_name}' not recognized.")
        sys.exit(1)

    return features


def run_cpd_method(method_name, dataset_name):
    """Run a cpd with the chosen method for a chosen dataset and
    plot detected change points."""

    data = load_dataset(dataset_name)
    data_np = data.to_numpy()
    dimensions = len(data_np[0])

    if method_name == "bocpd":
        buffer_size = 1000
        t = 0
        offset = 0
        change_points = []
        bocpd = MultivariateBOCPD(dimensions, p_cp=0.9, buffer_size=buffer_size)

        for x in data_np:
            detected_cp = bocpd.update(t, x, offset)

            if detected_cp:
                change_points.append(detected_cp)
            t += 1

            # Reinitialize after reaching window size
            if t % buffer_size == 0:
                bocpd = MultivariateBOCPD(dimensions, p_cp=0.9, buffer_size=buffer_size)
                t = 0
                offset += buffer_size

    elif method_name == "lrt":
        r_source = robjects.r['source']
        r_source("./src/algos/lrt.r")

        detect_and_segment = robjects.globalenv['detect_and_segment']

        r_data = numpy2ri.numpy2rpy(data_np)
        r_dimensions = robjects.IntVector([dimensions])

        lrt_changepoints = detect_and_segment(r_data, r_dimensions, 0, 1)

        change_points = list(lrt_changepoints)

    elif method_name == "newma" or method_name == "scanb":
        # common config fro Newma and Scan B,
        choice_sigma = 'median'
        numel = 100
        B = 60
        N = 3
        n = 50
        detector = None

        big_Lambda, small_lambda = select_optimal_parameters(B)
        thres_ff = small_lambda
        m = int((1 / 4) / (small_lambda + big_Lambda) ** 2)

        X = data_np

        data_sigma_estimate = X[:numel]
        W, sigmasq = generate_frequencies(m, dimensions, data=data_sigma_estimate,
                                          choice_sigma=choice_sigma)

        if method_name == "scanb":
            print('Start algo Scan B... (can be long !)')
            detector = ScanB(
                X[0],
                kernel_func=lambda x, y: gauss_kernel(x, y, np.sqrt(sigmasq)),
                window_size=B,
                nbr_windows=N,
                adapt_forget_factor=thres_ff)

        elif method_name == "newma":
            print('Start algo Newma RF ...')
            print('# RF: ', m)

            def feat_func(x):
                return fourier_feat(x, W)

            detector = NEWMA(X[0], forget_factor=big_Lambda, forget_factor2=small_lambda,
                             feat_func=feat_func, adapt_forget_factor=thres_ff)

        detector.apply_to_data(X)

        # compute performance metrics
        detection_stat = np.array([i[0] for i in detector.stat_stored])[int(10 * n):]
        online_th = np.array([i[1] for i in detector.stat_stored])[int(10 * n):]

        change_points = []

        detected_cp = np.where(detection_stat > online_th)[0]  # Detected change points
        # Do not add consecutive detected change points to the list
        for cp in detected_cp:
            if not change_points:
                change_points.append(cp)
            elif (cp) - change_points[-1] > 20:
                change_points.append(cp)

    elif method_name == "onnc":
        # Initialize the detector
        onnc = OnlineNNClassifier(periods=1, window_size=1, lag_size=100, step=1,
                                n_epochs=1, lr=0.01, lam=0.0001, optimizer="Adam")

        _, cps_pred = onnc.predict(data_np)
        change_points = cps_pred.tolist()

    elif method_name == "onnr":
        onnr = OnlineNNRuLSIF(net=SimpleNN, alpha=0.1, window_size=1, lag_size=100, step=1,
                n_epochs=1, lr=0.01, lam=0.001, optimizer="Adam")

        _, cps_pred = onnr.predict(data_np)
        change_points = cps_pred.tolist()

    else:
        print(f"Error: Method '{method_name}' not recognized.")
        sys.exit(1)

    print(f"Results for {method_name} on {dataset_name}:")
    print(change_points)
    if re.match(r"eld|13-05|20-05", dataset_name):
        plot_changes_for_every_cluster(data, change_points,
                                       f"{method_name} for {dataset_name} data")
    else:
        plot_changes_for_every_feature(data, change_points,
                                       f"{method_name} for {dataset_name} data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPD methods on chosen datasets")
    parser.add_argument("method", type=str,
                        help="CPD method name (bocpd, lrt, scanb, newma, onnc, onnr)")
    parser.add_argument(
        "dataset", type=str,
        help="Dataset name (ettm, etth2, eld, bee, occupancy, run_log, syn, 13-05, 20-05, s_a, s_g)"
    )

    args = parser.parse_args()
    run_cpd_method(args.method, args.dataset)
