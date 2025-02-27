import json
import numpy as np
import pandas as pd
import csv

def load_data_from_csv(path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    ----------
    path : str
        The file path to the CSV file.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the data loaded from the CSV file.
    """
    df = pd.read_csv(path)

    return df


def load_data_from_json(path):
    """
    Load time-series data from a JSON file into a pandas DataFrame.

    Parameters:
    ----------
    path : str
        The file path to the JSON file.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the time-indexed series data.
        The index corresponds to the "time" field, and columns are labeled series.
    """
    with open(path, 'r') as file:
        data = json.load(file)

    # Extract time index and series data
    time_index = data["time"]["index"]
    series_data = {s["label"]: s["raw"] for s in data["series"]}

    df = pd.DataFrame(series_data, index=time_index)

    return df


def load_eld_data(path):
    """
    Load and preprocess the Electricity Load Diagrams data from a .txt file.
    Preprocessing includes converting the first column into a datetime object, and
    removing clients that were created during 2011 or after 2011.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the clients loaded from a text file.
    """
    eld_df = pd.read_csv(path, delimiter=";", low_memory=False)

    eld_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    eld_df['date'] = pd.to_datetime(eld_df['date'])

    eld_df = eld_df.replace(',', '.', regex=True)

    for col in eld_df.columns:
        if col != 'date':  # Skip the 'date' column
            eld_df[col] = eld_df[col].astype('float32')

    df_jan_2011 = eld_df[
        (eld_df['date'].dt.year == 2011) &
        (eld_df['date'].dt.month.isin([1]))
    ]

    jan_zero_columns = df_jan_2011.drop(columns=['date']).columns[
        (df_jan_2011.drop(columns=['date']) == 0).all(axis=0)
    ]

    eld_df_cleaned = eld_df.drop(columns=jan_zero_columns)

    eld_features = eld_df_cleaned.iloc[:, 1:]

    return eld_features

def load_anomaly_data(sdir, sfiles):
    """
    Load and preprocess the synthetic anomaly dataset from a csv file.
    
    Returns:
    -------
    np.ndarray
        A ndarray containing relevant features loaded from a csv file.
    """
    ifile = 0

    nexamples = 2000
    with open(sdir + sfiles[ifile], 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        lrows = [row[1:] for row in reader][1:nexamples]

    xdata = np.array(lrows, dtype=float)
    xdata = xdata[:, np.std(xdata, axis=0) != 0]

    return xdata

def filter_changepoints(all_changepoints, detected_cp, range_threshold, offset=0):
    """
    Filter and add detected changepoints to a list, avoiding consecutive points.

    Parameters:
    ----------
    all_changepoints : list
        List to store filtered changepoints.
    detected_cp : list
        List of detected changepoints at the current step.
    range_threshold : int
        Minimum distance required between consecutive changepoints.
    offset : int
        Offset to be added to each changepoint.

    Returns:
    -------
    list
        Updated list of filtered changepoints.
    """
    for cp in detected_cp:
        if cp == 0: # Ignore the detected cp at time step 0, as R[0, 0] = 1 always
            continue
        if not all_changepoints:
            all_changepoints.append(cp + offset)
        elif (cp + offset) - all_changepoints[-1] > range_threshold:
            all_changepoints.append(cp + offset)

    return all_changepoints


def compute_fft(data):
    """
    Compute the Fast Fourier Transform (FFT) for each feature in the data.

    Parameters:
    ----------
    data : np.ndarray or pd.DataFrame
        Input data with each column representing a feature.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the magnitude of the FFT for each feature.
    """
    fft = np.fft.fft(data, axis=0)
    magnitudes = np.abs(fft)
    fft_results = pd.DataFrame(magnitudes)

    return fft_results


def evaluate_cpd_with_tolerance(detected_changepoints, ground_truth, series_length, tolerance=10):
    """
    Evaluate change point detection performance using precision, recall, and F1 score.

    Parameters:
    ----------
    detected_changepoints : list
        List of change points detected by the model.
    ground_truth : list
        List of true change points.
    tolerance : int, optional (default=10)
        Maximum allowable deviation from the true change point for a detection to be
        considered correct.

    Returns:
    -------
    tuple
        Precision, recall, and F1 score as floats and the amounts of true positives and
        true negatives as integers.
    """
    tp = 0  # True positives
    fn = 0  # False negatives
    matched_predictions = set()

    for gt in ground_truth:
        if any(abs(gt - det) <= \
            tolerance and det not in matched_predictions for det in detected_changepoints):
            tp += 1
            matched_predictions.add(
                next(det for det in detected_changepoints if abs(gt - det) <= tolerance))
        else:
            fn += 1

    # False positives
    fp = sum(
        1 for det in detected_changepoints if all(abs(det - gt) > tolerance for gt in ground_truth)
    )

    # True Negatives
    relevant_points = len(detected_changepoints) + len(ground_truth)
    tn = series_length - relevant_points + tp - fp - fn

    precision = tp / len(detected_changepoints) if detected_changepoints else 0
    recall = tp / len(ground_truth) if ground_truth else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall else 0
    accuracy = (tp + tn) / series_length if series_length > 0 else 0

    return precision, recall, f1, accuracy, tp, fn, fp, tn


def calculate_segments(data, changepoints):
    """"
    Calculate the segments between each pair of change points.

    Parameters:
    ----------
    data : np.ndarray or pd.DataFrame
        Input data with each column representing a feature.
    changepoints : list
        List of detected changepoints.

    Returns:
    list
        The input data divided into segments.
    """
    segments = []
    for i in range(len(changepoints) + 1):
        if i == 0:
            segment = data[:changepoints[i]]
        elif i == len(changepoints):
            segment = data[changepoints[i-1]:]
        else:
            segment = data[changepoints[i-1]:changepoints[i]]

        if len(segment) > 1:
            segments.append(segment)
    
    return segments
