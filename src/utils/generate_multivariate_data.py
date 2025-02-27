import numpy as np
import pandas as pd

def generate_multivariate_signal(lengths, change_points, dimensions=3, seed=42):
    """
    Generate multivariate data with changes in frequency, amplitude, and trend.
    
    Parameters:
        lengths (list): List of segment lengths between change points.
        change_points (list): Indices of change points.
        dimensions (int): Number of dimensions (e.g., 3 for x, y, z).
        seed (int): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Multivariate data with shape (sum(lengths), dimensions).
    """
    np.random.seed(seed)
    total_length = sum(lengths)
    data = np.zeros((total_length, dimensions))

    start = 0
    for i, cp in enumerate(change_points):
        end = start + lengths[i]
        for dim in range(dimensions):
            if i in [0, 1, 3, 4, 6, 7]:
                freq = np.random.uniform(0.1, 1.0)
                amplitude = np.random.uniform(2, 20)
                signal = amplitude * np.sin(np.linspace(0, freq * (end - start), end - start))
                signal = signal + amplitude  # Shift sine wave to positive range
                data[start:end, dim] = signal
                noise_level = amplitude * 0.15
            else:
                trend = np.random.uniform(-0.05, 0.05)
                base = np.linspace(0, (end - start) * trend, end - start)
                data[start:end, dim] = base + 20
                noise_level = np.random.uniform(0.1, 1.0)

            if np.random.rand() < 0.8:
                noise = np.random.normal(0, noise_level, size=(end - start))
                data[start:end, dim] += noise

        start = end

    column_names = ['x', 'y', 'z'][:dimensions]
    df = pd.DataFrame(data, columns=column_names)

    return df


def generate_ar_process(phi, n, noise_std=2):
    """
    Generate data from a first-order autoregressive process AR(1).

    Parameters:
    - phi: Coefficient for AR(1) process.
    - n: Number of time steps.
    - noise_std: Standard deviation of the random noise.
    - x0: Initial value of the process. If None and random_initialization=False, defaults to 0.
    - random_initialization: If True, initializes X[0] randomly.

    Returns:
    - x: Generated AR(1) process.
    """
    x = np.zeros(n)

    # Initialize X[0] randomly, for example from a standard normal distribution
    x[0] = np.random.uniform(0, noise_std)

    # Generate random white noise
    noise = np.random.normal(0, noise_std, n)

    # Generate the AR(1) process
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]

    return x


def create_s_abrupt(seed=1234):
    """
    Create the S-Abrupt synthetic dataset and return it with change points.

    Returns:
    - dataset: The concatenated S-Abrupt dataset.
    - change_points: A list of indices where change points occur.
    """
    np.random.seed(seed)

    phi_values = [0.1, 0.4, 0.6]
    noise_stds = [1.0, 1.5, 1.75]

    segments = [
        generate_ar_process(phi_values[0], 1000, noise_stds[0]),
        generate_ar_process(phi_values[1], 1000, noise_stds[1]),
        generate_ar_process(phi_values[2], 1000, noise_stds[2]),
        generate_ar_process(phi_values[0], 1000, noise_stds[0]),
        generate_ar_process(phi_values[1], 1000, noise_stds[1]),
        generate_ar_process(phi_values[2], 1000, noise_stds[2]),
    ]

    # Concatenate the segments and track the change points
    dataset = np.concatenate(segments)
    change_points = [len(segments[0])]  # First change point after the first segment
    for i in range(1, len(segments) - 1):
        change_points.append(change_points[-1] + len(segments[i]))

    dataset = np.stack([dataset] * 3, axis=1)
    column_names = ['x', 'y', 'z']
    dataset_df = pd.DataFrame(dataset, columns=column_names)

    return dataset_df, change_points


def create_s_gradual(seed=1234):
    """
    Create the S-Gradual synthetic dataset and return it with change points.

    Returns:
    - dataset: The concatenated S-Gradual dataset.
    - change_points: A list of indices where change points occur.
    """
    np.random.seed(seed)

    phi_values = [0.1, 0.4, 0.6]
    noise_stds = [1.0, 1.5, 1.75]

    segments = [
        generate_ar_process(phi_values[0], 800, noise_stds[0]),
        0.5 * (generate_ar_process(phi_values[0], 200, noise_stds[0]) + generate_ar_process(
            phi_values[1],
            200,
            noise_stds[2])),
        generate_ar_process(phi_values[1], 600, noise_stds[2]),
        0.5 * (generate_ar_process(phi_values[1], 200, 2) + generate_ar_process(
            phi_values[2],
            200,
            noise_stds[0])),
        generate_ar_process(phi_values[2], 600, noise_stds[0]),
        0.5 * (generate_ar_process(phi_values[2], 200, 1) + generate_ar_process(
            phi_values[0],
            200,
            noise_stds[0])),
        generate_ar_process(phi_values[0], 600, noise_stds[0]),
        0.5 * (generate_ar_process(phi_values[0], 200, noise_stds[0]) + generate_ar_process(
            phi_values[1],
            200,
            noise_stds[1])),
        generate_ar_process(phi_values[1], 600, noise_stds[2]),
        0.5 * (generate_ar_process(phi_values[1], 200, noise_stds[2]) + generate_ar_process(
            phi_values[2],
            200,
            noise_stds[0])),
        generate_ar_process(phi_values[2], 800, noise_stds[0]),
    ]

    # Concatenate the segments and track the change points
    dataset = np.concatenate(segments)
    change_points = [len(segments[0])]  # First change point after the first segment
    for i in range(1, len(segments) - 1):
        change_points.append(change_points[-1] + len(segments[i]))

    dataset = np.stack([dataset] * 3, axis=1)
    column_names = ['x', 'y', 'z']
    dataset_df = pd.DataFrame(dataset, columns=column_names)

    return dataset_df, change_points
