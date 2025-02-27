import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_changes_for_every_feature(data, changepoints, title, cp_color="orange"):
    """Plot change points for every feature."""
    # Number of features to plot separately
    num_features = len(data.T)

    # Define individual subplot height and width
    single_subplot_width = 18
    single_subplot_height = 2

    # Create a new figure with one subplot per feature
    fig, axes = plt.subplots(num_features, 1,
                             figsize=(single_subplot_width, single_subplot_height * num_features),
                             sharex=True, sharey=False)

    # Plot each feature on its own row
    for i in range(0, num_features):
        ax = axes[i]
        ax.plot(data.to_numpy().T[i])
        ax.set_ylabel(f'Feature {i}')

        # Add changepoints to the plot as vertical dashed lines
        for cp in changepoints:
            ax.axvline(x=cp, color=cp_color, linestyle='--', linewidth=1.5)

    # Add title and layout adjustments
    plt.suptitle(f"{title}")
    plt.tight_layout()
    plt.show()


def plot_changes_for_every_cluster(data, changepoints, title, n_clusters=5, cp_color="orange"):
    """Cluster features and plot aggregated data for each cluster in separate subplots."""
    # Compute the correlation matrix
    corr_matrix = data.corr()

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(corr_matrix)

    # Create a DataFrame to map feature names to their clusters
    feature_clusters = pd.DataFrame({"Feature": data.columns, "Cluster": cluster_labels})

    # Create subplots for each cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 3 * n_clusters), sharex=True, dpi=300)
    if n_clusters == 1:
        axes = [axes]  # Ensure axes is iterable when n_clusters == 1

    for cluster in range(n_clusters):
        # Get features in the current cluster
        cluster_features = feature_clusters[feature_clusters["Cluster"] == cluster]["Feature"]

        # Aggregate features in the cluster (e.g., take the mean)
        aggregated_cluster_data = data[cluster_features].mean(axis=1)

        # Plot the aggregated cluster data
        ax = axes[cluster]
        ax.plot(
            aggregated_cluster_data,
            label=f"Mean of {len(cluster_features)} Features",
            linewidth=1.5)
        ax.set_ylabel(f"Cluster {cluster}")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", linewidth=0.5)

        # Add changepoints to the plot as vertical dashed lines
        for cp in changepoints:
            ax.axvline(x=cp, color=cp_color, linestyle='--', linewidth=1.5)

    plt.suptitle(f"{title}")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def segment_data(data, changepoints):
    """Divide the data into segments for plotting."""
    segments = []
    for i in range(len(changepoints) + 1):
        if i == 0:
            # First segment from the start to the first changepoint
            segment = data[:changepoints[i]]
        elif i == len(changepoints):
            # Last segment from the last changepoint to the end
            segment = data[changepoints[i-1]:]
        else:
            # Segments between two changepoints
            segment = data[changepoints[i-1]:changepoints[i]]

        # Make sure that the segment is big enough to compute the covariance matrix
        if len(segment) > 1:
            segments.append(segment)

    return segments


def plot_two_d_vectors(segments):
    """Plot 2 eigenvectors with highest eigenvalues for each segment."""
    # Parameters
    num_cols = 5
    num_rows = math.ceil(len(segments) / num_cols)  # Number of columns in subplot grid

    # Define base subplot size
    subplot_width = 4
    subplot_height = 4

    # Calculate dynamic figsize based on number of rows and columns
    figsize = (num_cols * subplot_width, num_rows * subplot_height)

    # Create figures for scatter plots and vector plots
    fig_vector, axes_vector = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten the 2D arrays of axes to 1D for easy indexing
    axes_vector = axes_vector.flatten()

    # Color map for different colors
    colors = ['r', 'b']

    start = 0
    end = 0

    # Plot each segment
    for idx, segment in enumerate(segments):
        end += len(segment)
        if idx >= len(segments):
            break  # Stop if we exceed the number of plots we want

        # Compute the covariance matrix
        cov_matrix = np.cov(segment, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvectors_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

        # Reduce the dimensionality of the eigenvectors to 2D using PCA
        pca = PCA(n_components=2)
        eigenvectors_2d = pca.fit_transform(eigenvectors_normalized)

        # Plot the vector plot
        ax_vector = axes_vector[idx]
        for i in range(eigenvectors_2d.shape[1]):
            ax_vector.quiver(0, 0, eigenvectors_normalized[0, i], eigenvectors_normalized[1, i], \
                angles='xy', scale_units='xy', scale=1, color=colors[i])

        # Set plot limits to ensure the eigenvectors fit
        ax_vector.set_xlim(-1, 1)
        ax_vector.set_ylim(-1, 1)
        ax_vector.set_aspect('equal', adjustable='box')
        ax_vector.grid(True)
        ax_vector.set_title(f'Data points: {start}-{end}')
        start = end+1

    for i in range(len(segments), len(axes_vector)):
        fig_vector.delaxes(axes_vector[i])  # Remove unused axes

    fig_vector.tight_layout()
    fig_vector.show()
