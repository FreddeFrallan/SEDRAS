import numpy as np
from sklearn.neighbors import NearestNeighbors


class LocallyScaledExponentialError:
    """
    Implements the Locally Scaled Exponential Error (LSEE) metric.

    Formula:
        Score = 1 - exp( - ( |y_true - y_pred| * strictness ) / sigma(y_true) )

    Where sigma(y_true) is the local density (mean distance to k-nearest neighbors).
    """

    def __init__(self, y_train, k_neighbors=5, strictness=3.0, epsilon=1e-6):
        """
        Args:
            y_train (array-like): The training labels used to learn the distribution.
            k_neighbors (int): Number of neighbors to estimate local density (sigma).
            strictness (float): Lambda parameter. Higher = stricter penalties.
            epsilon (float): Small constant to prevent division by zero in dense clusters.
        """
        self.k = k_neighbors
        self.strictness = strictness
        self.epsilon = epsilon

        # Ensure y_train is the correct shape (N, 1) for NearestNeighbors
        self.y_train = np.array(y_train).reshape(-1, 1)

        # Fit the density estimator immediately
        self.nbrs = NearestNeighbors(n_neighbors=self.k).fit(self.y_train)

    def __call__(self, y_true, y_pred):
        """
        Calculate the score for given truth and prediction.
        Can handle single floats or arrays.
        """
        # Ensure inputs are numpy arrays
        y_true = np.atleast_1d(y_true)
        y_pred = np.atleast_1d(y_pred)

        # 1. Estimate Local Scale (Sigma)
        # We query the k-nearest neighbors for the TRUE labels
        # This tells us: "How sparse is the region where this point belongs?"
        distances, _ = self.nbrs.kneighbors(y_true.reshape(-1, 1))

        # Sigma is the mean distance to the k neighbors
        sigma = np.mean(distances, axis=1)

        # Clip sigma to epsilon to avoid division by zero (if points are identical)
        sigma = np.maximum(sigma, self.epsilon)

        # 2. Calculate Absolute Error
        abs_error = np.abs(y_true - y_pred)

        # 3. Apply Bounded Exponential Formula
        # Score approaches 1.0 as error increases
        score = 1.0 - np.exp(- (abs_error * self.strictness) / sigma)

        # Return float if single input, else array
        if score.size == 1:
            return float(score)
        return score


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Setup your training distribution (Mixture of dense and sparse data)
    train_data = np.concatenate([
        np.random.normal(20, 2, 50),  # Dense Bulk (Sigma approx 0.5)
        np.random.normal(150, 15, 30),  # Medium Cluster (Sigma approx 10)
        [400, 550, 700, 900]  # Sparse Tail (Sigma approx 150)
    ])

    # 2. Initialize the Metric
    metric = LocallyScaledExponentialError(train_data, strictness=3.0)

    print(f"{'Scenario':<25} | {'True':<5} | {'Pred':<5} | {'Error':<5} | {'LSEE Score':<10}")
    print("-" * 65)

    # Case A: Dense Region (Strict)
    # Error is 5.0, but density is high. Should be terrible score.
    score_dense = metric(y_true=20, y_pred=25)
    print(f"{'Dense Bulk (Strict)':<25} | {20:<5} | {25:<5} | {5:<5} | {score_dense:.4f}")

    # Case B: Medium Region
    # Error is 5.0 (same absolute error). Should be a moderate score.
    score_med = metric(y_true=150, y_pred=155)
    print(f"{'Medium Cluster':<25} | {150:<5} | {155:<5} | {5:<5} | {score_med:.4f}")

    # Case C: Sparse Tail (Lenient)
    # Error is 5.0 (same absolute error). Should be near zero score.
    score_tail = metric(y_true=550, y_pred=555)
    print(f"{'Sparse Tail (Lenient)':<25} | {550:<5} | {555:<5} | {5:<5} | {score_tail:.4f}")

    # Case D: Sparse Tail Large Error
    # Error is 150. Even in sparse regions, this should eventually result in a high penalty.
    score_tail_bad = metric(y_true=550, y_pred=700)
    print(f"{'Sparse Tail (Big Err)':<25} | {550:<5} | {700:<5} | {150:<5} | {score_tail_bad:.4f}")