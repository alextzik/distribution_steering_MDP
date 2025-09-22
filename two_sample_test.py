import numpy as np
import matplotlib.pyplot as plt
import argparse

# ---------------------------------------------------------------------------
# Heuristic half-space distance between two empirical sample sets
# We mimic compute_heur_dist but both arguments are now empirical samples.
# Logic: Draw random half-spaces defined by (q, b). For each half-space h,
# estimate probability mass of each sample set lying in { x : q^T x + b >= 0 }.
# Distance = average absolute difference of these estimated probabilities.
# ---------------------------------------------------------------------------

def heuristic_halfspace_distance(X: np.ndarray, Y: np.ndarray, num_halfspaces: int = 64, rng: np.random.Generator | None = None) -> float:
    """Symmetric heuristic distance based on random half-spaces.

    Parameters
    ----------
    X, Y : array-like shape (n_samples, dim)
        Empirical samples.
    num_halfspaces : int
        Number of random half-spaces used.
    rng : np.random.Generator | None
        Optional randomness source.

    Returns
    -------
    float
        Average absolute difference of half-space empirical probabilities.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays (n_samples, dim)")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Sample dimensions differ: X dim != Y dim")

    n_dim = X.shape[1]

    # Sample uniformly inside the unit sphere in R^d:
    # 1. Sample directions from standard normal and normalize
    dirs = rng.normal(size=(num_halfspaces, n_dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    Q = dirs

    # For each direction q, create 100 threshold values t spanning [min(q^T X), max(q^T Y)].
    # For each (q, t) we consider half-space {x : q^T x >= t}. Equivalent to q^T x + b >= 0 with b = -t.
    num_thresholds = 100
    diffs = []
    # Loop over directions (num_halfspaces expected to be moderate; if large, vectorization can be added)
    for q in Q:
        proj_X = X @ q  # shape (n_X,)
        proj_Y = Y @ q  # shape (n_Y,)
        min_q = proj_X.min()
        max_q = proj_X.max()
        # Handle degenerate case where max_q < min_q (swap) or they are equal.
        t_vals = np.linspace(min_q, max_q, num_thresholds)
        # Broadcast to compute indicator matrices: (n_samples, num_thresholds)
        probs_X = (proj_X[:, None] >= t_vals[None, :]).mean(axis=0)
        probs_Y = (proj_Y[:, None] >= t_vals[None, :]).mean(axis=0)
        diffs.append(np.abs(probs_X - probs_Y))
    if not diffs:
        return 0.0
    diffs = np.concatenate(diffs)  # flatten all (num_thresholds * num_used_q,)
    return diffs.mean()


def simulate_distances(n_samples: int = 100, dim: int = 2, n_trials: int = 500, num_halfspaces: int = 64,
                       mean_alt: float = 1.0, var_alt=1.0, seed: int | None = None):
    rng = np.random.default_rng(seed)
    null_distances = np.empty(n_trials)
    alt_distances = np.empty(n_trials)
    for i in range(n_trials):
        # Null hypothesis
        X_null = rng.normal(0, 1, size=(n_samples, dim))
        Y_null = rng.normal(0, 1, size=(n_samples, dim))
        null_distances[i] = heuristic_halfspace_distance(X_null, Y_null, num_halfspaces, rng)
        # Alternative
        X_alt = rng.normal(0, 1, size=(n_samples, dim))
        Y_alt = rng.normal(mean_alt, np.sqrt(var_alt), size=(n_samples, dim))
        alt_distances[i] = heuristic_halfspace_distance(X_alt, Y_alt, num_halfspaces, rng)
    return null_distances, alt_distances


def plot_distributions(null_distances: np.ndarray, alt_distances: np.ndarray, bins: int = 40, show: bool = True, save_path: str | None = None):
    plt.figure(figsize=(7,4))
    # increae font size everywhere in the plot
    plt.rcParams.update({'font.size': 20, 'font.family': 'Times New Roman'})
    plt.hist(null_distances, bins=bins, alpha=0.6, label='Null', density=True)
    plt.hist(alt_distances, bins=bins, alpha=0.6, label='Alternative', density=True)
    plt.axvline(null_distances.mean(), color='blue', linestyle='--', linewidth=1)
    plt.axvline(alt_distances.mean(), color='orange', linestyle='--', linewidth=1)
    plt.xlabel('Approximate distance')
    plt.ylabel('Density')
    # plt.title('Heuristic Half-Space Distance Distributions')
    plt.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def empirical_test_threshold(null_distances: np.ndarray, alpha: float = 0.05):
    thresh = np.quantile(null_distances, 1 - alpha)
    return thresh


def compute_power(null_distances: np.ndarray, alt_distances: np.ndarray, alpha: float = 0.05):
    thresh = empirical_test_threshold(null_distances, alpha)
    power = (alt_distances > thresh).mean()
    return thresh, power


def parse_args():
    parser = argparse.ArgumentParser(description='Two-sample test using heuristic half-space distance.')
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--num-halfspaces', type=int, default=300)
    parser.add_argument('--mean_alt', type=float, default=0.2, help='Mean shift under alternative (added to all dims).')
    parser.add_argument('--var_alt', type=float, default=1., help='Variance under alternative.')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-show', action='store_true', help='Do not display plot.')
    parser.add_argument('--save', type=str, default="mean_big_n.pdf", help='Optional path to save histogram figure.')
    return parser.parse_args()


args = parse_args()
null_d, alt_d = simulate_distances(n_samples=args.n_samples, dim=args.dim, n_trials=args.n_trials,
                                    num_halfspaces=args.num_halfspaces, mean_alt=args.mean_alt, var_alt=args.var_alt, seed=args.seed)
thresh, power = compute_power(null_d, alt_d, alpha=args.alpha)
print(f'Alpha={args.alpha:.3f} threshold={thresh:.4f} | Alt mean={alt_d.mean():.4f} | Power≈{power:.3f}')
plot_distributions(null_d, alt_d, show=not args.no_show, save_path=args.save)

