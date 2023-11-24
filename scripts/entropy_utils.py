import numpy as np
from scipy.special import xlogy
from sklearn.neighbors import NearestNeighbors
from warnings import warn
    

def mplog2p(x):
    return -xlogy(x, x) / np.log(2)


def bient(x):
    return mplog2p(x) + mplog2p(1-x)


def conditional_entropy_discrete(
    classes,
    points,
    n_classes=2,
    n_neighbors=10,
    correction=True,
):
    "estimates H(class | point)"

    assert len(classes.shape) == 1
    assert len(points.shape) == 2
    assert classes.shape[0] == points.shape[0]
    assert set(np.unique(classes)) <= set(range(n_classes))
    if n_neighbors >= len(points):
        warn(f"Expected n_neighbors < n_samples, found n_samples = {len(points)}, {n_neighbors = }. Falling back to n_neighbors = {len(points) - 1}")
        n_neighbors = len(points) - 1

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    neighs = nn.kneighbors(return_distance=False)
    
    neighs_classes = np.take(classes, neighs)

    neighs_dists = (
        neighs_classes.reshape(-1, n_neighbors, 1) == np.arange(n_classes).reshape(1, 1, -1)
    ).mean(axis=1)

    neighs_n_classes_observed = (neighs_dists > 0).sum(axis=-1)

    # Miller correction (in bits)
    corr = (neighs_n_classes_observed - 1) / 2 / n_neighbors / np.log(2) if correction else 0
    cond_entropies_naive = mplog2p(neighs_dists).sum(axis=-1)
    cond_entropies = cond_entropies_naive + corr
    cond_entropy = cond_entropies.mean()

    return cond_entropy


def conditional_entropy_discrete_reconstruction(
    classes: np.ndarray,
    points: np.ndarray,
    n_classes=2,
    n_neighbors=11,
):
    "estimates H(class | point)"

    assert len(classes.shape) == 1
    assert len(points.shape) == 2
    assert classes.shape[0] == points.shape[0]
    assert set(np.unique(classes)) <= set(range(n_classes))
    assert n_neighbors % 2, "Number of neighbors must be odd"
    if n_neighbors >= len(points):
        warn(f"Expected n_neighbors < n_samples, found n_samples = {len(points)}, {n_neighbors = }. Falling back to n_neighbors = {len(points) - 1}")
        n_neighbors = len(points) - 1

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    neighs = nn.kneighbors(return_distance=False)
    neighs_classes = np.take(classes, neighs)

    neighs_dists = (
        neighs_classes.reshape(-1, n_neighbors, 1) == np.arange(n_classes).reshape(1, 1, -1)
    ).sum(axis=1)
    is_mle = (neighs_dists == neighs_dists.max(axis=-1).reshape(-1, 1))
    is_mle_ties_normalized = is_mle / is_mle.sum(axis=-1).reshape(-1, 1)

    confusion_matrix = (
        (classes.reshape(-1, 1) == np.arange(n_classes).reshape(1, -1)).reshape(-1, n_classes, 1).repeat(n_classes, axis=2) 
        * is_mle_ties_normalized.reshape(-1, 1, n_classes).repeat(n_classes, axis=1)
    ).mean(axis=0)

    cond_entropy = mplog2p(confusion_matrix).sum().sum() - mplog2p(confusion_matrix.sum(axis=0)).sum()

    return cond_entropy


