import numpy as np
from scipy.special import xlogy
from sklearn.neighbors import NearestNeighbors
    

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

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    dists, neighs = nn.kneighbors()
    
    neighs_classes = np.take(classes, neighs)

    neighs_dists = (
        neighs_classes.reshape(-1, n_neighbors, 1) == np.arange(n_classes).reshape(1, 1, -1)
    ).mean(axis=1)

    # Miller correction (in bits)
    corr = (n_classes - 1) / 2 / n_neighbors / np.log(2) if correction else 0
    cond_entropies_naive = mplog2p(neighs_dists).sum(axis=-1)
    cond_entropies = cond_entropies_naive + corr
    cond_entropy = cond_entropies.mean()

    return cond_entropy
