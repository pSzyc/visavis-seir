import numpy as np
from scipy.special import xlogy
from sklearn.neighbors import NearestNeighbors
from warnings import warn

from pathlib import Path
import sys
root_repo_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_repo_dir)) # in order to be able to import from scripts.py

from scripts.defaults import PARAMETERS_DEFAULT

def xlog2x(x):
    return xlogy(x, x) / np.log(2)


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
    cond_entropies_naive = -xlog2x(neighs_dists).sum(axis=-1)
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

    cond_entropy = -xlog2x(confusion_matrix).sum().sum() - mplog2p(confusion_matrix.sum(axis=0)).sum()

    return cond_entropy



def conditional_entropy_discrete_bins_or_neighbors_pandas(
    data, 
    xfield,
    yfields,
    classes=None,
    n_neighbors=10,
    correction=True,
):
    "estimates H(class | point)"

    if classes == None:
        classes = data[xfield].unique().sort_values()

    n_samples = len(data)
    assert n_samples > 0

    n_occurrences = data.groupby([xfield] + yfields).size().unstack(xfield).fillna(0).reindex(columns=classes)
    n_occurrences_tot = n_occurrences.sum(axis=1)
    n_occurrences_tot.name = 'n_occurrences_tot'
    multiple_duplicates = n_occurrences[n_occurrences_tot >= n_neighbors]
    is_multiple_duplicate = data.join(n_occurrences_tot, on=yfields)['n_occurrences_tot'] >= n_neighbors

    # For ys with at least n_neighbors samples with equal y, compute conditional entropy based on occurrence count
    correction_for_multiple_duplicates = (((multiple_duplicates > 0).sum(axis=1) - 1) / 2 / np.log(2)).sum() if correction else 0
    cond_entropy_for_multiple_duplicates = (-xlog2x(multiple_duplicates).sum(axis=1) + xlog2x(multiple_duplicates.sum(axis=1))).sum() + correction_for_multiple_duplicates

    # For ys with less than n_neighbors samples with equal y, estimate conditional entropy based on neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(data[yfields].to_numpy())
    neighs = nn.kneighbors(data[~is_multiple_duplicate][yfields].to_numpy(), return_distance=False)

    neigh_labels = np.take(data[xfield].to_numpy(), neighs)

    conditional_distros = (
        neigh_labels.reshape(-1, n_neighbors, 1) == np.array(classes).reshape(1, 1, -1)
    ).mean(axis=1)

    neighs_n_classes_observed = (conditional_distros > 0).sum(axis=-1)

    # Miller correction (in bits)
    corr = (neighs_n_classes_observed - 1) / 2 / n_neighbors / np.log(2) if correction else 0
    cond_entropies_naive = -xlog2x(conditional_distros).sum(axis=-1)
    cond_entropies = cond_entropies_naive + corr
    cond_entropy_for_nonduplicates = cond_entropies.sum()

    print(cond_entropy_for_multiple_duplicates/n_samples, cond_entropy_for_nonduplicates/n_samples)
    conditional_entropy = (cond_entropy_for_multiple_duplicates + cond_entropy_for_nonduplicates) / n_samples

    return conditional_entropy



def get_cycle_time(parameters=PARAMETERS_DEFAULT):
    return (
        parameters['e_subcompartments_count'] / parameters['e_forward_rate']
        + parameters['i_subcompartments_count'] / parameters['i_forward_rate']
        + parameters['r_subcompartments_count'] / parameters['r_forward_rate']
        + 1/3 / parameters['c_rate']
    )

def get_cycle_time_std(parameters=PARAMETERS_DEFAULT):
    return np.sqrt(
        parameters['e_subcompartments_count'] / parameters['e_forward_rate']**2
        + parameters['i_subcompartments_count'] / parameters['i_forward_rate']**2
        + parameters['r_subcompartments_count'] / parameters['r_forward_rate']**2
        + 1/3**2 / parameters['c_rate']**2
    )

def get_efficiency_from_extinction_probab(p):
    return 1/2 + xlog2x(p/2) - xlog2x((p+1)/2)



