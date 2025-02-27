import numpy as np

# ========== 1. Single-Dimension Interval Similarity Measures ==========
def jaccard_similarity(interval1, interval2):
    """
    Calculate the Jaccard similarity between two interval values.

    Parameters:
        interval1 (np.array): Interval values 1
        interval2 (np.array): Interval values 2

    Returns:
        float: Jaccard similarity between the two interval values.
    """
    # calculate the intersection of the two intervals
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    # calculate the union of the two intervals
    if intersection > 0:
        union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
    else:
        union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0])
    # calculate the Jaccard similarity
    jaccard = intersection / union if union > 0 else 0

    return jaccard

def dice_similarity(interval1,interval2):
    """
    Calculate the Dice similarity between two interval values.

    Parameters:
        interval1 (np.array): Interval values 1
        interval2 (np.array): Interval values 2

    Returns:
        float: Dice similarity between the two interval values.
    """
    # calculate the intersection of the two intervals
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    # calculate the sum of the two intervals
    sum_intervals = interval1[1] - interval1[0] + interval2[1] - interval2[0]
    # calculate the Dice similarity
    dice = 2 * intersection / sum_intervals

    return dice

def bidirectional_similarity_min(interval1, interval2):
    """
    Calculate the bidirectional subset similarity between two interval values.

    Parameters:
        interval1 (np.array): Interval values 1
        interval2 (np.array): Interval values 2
        option (str): Option to specify the type of bidirectional subset similarity to calculate.
                        1. 'minimum': Minimum bidirectional subset similarity.
                        2. 'product': Product bidirectional subset similarity.
        
    Returns:
        float: Bidirectional subset similarity between the two interval values.
    """
    # calculate the intersection of the two intervals
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    # calculate the non-overlapping regions of the intervals
    non_overlap_a_b = max(0, (interval1[1] - interval1[0]) - intersection)
    non_overlap_b_a = max(0, (interval2[1] - interval2[0]) - intersection)

    # calculate the reciprocal subsethood
    reciprocal_subsethood_a_b = intersection / (intersection + non_overlap_a_b)
    reciprocal_subsethood_b_a = intersection / (intersection + non_overlap_b_a)

    bidirectional_subset_similarity = min(reciprocal_subsethood_a_b, reciprocal_subsethood_b_a)
    
    return bidirectional_subset_similarity

def bidirectional_similarity_prod(interval1, interval2):
    """
    Calculate the bidirectional subset similarity between two interval values.

    Parameters:
        interval1 (np.array): Interval values 1
        interval2 (np.array): Interval values 2
        option (str): Option to specify the type of bidirectional subset similarity to calculate.
                        1. 'minimum': Minimum bidirectional subset similarity.
                        2. 'product': Product bidirectional subset similarity.
        
    Returns:
        float: Bidirectional subset similarity between the two interval values.
    """
    # calculate the intersection of the two intervals
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    # calculate the non-overlapping regions of the intervals
    non_overlap_a_b = max(0, (interval1[1] - interval1[0]) - intersection)
    non_overlap_b_a = max(0, (interval2[1] - interval2[0]) - intersection)

    # calculate the reciprocal subsethood
    reciprocal_subsethood_a_b = intersection / (intersection + non_overlap_a_b)
    reciprocal_subsethood_b_a = intersection / (intersection + non_overlap_b_a)

    bidirectional_subset_similarity = reciprocal_subsethood_a_b * reciprocal_subsethood_b_a
    
    return bidirectional_subset_similarity

def marginal_similarity(interval1, interval2):
    """
    Calculate the generalized Jaccard similarity between two interval values.

    Parameters:
        interval1 (np.array): Interval values 1
        interval2 (np.array): Interval values 2

    Returns:
        float: Generalized Jaccard similarity between the two interval values.
    """
    # calculate the intersection of the two intervals
    intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
    # calculate the union of the two intervals
    union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
    # calculate the distance between the two intervals
    distance = max(0, max(interval1[0], interval2[0]) - min(interval1[1], interval2[1]))
    # calculate the domain of the two intervals
    domain = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])

    marginal_similarity = 0.5 * (intersection / union + 1 - distance / domain)

    return marginal_similarity

# ========== 3. Multi-Dimension Interval Measures ==========
def jaccard_sim_md(interval_a, interval_b):
    """
    Multi-dimensional jaccard similarity.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    sim = []
    for d in range(n_dims):
        sim_1d = jaccard_similarity(interval_a[d], interval_b[d])
        sim.append(sim_1d)

    return np.mean(sim)

def dice_sim_md(interval_a, interval_b):
    """
    Multi-dimensional dice similarity.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    sim = []
    for d in range(n_dims):
        sim_1d = dice_similarity(interval_a[d], interval_b[d])
        sim.append(sim_1d)

    return np.mean(sim)

def bidirectional_sim_min_md(interval_a, interval_b):
    """
    Multi-dimensional bidrectional subset similarity.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    sim = []
    for d in range(n_dims):
        sim_1d = bidrectional_similarity_min(interval_a[d], interval_b[d])
        sim.append(sim_1d)

    return np.mean(sim)

def bidirectional_sim_prod_md(interval_a, interval_b):
    """
    Multi-dimensional bidrectional subset similarity.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    sim = []
    for d in range(n_dims):
        sim_1d = bidrectional_similarity_prod(interval_a[d], interval_b[d])
        sim.append(sim_1d)

    return np.mean(sim)

def marginal_sim_md(interval_a, interval_b):
    """
    Multi-dimensional generalized jaccard similarity.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    sim = []
    for d in range(n_dims):
        sim_1d = marginal_similarity(interval_a[d], interval_b[d])
        sim.append(sim_1d)
    
    return np.mean(sim)

SIMILARITY_FUNCTIONS = {
    "jaccard": jaccard_similarity,
    "dice": dice_similarity,
    "bidirectional_min": bidirectional_similarity_min,
    "bidirectional_prod": bidirectional_similarity_prod,
    "marginal": marginal_similarity,
}

MULTI_SIMILARITY_FUNCTIONS = {
    "jaccard": jaccard_sim_md,
    "dice": dice_sim_md,
    "bidirectional_min": bidirectional_sim_min_md,
    "bidirectional_prod": bidirectional_sim_prod_md,
    "marginal": marginal_sim_md,
}

