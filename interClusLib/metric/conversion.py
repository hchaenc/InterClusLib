import numpy as np

def sim_to_dist(sim_value, mode="1_minus"):
    """
    Convert similarity to distance.
    
    Parameters:
    -----------
    sim_value : float or numpy.ndarray
        Similarity value(s) to convert
    mode : str, default="1_minus"
        Conversion mode, currently only supports "1_minus"
        
    Returns:
    --------
    float or numpy.ndarray
        Converted distance value(s)
    """
    if mode == "1_minus":
        # Handle different input types
        if isinstance(sim_value, list):
            sim_value = np.array(sim_value)
        return 1 - sim_value
    else:
        raise ValueError(f"Unsupported conversion mode: {mode}")

def dist_to_sim(dist_value, mode="1_over_1_plus"):
    """
    Convert distance to similarity.
    
    Parameters:
    -----------
    dist_value : float or numpy.ndarray
        Distance value(s) to convert
    mode : str, default="1_over_1_plus"
        Conversion mode, currently only supports "1_over_1_plus"
        
    Returns:
    --------
    float or numpy.ndarray
        Converted similarity value(s)
    """
    if mode == "1_over_1_plus":
        # Handle different input types
        if isinstance(dist_value, list):
            dist_value = np.array(dist_value)
        return 1 / (1 + dist_value)
    else:
        raise ValueError(f"Unsupported conversion mode: {mode}")