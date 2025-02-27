def sim_to_dist(sim_value, mode="1_minus"):
    if mode == "1_minus":
        return 1 - sim_value
    else:
        raise ValueError(f"Unsupported conversion mode: {mode}")

def dist_to_sim(dist_value, mode="1_over_1_plus"):
    if mode == "1_over_1_plus":
        return 1 / (1 + dist_value)
    else:
        raise ValueError(f"Unsupported conversion mode: {mode}")