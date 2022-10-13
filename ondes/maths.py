import numpy as np

def compress(data, threshold, level):
    """for an effective compression, threshold must be smaller than level. Knee is at (threshold, level).
    """
    return np.where(
        np.abs(data) < threshold,
        data * level / threshold,
        (data + (-1) * np.sign(data) * threshold) * (1 - level) / (1 - threshold) + np.sign(data) * level)

def expand(data, threshold, log_level):
    return np.where(np.abs(data) < threshold, data / 10**log_level, (data / 10**log_level + (data + (-np.sign(data)) * threshold)))
