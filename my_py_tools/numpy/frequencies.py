import numpy as np


# Input: 1D list or numpy array
def get_frequencies(data, return_unique_and_counts=False):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError('The input data should be 1D list or numpy array.')
    (unique, counts) = np.unique(data, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    if return_unique_and_counts:
        return frequencies, unique, counts
    else:
        return frequencies
