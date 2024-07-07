import numpy as np
from typing import List

def get_centroid(
    embeddings: np.ndarray
):
    """
    Calculates the centroid of a list of embeddings. The centroid is the average embedding vector.

    Parameters:
    - embeddings (np.ndarray): A numpy array of embedding vectors.

    Returns:
    - np.ndarray: The centroid embedding vector.
    """
    
    centroid = np.zeros(len(embeddings[0]))
    for i in range(len(embeddings)):
        centroid += embeddings[i]
    centroid /= len(embeddings)
    return centroid