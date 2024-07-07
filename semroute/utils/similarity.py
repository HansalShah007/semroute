import numpy as np

def cosine_similarity(vec1, vec2) -> float:
    """
    Calculate the cosine similarity between two numpy arrays.

    Parameters:
    vec1 (np.ndarray): First vector.
    vec2 (np.ndarray): Second vector.

    Returns:
    float: Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)