import numpy as np

def jaccard_similarity(x, y):
    """
    Calculate the Jaccard similarity between two vectors x and y

    Args:
    - x (np.array): The first vector
    - y (np.array): The second vector

    Returns:
    - the Jaccard similarity between two vectors x and y
    """
    # Q1
    # TODO
    inter = np.vstack((x,y))
    j = (np.sum(np.min(inter, axis=0)))/(np.sum(np.max(inter,axis=0)))
    return j

def euclidean_similarity(x, y):
    """
    Calculate the Euclidean similarity between two vectors x and y

    Args:
    - x (np.array): The first vector
    - y (np.array): The second vector

    Returns:
    - the Euclidean similarity between two vectors x and y
    """
    # Q2
    # TODO
    e_similarity = -np.linalg.norm(x-y)
    return e_similarity

def manhattan_similarity(x, y):
    """
    Calculate the Manhattan similarity between two vectors x and y

    Args:
    - x (np.array): The first vector
    - y (np.array): The second vector

    Returns:
    - the Manhattan similarity between two vectors x and y
    """
    # Q3
    # TODO
    man = sum(-abs(val1-val2) for val1, val2 in zip(x,y))
    return man

def cosine_similarity(x, y):
    """
    Calculates the Cosine similarity between two vectors x and y
    
    Args:
    - x (np.array): The first vector
    - y (np.array): The second vector

    Returns:
    - the Cosine similarity between two vectors x and y
    """
    # Q4
    # TODO
    from numpy.linalg import norm
    cos = np.dot(x,y)/(norm(x)*norm(y))
    return(cos)

def compute_classification_accuracy(classification_matrix):
    """
    Computes classification accuracy from a classification matrix
    
    Args:
    - classification matrix (np.array): A classification matrix

    Returns:
    - The classification accuracy for a classification matrix.
    """
    # Q5
    # TODO
    correct = np.sum(np.diag(classification_matrix))
    tot = np.sum(classification_matrix)
    accuracy = correct/tot
    return accuracy
