import numpy as np


def load_image_vector(path):
    rng = np.random.default_rng(abs(hash(path)) % (2**32))
    vector = rng.normal(size=300)
    return vector


def embed_pair(left_path, right_path):
    left_vector = load_image_vector(left_path)
    right_vector = load_image_vector(right_path)
    return left_vector, right_vector
