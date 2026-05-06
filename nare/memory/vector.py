import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import faiss
import numpy as np
import logging

def make_hnsw_index(dim: int, max_elements: int = 1000) -> faiss.Index:
    """Create an HNSW index for O(log N) approximate nearest neighbour.
    
    M=32 is the connectivity parameter.
    """
    M = 32
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 64
    index.hnsw.efSearch = 32
    return index

def make_quantized_index(dim: int) -> faiss.Index:
    """Create a quantized HNSW index for memory efficiency.

    Note: IndexIVFPQ requires training before use. The index is returned
    untrained - caller must call index.train(data) before adding vectors.
    For immediate use without training data, use make_hnsw_index() instead.
    """
    m = 64
    nbits = 8
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, 100, m, nbits)
    # Index is NOT trained - caller must train before adding vectors
    return index

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Ensure vector is float32, 2D, and L2 normalized."""
    v = np.array(vector, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    faiss.normalize_L2(v)
    return v
