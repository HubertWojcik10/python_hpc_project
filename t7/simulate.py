"""
    Simulate module for ex 7.
"""

from os.path import join
import sys
import time
import numpy as np
from numba import jit

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """Original Jacobi iteration implementation using NumPy operations."""
    u = np.copy(u)
    
    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        
        if delta < atol:
            break
    
    return u

@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter, atol=1e-6):
    """Optimized Jacobi iteration implementation using Numba JIT."""
    u_copy = np.copy(u)
    rows, cols = u_copy.shape
    
    for _ in range(max_iter):
        max_delta = 0.0 # maximum change in the iteration
        
        # iterate over the interior points
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if interior_mask[i-1, j-1]:
                    # calc average of the four neighbors
                    u_new = 0.25 * (
                        u_copy[i, j-1] +  # left
                        u_copy[i, j+1] +  # right
                        u_copy[i-1, j] +  # up
                        u_copy[i+1, j]    # down
                    )
                    
                    delta = abs(u_copy[i, j] - u_new)
                    max_delta = max(max_delta, delta)
                    
                    u_copy[i, j] = u_new
        
        if max_delta < atol:
            break
            
    return u_copy

def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

if __name__ == '__main__':
    # configuration
    LOAD_DIR = '../data/modified_swiss_dwellings/'
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    TOTAL_SIZE = len(building_ids)
    
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    
    # load N floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids[:N]):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    
    # small subset "(a) Run and time the new solution for a small subset of floorplans"
    subset_size = min(5, N)
    
    # warm up jit
    _ = jacobi_jit(all_u0[0], all_interior_mask[0], MAX_ITER, ABS_TOL)
    
    # original implementation on the small subset
    start_time = time.time()
    all_u_original_subset = []
    for i in range(subset_size):
        u = jacobi(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL)
        all_u_original_subset.append(u)
    original_time = time.time() - start_time
    
    # numba implementation on the small subset
    start_time = time.time()
    all_u_numba_subset = []
    for i in range(subset_size):
        u = jacobi_jit(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL)
        all_u_numba_subset.append(u)
    numba_time = time.time() - start_time
    
    # print comparison results
    print("\n--- Performance Comparison ---")
    print(f"Original implementation: {original_time:.4f} seconds")
    print(f"Numba JIT implementation: {numba_time:.4f} seconds")
    print(f"Speedup: {original_time/numba_time:.2f}x faster")
    
    # check if results match
    results_match = all(np.allclose(all_u_original_subset[i], all_u_numba_subset[i], rtol=1e-4, atol=1e-4) 
                        for i in range(subset_size))
    print(f"Results match: {results_match}")

    # estimate time for N floorplans 
    estimated_full_time = numba_time * (N / subset_size)
    print(f"\nEstimated time for all {N} floorplans: {estimated_full_time:.2f} seconds")
    
    # estimate time for all floorplans "(c) How long would it now take to process all floorplans?"
    estimated_full_time = numba_time * (TOTAL_SIZE / subset_size)
    print(f"Estimated time for all {TOTAL_SIZE} floorplans: {estimated_full_time:.2f} seconds")
    
    # N floorplans with numba
    print("\nProcessing N floorplans with Numba implementation...")
    start_time = time.time()
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.4f} seconds")