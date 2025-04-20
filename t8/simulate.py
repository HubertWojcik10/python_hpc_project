"""
    Simulate module for ex 8 - CUDA implementation.
"""

from os.path import join
import sys
import time
import numpy as np
from numba import cuda

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

# CUDA kernel for a single Jacobi iteration
@cuda.jit
def jacobi_kernel(u_in, u_out, interior_mask):
    """CUDA kernel that performs a single iteration of the Jacobi method."""
    # Get thread indices
    i, j = cuda.grid(2)
    
    # Check if the thread is within the grid bounds and it's an interior point
    if (i > 0 and i < u_in.shape[0] - 1 and 
        j > 0 and j < u_in.shape[1] - 1 and 
        interior_mask[i-1, j-1]):
        
        # Compute new value as average of neighbors
        u_out[i, j] = 0.25 * (
            u_in[i, j-1] +    # left
            u_in[i, j+1] +    # right
            u_in[i-1, j] +    # up
            u_in[i+1, j]      # down
        )
    elif i < u_in.shape[0] and j < u_in.shape[1]:
        # For non-interior points, keep the original value
        u_out[i, j] = u_in[i, j]

def jacobi_cuda(u, interior_mask, max_iter, atol=None):
    """
    Helper function that uses CUDA for Jacobi iterations.
    Note: early stopping is not implemented, runs for a fixed number of iterations.
    """
    # Create copies for the GPU
    u_copy = np.copy(u)
    
    # Move data to the GPU
    u_in = cuda.to_device(u_copy)
    u_out = cuda.to_device(u_copy)
    d_interior_mask = cuda.to_device(interior_mask)
    
    # Define grid and block dimensions for efficient GPU utilization
    # Adjust these values based on your GPU capabilities
    block_dim = (16, 16)
    grid_dim = (
        (u.shape[0] + block_dim[0] - 1) // block_dim[0],
        (u.shape[1] + block_dim[1] - 1) // block_dim[1]
    )
    
    # Perform iterations
    for _ in range(max_iter):
        # Launch kernel for one iteration
        jacobi_kernel[grid_dim, block_dim](u_in, u_out, d_interior_mask)
        
        # Swap input and output for next iteration
        u_in, u_out = u_out, u_in
    
    # Copy result back to host (CPU)
    # Since we swapped u_in and u_out, the final result is in u_in
    result = u_in.copy_to_host()
    
    return result

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
    
    # Original implementation on the small subset
    start_time = time.time()
    all_u_original_subset = []
    for i in range(subset_size):
        u = jacobi(all_u0[i], all_interior_mask[i], MAX_ITER, ABS_TOL)
        all_u_original_subset.append(u)
    original_time = time.time() - start_time
    
    # CUDA implementation on the small subset
    # Warm up CUDA
    _ = jacobi_cuda(all_u0[0], all_interior_mask[0], MAX_ITER)
    
    start_time = time.time()
    all_u_cuda_subset = []
    for i in range(subset_size):
        u = jacobi_cuda(all_u0[i], all_interior_mask[i], MAX_ITER)
        all_u_cuda_subset.append(u)
    cuda_time = time.time() - start_time
    
    # print comparison results
    print("\n--- Performance Comparison ---")
    print(f"Original implementation: {original_time:.4f} seconds")
    print(f"CUDA implementation: {cuda_time:.4f} seconds")
    print(f"Speedup: {original_time/cuda_time:.2f}x faster")
    
    # check if results match (with some tolerance due to floating point differences)
    results_match = all(np.allclose(all_u_original_subset[i], all_u_cuda_subset[i], rtol=1e-3, atol=1e-3) 
                        for i in range(subset_size))
    print(f"Results match: {results_match}")

    # estimate time for N floorplans 
    estimated_full_time = cuda_time * (N / subset_size)
    print(f"\nEstimated time for all {N} floorplans: {estimated_full_time:.2f} seconds")
    
    # estimate time for all floorplans "(c) How long would it now take to process all floorplans?"
    estimated_full_time = cuda_time * (TOTAL_SIZE / subset_size)
    print(f"Estimated time for all {TOTAL_SIZE} floorplans: {estimated_full_time:.2f} seconds")
    
    # N floorplans with CUDA
    print("\nProcessing N floorplans with CUDA implementation...")
    start_time = time.time()
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        all_u[i] = jacobi_cuda(u0, interior_mask, MAX_ITER)
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.4f} seconds")