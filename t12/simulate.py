from os.path import join
import sys
import numpy as np
from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

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
    """
    u_copy = np.copy(u)
    
    # move data to the gpu
    u_in = cuda.to_device(u_copy)
    u_out = cuda.to_device(u_copy)
    d_interior_mask = cuda.to_device(interior_mask)
    
    # define threads, block and grid dimensions
    TPB = (16, 16)
    grid_dim = (
        (u.shape[0] + TPB[0] - 1) // TPB[0],
        (u.shape[1] + TPB[1] - 1) // TPB[1]
    )
    
    # perform iterations
    for _ in range(max_iter):
        # launch kernel
        jacobi_kernel[grid_dim, TPB](u_in, u_out, d_interior_mask)
        
        # swap input and output for next iteration (part of Jacoby's method)
        u_in, u_out = u_out, u_in
    
    # copy the results back to cpu
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
    # Load data
    LOAD_DIR = '../data/modified_swiss_dwellings/'
    
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, ", ", ".join(str(stats[k]) for k in stat_keys))