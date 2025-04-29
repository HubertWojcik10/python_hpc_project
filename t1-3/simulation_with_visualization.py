"""
    Timed simulation that helps us assess the time to process all floorplans (Task 2).
    Added visualization capability for temperature distributions.
"""

from os.path import join
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
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

def visualize_temperature(u, interior_mask, bid, save_dir='./plots'):
    # create a masked array for the interior
    temperature = np.ma.masked_array(
        u[1:-1, 1:-1], 
        mask=~interior_mask
    )
    
    # create a custom colormap from blue to red (cold to hot)
    colors = [(0, 0, 0.8), (0.5, 0.5, 1), (1, 1, 1), (1, 0.5, 0.5), (0.8, 0, 0)]
    cmap = LinearSegmentedColormap.from_list('temp_cmap', colors, N=256)
    
    plt.figure(figsize=(10, 8))
    img = plt.imshow(temperature, cmap=cmap, vmin=10, vmax=25)
    plt.colorbar(img, label='Temperature (°C)')
    plt.title(f'Building {bid} - Temperature Distribution')
    plt.tight_layout()
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, f'{bid}_temperature.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save plot for {bid}: {e}")

if __name__ == '__main__':
    # Load data
    start_time = time.time()
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
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
        
        # Visualize the result for this floorplan
        visualize_temperature(u, all_interior_mask[i], building_ids[i])

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, ", ", ".join(str(stats[k]) for k in stat_keys))

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f"TOTAL TIME OF EXECUTION FOR N={N}: {total_time}")

    # Create a summary visualization if multiple buildings
    if N > 1:
        fig, axes = plt.subplots(1, min(N, 3), figsize=(15, 5))
        if N == 1:  # Handle case with just one building
            axes = [axes]
            
        for i in range(min(N, 3)):  # Show at most 3 buildings in summary
            temperature = np.ma.masked_array(
                all_u[i][1:-1, 1:-1], 
                mask=~all_interior_mask[i]
            )
            
            colors = [(0, 0, 0.8), (0.5, 0.5, 1), (1, 1, 1), (1, 0.5, 0.5), (0.8, 0, 0)]
            cmap = LinearSegmentedColormap.from_list('temp_cmap', colors, N=256)
            
            im = axes[i].imshow(temperature, cmap=cmap, vmin=10, vmax=25)
            axes[i].set_title(f'Building {building_ids[i]}')
            axes[i].axis('off')
            
        plt.colorbar(im, ax=axes, label='Temperature (°C)', shrink=0.8)
        plt.suptitle(f'Temperature Distribution - {N} Buildings')
        plt.tight_layout()
        plt.savefig('./plots/summary_temperature.png', dpi=150)
        plt.close()