import os
import re
import glob
import numpy as np
import pandas as pd
import pyvista as pv

import pyvista as pv

import os
import re
import glob
import numpy as np
import pandas as pd
import pyvista as pv

def extract_free_surface_2d(vtu_path, tol=1e-4):
    """
    Extracts the free surface profile h(r) by reading 2D mesh boundary 
    and stripping away fixed domain boundaries (r_min, r_max, z_min).
    """
    mesh = pv.read(vtu_path)
    
    # Extract the 1D outer boundary line segment mesh
    boundary = mesh.extract_surface()
    pts = boundary.points  # pts[:, 0] = r, pts[:, 1] = z
    
    r = pts[:, 0]
    z = pts[:, 1]
    
    # Determine physical extent of domain boundaries
    r_min, r_max = np.min(r), np.max(r)
    z_min = np.min(z)
    
    # Filter OUT points that lie on rigid domain boundaries:
    # - Axis of symmetry / rod surface (r close to r_min)
    # - Outer container wall (r close to r_max)
    # - Bottom substrate (z close to z_min)
    
    fs_mask = (
        (r > r_min + tol) & 
        (r < r_max - tol) & 
        (z > z_min + tol)
    )
    
    r_fs = r[fs_mask]
    z_fs = z[fs_mask]
    
    if len(r_fs) == 0:
        raise ValueError("Free surface mask resulted in 0 points. Consider lowering 'tol'.")

    # Group by radial position to handle dual-node boundary points cleanly
    r_rounded = np.round(r_fs, decimals=6)
    unique_r = np.unique(r_rounded)
    
    r_list, h_list = [], []
    for r_val in unique_r:
        mask = (r_rounded == r_val)
        r_list.append(r_val)
        h_list.append(np.max(z_fs[mask]))
        
    r_arr = np.array(r_list)
    h_arr = np.array(h_list)
    
    # Ensure radial ordering from inner radius to outer radius
    sort_idx = np.argsort(r_arr)
    return r_arr[sort_idx], h_arr[sort_idx]

def process_vtu_folder(folder_path, output_csv="free_surface_hr.csv", tol=1e-4):
    file_pattern = os.path.join(folder_path, "*.vtu")
    vtu_files = glob.glob(file_pattern)
    
    if not vtu_files:
        print(f"No VTU files found in '{folder_path}'")
        return

    # Sort files numerically based on index in filename
    def extract_index(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1

    vtu_files = sorted(vtu_files, key=extract_index)
    rows = []

    print(f"Processing {len(vtu_files)} files...")

    for filepath in vtu_files:
        filename = os.path.basename(filepath)
        match = re.search(r'(\d+)', filename)
        file_idx_str = match.group(1) if match else filename

        try:
            r_vals, h_vals = extract_free_surface_2d(filepath, tol=tol)
            
            for r, h in zip(r_vals, h_vals):
                rows.append({
                    "file_index": file_idx_str,
                    "r": r,
                    "h": h
                })
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully exported free surface profile to '{output_csv}'")

if __name__ == "__main__":
    VTU_FOLDER = "./refined_0.2s_relTime_logSpacedOmega_realisticVals_rod_climbing_output/domain"
    OUTPUT_FILE = "free_surface_h_r.csv"
    
    # Set tolerance based on grid spacing near walls (e.g. 1e-4 or 1e-3)
    process_vtu_folder(VTU_FOLDER, OUTPUT_FILE, tol=1e-4)

# if __name__ == "__main__":
#     VTU_FOLDER = "./refined_0.2s_relTime_logSpacedOmega_realisticVals_rod_climbing_output/domain"
#     OUTPUT_FILE = "free_surface_h_r.csv"
    
#     process_vtu_folder(VTU_FOLDER, OUTPUT_FILE)