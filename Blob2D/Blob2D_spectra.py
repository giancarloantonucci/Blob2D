#!/usr/bin/env python3
"""
Postprocess Blob2D simulation VTK output to compute energy spectra.
Reads PVD files and computes radially averaged spectra for each timestep.
"""

import numpy as np
import os
import glob
from scipy.fft import dct, dctn, fftfreq
from scipy.interpolate import griddata
import argparse
try:
    import pyvista as pv
except ImportError:
    print("ERROR: pyvista required for VTK reading. Install with: pip install pyvista")
    exit(1)


def apply_hann_window_2d(field):
    """Apply 2D Hann window to reduce spectral leakage."""
    ny, nx = field.shape
    hann_x = np.hanning(nx)
    hann_y = np.hanning(ny)
    window_2d = np.outer(hann_y, hann_x)
    return field * window_2d


def compute_spectra_periodic(potential, vorticity, nx, ny, domain_size):
    """Compute spectra for periodic boundary conditions using FFT."""
    # 2D FFTs
    vorticity_fft = np.fft.fft2(vorticity)
    potential_fft = np.fft.fft2(potential)
    
    # Wavenumbers
    kx = fftfreq(nx, d=domain_size/nx) * 2 * np.pi
    ky = fftfreq(ny, d=domain_size/ny) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # Compute E×B velocity components in Fourier space
    # u = E×B/B² where B = B₀ẑ (assumed constant)
    # E = -∇φ, so in Fourier space: E_k = -ik φ_k
    # u_x = E_y/B₀ = -i k_y φ_k/B₀
    # u_y = -E_x/B₀ = i k_x φ_k/B₀
    # (B₀ cancels in the normalised spectrum)
    u_x_fft = -1j * ky_grid * potential_fft
    u_y_fft = 1j * kx_grid * potential_fft
    
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Power spectra (properly normalised using Parseval's theorem)
    norm_factor = (nx * ny)**2
    enstrophy_power = np.abs(vorticity_fft)**2 / norm_factor
    kinetic_power = 0.5 * (np.abs(u_x_fft)**2 + np.abs(u_y_fft)**2) / norm_factor
    
    return k_magnitude, enstrophy_power, kinetic_power


def compute_spectra_dirichlet(vorticity, u_x, u_y, nx, ny, domain_size):
    """Compute spectra for Dirichlet boundary conditions using DCT."""
    # Apply Hann window
    vorticity_windowed = apply_hann_window_2d(vorticity)
    u_x_windowed = apply_hann_window_2d(u_x)
    u_y_windowed = apply_hann_window_2d(u_y)
    
    # 2D Type-II DCT
    vorticity_dct = dctn(vorticity_windowed, type=2, norm='ortho')
    u_x_dct = dctn(u_x_windowed, type=2, norm='ortho')
    u_y_dct = dctn(u_y_windowed, type=2, norm='ortho')
    
    # Wavenumbers for DCT
    kx = np.pi * np.arange(nx) / domain_size
    ky = np.pi * np.arange(ny) / domain_size
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Power spectra
    enstrophy_power = np.abs(vorticity_dct)**2
    kinetic_power = 0.5 * (np.abs(u_x_dct)**2 + np.abs(u_y_dct)**2)
    
    # Correct for window energy loss using mean of squared window
    window = apply_hann_window_2d(np.ones((ny, nx)))
    window_power_correction = np.mean(window**2)
    enstrophy_power /= window_power_correction
    kinetic_power /= window_power_correction
    
    return k_magnitude, enstrophy_power, kinetic_power


def radial_average_spectrum(k_magnitude, power_spectrum, n_bins=None):
    """Compute radially averaged power spectrum."""
    if n_bins is None:
        n_bins = min(k_magnitude.shape) // 4
    
    k_max = np.max(k_magnitude)
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    spectrum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    bin_indices = np.digitize(k_magnitude.ravel(), k_bins) - 1
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        if np.any(mask):
            spectrum[i] = np.mean(power_spectrum.ravel()[mask])
            counts[i] = np.sum(mask)
    
    valid = counts > 0
    return k_centers[valid], spectrum[valid]


def parse_pvd_file(pvd_filename):
    """Parse a PVD file to extract VTU filenames and timestamps."""
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(pvd_filename)
    root = tree.getroot()
    
    files_info = []
    base_dir = os.path.dirname(pvd_filename)
    
    for dataset in root.findall('.//DataSet'):
        vtu_file = dataset.get('file')
        timestep = float(dataset.get('timestep', 0))
        
        # Make path relative to PVD file location
        full_path = os.path.join(base_dir, vtu_file)
        files_info.append((full_path, timestep))
    
    return files_info


def interpolate_to_regular_grid(mesh, field_name, nx, ny, bounds):
    """Interpolate field from unstructured mesh to regular grid."""
    # Extract field data
    field_data = mesh.point_data[field_name]
    
    # Get mesh points
    points = mesh.points[:, :2]  # Only x, y coordinates
    
    # Create regular grid
    x = np.linspace(bounds[0], bounds[1], nx)
    y = np.linspace(bounds[2], bounds[3], ny)
    X, Y = np.meshgrid(x, y)
    
    # Interpolate to regular grid
    field_regular = griddata(points, field_data, (X, Y), method='linear')
    
    # Fill any NaN values (at boundaries) with nearest neighbour
    if np.any(np.isnan(field_regular)):
        mask = np.isnan(field_regular)
        field_regular[mask] = griddata(points, field_data, (X[mask], Y[mask]), method='nearest')
    
    return field_regular


def process_vtk_file(filename, time, boundary_type, domain_size, output_dir):
    """Process a single VTK file to compute spectra."""
    # Read VTK file
    mesh = pv.read(filename)
    
    # Extract field data
    if 'vorticity' not in mesh.point_data:
        print(f"WARNING: No vorticity field in {filename}")
        return None
    
    # Determine grid dimensions from unstructured mesh
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    x_coords = mesh.points[:, 0]
    y_coords = mesh.points[:, 1]
    
    # Find unique x and y coordinates (with tolerance for floating point)
    x_unique = np.unique(np.round(x_coords, decimals=8))
    y_unique = np.unique(np.round(y_coords, decimals=8))
    nx = len(x_unique)
    ny = len(y_unique)
    
    print(f"  Detected grid resolution: {nx} x {ny}")
    
    # Infer domain size if not provided
    if domain_size is None:
        domain_size_x = bounds[1] - bounds[0]
        domain_size_y = bounds[3] - bounds[2]
        # Assume square domain
        domain_size = max(domain_size_x, domain_size_y)
        print(f"  Inferred domain size: {domain_size:.6f}")
    
    # Interpolate fields to regular grid
    vorticity = interpolate_to_regular_grid(mesh, 'vorticity', nx, ny, bounds)
    density = interpolate_to_regular_grid(mesh, 'density', nx, ny, bounds) if 'density' in mesh.point_data else None
    potential = interpolate_to_regular_grid(mesh, 'potential', nx, ny, bounds) if 'potential' in mesh.point_data else None
    
    if potential is None:
        print(f"WARNING: No potential field in {filename}")
        return None
    
    # Compute spectra based on boundary type
    if boundary_type == "periodic":
        # For periodic, compute velocity in Fourier space
        k_magnitude, enstrophy_power, kinetic_power = compute_spectra_periodic(
            potential, vorticity, nx, ny, domain_size)
    else:
        # For Dirichlet, compute E×B velocity using finite differences
        dy = domain_size / (ny - 1)
        dx = domain_size / (nx - 1)
        
        # E×B velocity: u = (E_y, -E_x) where E = -∇φ
        # u_x = E_y = -(∂φ/∂y)
        # u_y = -E_x = -(-∂φ/∂x) = ∂φ/∂x
        grad_phi_y, grad_phi_x = np.gradient(potential, dy, dx)
        u_x = -grad_phi_y
        u_y = grad_phi_x
        
        k_magnitude, enstrophy_power, kinetic_power = compute_spectra_dirichlet(
            vorticity, u_x, u_y, nx, ny, domain_size)
    
    # Set k=0 (DC) mode to zero - it's not relevant for turbulence analysis
    enstrophy_power[0, 0] = 0.0
    kinetic_power[0, 0] = 0.0
    
    # Radial averaging
    k_enstrophy, enstrophy_spectrum = radial_average_spectrum(k_magnitude, enstrophy_power)
    k_kinetic, kinetic_spectrum = radial_average_spectrum(k_magnitude, kinetic_power)
    
    # Extract step from filename
    import re
    step_match = re.search(r'_(\d+)\.', filename)
    step = int(step_match.group(1)) if step_match else 0
    
    # Save spectra
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    header = (f"Blob2D spectra from {basename}\n"
             f"Time = {time:.6f}, Step = {step}\n"
             f"Boundary type: {boundary_type}, Resolution: {nx}x{ny}\n"
             f"k [1/rho_s], spectrum")
    
    np.savetxt(os.path.join(output_dir, f'enstrophy_spectrum_{basename}.txt'),
              np.column_stack([k_enstrophy, enstrophy_spectrum]), header=header)
    np.savetxt(os.path.join(output_dir, f'kinetic_spectrum_{basename}.txt'),
              np.column_stack([k_kinetic, kinetic_spectrum]), header=header)
    
    # Compute integrated quantities using trapezoidal rule
    total_enstrophy = np.trapezoid(enstrophy_spectrum, x=k_enstrophy)
    total_kinetic = np.trapezoid(kinetic_spectrum, x=k_kinetic)
    
    print(f"  Enstrophy = {total_enstrophy:.3e}, KE = {total_kinetic:.3e}")
    
    return {
        'filename': basename,
        'time': time,
        'step': step,
        'total_enstrophy': total_enstrophy,
        'total_kinetic': total_kinetic
    }


def main():
    parser = argparse.ArgumentParser(description='Compute energy spectra from Blob2D VTK output')
    parser.add_argument('pvd_file', help='Input PVD file')
    parser.add_argument('-b', '--boundary', choices=['periodic', 'dirichlet'], 
                       default=None, help='Boundary condition type (auto-detect from filename if not specified)')
    parser.add_argument('-L', '--domain-size', type=float, default=None,
                       help='Domain size in rho_s units (auto-detect if not specified)')
    parser.add_argument('-o', '--output-dir', default='spectra',
                       help='Output directory for spectra files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse PVD file
    print(f"Reading PVD file: {args.pvd_file}")
    files_info = parse_pvd_file(args.pvd_file)
    print(f"Found {len(files_info)} timesteps in PVD file")
    
    # Determine boundary type
    if args.boundary is None:
        # Try to infer from filename
        pvd_basename = os.path.basename(args.pvd_file)
        if 'periodic' in pvd_basename.lower():
            args.boundary = 'periodic'
            print(f"Detected boundary type: periodic (from filename)")
        elif 'dirichlet' in pvd_basename.lower():
            args.boundary = 'dirichlet'
            print(f"Detected boundary type: dirichlet (from filename)")
        else:
            print("ERROR: Cannot infer boundary type from filename. Please specify with -b")
            return 1
    
    print(f"Boundary type: {args.boundary}")
    if args.domain_size is not None:
        print(f"Domain size: {args.domain_size}")
    else:
        print(f"Domain size: auto-detect from mesh")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Process all files
    results = []
    for vtu_file, time in files_info:
        print(f"Processing {os.path.basename(vtu_file)} (t = {time:.3f})")
        
        result = process_vtk_file(vtu_file, time, args.boundary, args.domain_size, args.output_dir)
        if result:
            results.append(result)
    
    # Save summary
    if results:
        summary_file = os.path.join(args.output_dir, 'spectra_summary.csv')
        with open(summary_file, 'w') as f:
            f.write("filename,time,step,total_enstrophy,total_kinetic_energy\n")
            for r in sorted(results, key=lambda x: x['time']):
                f.write(f"{r['filename']},{r['time']:.6f},{r['step']},"
                       f"{r['total_enstrophy']:.6e},{r['total_kinetic']:.6e}\n")
        print(f"\nSummary saved to {summary_file}")
    
    print(f"\nProcessed {len(results)} files successfully")
    return 0


if __name__ == '__main__':
    exit(main())
