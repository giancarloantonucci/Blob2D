#!/usr/bin/env python3
"""
Plot Blob2D spectra from the output of Blob2D_spectra.py
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse


def plot_single_spectrum(spectrum_file, ax, label=None, color=None, alpha=1.0):
    """Plot a single spectrum file."""
    # Read data, skipping header lines
    data = np.loadtxt(spectrum_file, skiprows=4)  # Skip the 4 header lines
    k = data[:, 0]
    spectrum = data[:, 1]
    
    # Extract time from header
    with open(spectrum_file, 'r') as f:
        header = f.readline()
        time_str = f.readline()
        time = float(time_str.split('=')[1].split(',')[0])
    
    # Plot
    if label is None:
        label = f't = {time:.1f}'
    
    ax.loglog(k, spectrum, 'o-', label=label, color=color, alpha=alpha, markersize=4)
    
    return k, spectrum, time


def plot_spectrum_evolution(spectrum_pattern, output_file=None):
    """Plot evolution of spectra over time."""
    # Find all spectrum files
    files = sorted(glob.glob(spectrum_pattern))
    if not files:
        print(f"No files found matching: {spectrum_pattern}")
        return
    
    # Determine spectrum type from filename
    if 'enstrophy' in spectrum_pattern:
        spectrum_type = 'Enstrophy'
        ylabel = r'Enstrophy spectrum $\mathcal{E}_\omega(k)$'
    else:
        spectrum_type = 'Kinetic Energy'
        ylabel = r'Kinetic energy spectrum $E_k(k)$'
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Colormap for time evolution
    times = []
    for f in files:
        with open(f, 'r') as fh:
            fh.readline()  # Skip first header line
            time_str = fh.readline()
            times.append(float(time_str.split('=')[1].split(',')[0]))
    
    if len(files) > 1:
        colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    else:
        colors = ['blue']
    
    # Track data range for reference scalings
    y_max = -np.inf
    k_min, k_max = np.inf, -np.inf
    
    # Plot each timestep
    for i, (f, color) in enumerate(zip(files, colors)):
        # For many files, only label first, middle, and last
        if len(files) > 20:
            if i == 0:
                label = f't = {times[i]:.1f}'
            elif i == len(files) // 2:
                label = f't = {times[i]:.1f}'
            elif i == len(files) - 1:
                label = f't = {times[i]:.1f}'
            else:
                label = None
        elif len(files) > 10:
            # For moderate number, label every 5th
            if i % 5 == 0 or i == len(files) - 1:
                label = f't = {times[i]:.1f}'
            else:
                label = None
        else:
            # For few files, label all
            label = f't = {times[i]:.1f}'
        
        # Use lower alpha for intermediate times to emphasise evolution
        if label is None:
            alpha = 0.3
        else:
            alpha = 0.8
        
        k, spectrum, _ = plot_single_spectrum(f, ax, label=label, color=color, alpha=alpha)
        
        # Update data ranges
        y_max = max(y_max, np.max(spectrum))
        k_min = min(k_min, np.min(k))
        k_max = max(k_max, np.max(k))
    
    # Add reference scalings based on actual data range
    if k_min < k_max:  # Check that we have a valid range
        k_ref = np.logspace(np.log10(k_min), np.log10(k_max), 50)
        # Scale reference lines to be visible near the data
        scaling_amp = y_max * 0.5  # Place at half the maximum for visibility
        ax.loglog(k_ref, scaling_amp * (k_ref / k_ref[-1])**(-5/3), 'k--', alpha=0.7, label=r'$k^{-5/3}$')
        ax.loglog(k_ref, scaling_amp * (k_ref / k_ref[-1])**(-3), 'k:', alpha=0.7, label=r'$k^{-3}$')
    
    ax.set_xlabel(r'Wavenumber $k$ [$\rho_s^{-1}$]')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{spectrum_type} Spectrum Evolution')
    ax.grid(True, alpha=0.3)
    
    # Position legend better and add a colorbar for time
    if len(files) > 20:
        # For many files, use a colorbar instead of legend
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=times[0], vmax=times[-1]))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=r'Time [$\omega_{ci}^{-1}$]')
        
        # Still show the reference scalings in legend
        handles, labels = ax.get_legend_handles_labels()
        # Get only the reference scaling entries (last 2)
        if len(handles) >= 2:
            ref_handles = handles[-2:]
            ref_labels = labels[-2:]
            ax.legend(ref_handles, ref_labels, loc='lower left', fontsize='small')
    else:
        # For fewer files, use regular legend but position it better
        ax.legend(loc='best', fontsize='small', ncol=1 if len(files) <= 5 else 2)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_comparison(enstrophy_files, kinetic_files, time_index=-1, output_file=None):
    """Plot enstrophy and kinetic spectra side by side for a given time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Track data ranges for reference scalings
    y_max_1, y_max_2 = -np.inf, -np.inf
    k_min_1, k_max_1 = np.inf, -np.inf
    k_min_2, k_max_2 = np.inf, -np.inf
    
    # Plot enstrophy
    if enstrophy_files:
        k, spectrum, time = plot_single_spectrum(enstrophy_files[time_index], ax1, 
                                                label='Enstrophy', color='red')
        ax1.set_title(f'Enstrophy Spectrum at t = {time:.1f}')
        ax1.set_ylabel(r'$\mathcal{E}_\omega(k)$')
        y_max_1 = np.max(spectrum)
        k_min_1, k_max_1 = np.min(k), np.max(k)
    
    # Plot kinetic energy
    if kinetic_files:
        k, spectrum, time = plot_single_spectrum(kinetic_files[time_index], ax2, 
                                                label='Kinetic Energy', color='blue')
        ax2.set_title(f'Kinetic Energy Spectrum at t = {time:.1f}')
        ax2.set_ylabel(r'$E_k(k)$')
        y_max_2 = np.max(spectrum)
        k_min_2, k_max_2 = np.min(k), np.max(k)
    
    # Add reference scalings to both axes
    for ax, y_max, k_min, k_max in [(ax1, y_max_1, k_min_1, k_max_1), 
                                     (ax2, y_max_2, k_min_2, k_max_2)]:
        if k_min < k_max:  # Check that we have a valid range
            k_ref = np.logspace(np.log10(k_min), np.log10(k_max), 50)
            # Scale reference lines to be visible near the data
            scaling_amp = y_max * 0.5  # Place at half the maximum for visibility
            ax.loglog(k_ref, scaling_amp * (k_ref / k_ref[-1])**(-5/3), 'k--', alpha=0.7, label=r'$k^{-5/3}$')
            ax.loglog(k_ref, scaling_amp * (k_ref / k_ref[-1])**(-3), 'k:', alpha=0.7, label=r'$k^{-3}$')
        
        ax.set_xlabel(r'Wavenumber $k$ [$\rho_s^{-1}$]')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_integrated_quantities(summary_file, output_file=None):
    """Plot time evolution of integrated enstrophy and kinetic energy."""
    # Read CSV summary
    data = np.genfromtxt(summary_file, delimiter=',', names=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Enstrophy evolution
    ax1.semilogy(data['time'], data['total_enstrophy'], 'ro-', label='Total Enstrophy')
    ax1.set_ylabel(r'Total Enstrophy $\int \mathcal{E}_\omega(k) dk$')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Kinetic energy evolution
    ax2.semilogy(data['time'], data['total_kinetic_energy'], 'bo-', label='Total Kinetic Energy')
    ax2.set_xlabel(r'Time [$\omega_{ci}^{-1}$]')
    ax2.set_ylabel(r'Total Kinetic Energy $\int E_k(k) dk$')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle('Evolution of Integrated Quantities')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Blob2D spectra')
    parser.add_argument('spectra_dir', help='Directory containing spectra files')
    parser.add_argument('-t', '--type', choices=['evolution', 'comparison', 'integrated', 'all'], 
                       default='all', help='Type of plot to create')
    parser.add_argument('-o', '--output', help='Output file prefix (saves as PNG)')
    
    args = parser.parse_args()
    
    # Find files
    enstrophy_files = sorted(glob.glob(os.path.join(args.spectra_dir, 'enstrophy_spectrum_*.txt')))
    kinetic_files = sorted(glob.glob(os.path.join(args.spectra_dir, 'kinetic_spectrum_*.txt')))
    summary_file = os.path.join(args.spectra_dir, 'spectra_summary.csv')
    
    if args.type in ['evolution', 'all']:
        # Plot spectrum evolution
        if enstrophy_files:
            output = f"{args.output}_enstrophy_evolution.png" if args.output else None
            plot_spectrum_evolution(os.path.join(args.spectra_dir, 'enstrophy_spectrum_*.txt'), output)
        
        if kinetic_files:
            output = f"{args.output}_kinetic_evolution.png" if args.output else None
            plot_spectrum_evolution(os.path.join(args.spectra_dir, 'kinetic_spectrum_*.txt'), output)
    
    if args.type in ['comparison', 'all']:
        # Plot comparison at final time
        if enstrophy_files and kinetic_files:
            output = f"{args.output}_spectra_comparison.png" if args.output else None
            plot_comparison(enstrophy_files, kinetic_files, time_index=-1, output_file=output)
    
    if args.type in ['integrated', 'all']:
        # Plot integrated quantities
        if os.path.exists(summary_file):
            output = f"{args.output}_integrated.png" if args.output else None
            plot_integrated_quantities(summary_file, output)
        else:
            print(f"Summary file not found: {summary_file}")
            print("Run spectral analysis with --summary flag to generate it")


if __name__ == '__main__':
    main()
