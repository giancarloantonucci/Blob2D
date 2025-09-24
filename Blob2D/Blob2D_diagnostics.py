"""
Diagnostic functions for Blob2D simulations.
Used by both explicit and implicit solvers for runtime checks only.
"""

import numpy as np


def check_for_nan(w, n, phi, step, t):
    """Check for NaN in any field."""
    if (np.isnan(w.dat.data_ro).any() or 
        np.isnan(n.dat.data_ro).any() or 
        np.isnan(phi.dat.data_ro).any()):
        print(f"\n*** NaN detected at step {step}, t = {t:.3f} ***")
        return False
    return True


def compute_field_stats(w, n, phi):
    """Compute min/max statistics for all fields."""
    n_max = n.dat.data_ro.max()
    n_min = n.dat.data_ro.min()
    w_max = abs(w.dat.data_ro).max()
    phi_max = abs(phi.dat.data_ro).max()
    
    return n_min, n_max, w_max, phi_max


def check_rapid_growth(n_max_history, n_max, step, threshold=1.5):
    """Check if density is growing too rapidly."""
    if len(n_max_history) > 1 and n_max > threshold * n_max_history[-2]:
        print(f"WARNING at step {step}: Rapid growth detected (factor {n_max/n_max_history[-2]:.2f})!")
        return True
    return False


def compute_conserved_quantities(w, n, phi, mesh):
    """Compute conserved quantities for monitoring."""
    from firedrake import dx, assemble
    
    # Total particle number
    total_particles = assemble(n * dx)
    
    # Total enstrophy (vorticity squared)
    total_enstrophy = assemble(0.5 * w * w * dx)
    
    # Total energy (kinetic + potential)
    # KE = 0.5 * |∇φ|²
    kinetic_energy = assemble(0.5 * (phi.dx(0)**2 + phi.dx(1)**2) * dx)
    
    # Potential energy (if there's a background gradient, etc.)
    # For now just KE
    total_energy = kinetic_energy
    
    return {
        'particles': float(total_particles),
        'enstrophy': float(total_enstrophy),
        'energy': float(total_energy)
    }


def check_cfl_condition(phi, dt, mesh_size):
    """Check CFL condition for explicit timestepping."""
    # Maximum E×B velocity
    max_vel = np.sqrt(2) * abs(phi.dat.data_ro).max() / mesh_size
    
    # CFL number
    cfl = max_vel * dt / mesh_size
    
    if cfl > 0.5:
        print(f"WARNING: CFL = {cfl:.2f} > 0.5 (max velocity ~ {max_vel:.2f})")
    
    return cfl
