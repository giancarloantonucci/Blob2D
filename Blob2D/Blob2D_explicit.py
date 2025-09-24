# ~ 5 mins
import os
import time
from firedrake import *

from Blob2D_diagnostics import *

# ======================
# PARAMETERS
# ======================

# Disable OpenMP threading for better performance with MPI
os.environ["OMP_NUM_THREADS"] = "1"

# Physical parameters (Non-dimensional)
g = 1.0  # Curvature parameter (g = 2 * rho_s0 / R_c) # Constant(20.0 / 9.0)
alpha = 0.1  # Parallel loss parameter (alpha = rho_s0 / L_parallel)

# Boundary conditions
# BOUNDARY_TYPE = "dirichlet"
BOUNDARY_TYPE = "periodic"

# Initial condition
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1

# Simulation setup
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 128
END_TIME = 10.0
TIME_STEPS = 10000
DT = END_TIME / TIME_STEPS

# Printing
OUTPUT_INTERVAL = 100

# =================
# SETUP
# =================

# Create mesh
if BOUNDARY_TYPE == "periodic":
    mesh = PeriodicSquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)
else:
    mesh = SquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)

x, y = SpatialCoordinate(mesh)
normal = FacetNormal(mesh)

# Function Spaces (DG for advected fields, CG for potential)
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 1)

# Fields at current time step
w = Function(V_w, name="vorticity")
n = Function(V_n, name="density")
phi = Function(V_phi, name="potential")

# Fields at previous time step
w_old = Function(V_w)
n_old = Function(V_n)

# Test functions
v_w = TestFunction(V_w)
v_n = TestFunction(V_n)
v_phi = TestFunction(V_phi)

# ======================
# INITIAL CONDITIONS
# ======================

# Zero initial vorticity
w.interpolate(0.0)
w_old.assign(w)

# Initial Gaussian blob density profile
x_centre = y_centre = DOMAIN_SIZE / 2.0
n0 = 1.0 + BLOB_AMPLITUDE * exp(-((x - x_centre)**2 + (y - y_centre)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)
n_old.assign(n)

# ======================
# BOUNDARY CONDITIONS
# ======================

# Set boundary conditions
if BOUNDARY_TYPE == "dirichlet":
    # Zero potential on all boundaries (sheath-connected walls)
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    # No boundary conditions for periodic case
    bcs = []

# ======================
# WEAK FORMULATION
# ======================

def advection_term(w, v_w, driftvel):
    """Discontinuous Galerkin advection term with upwinding."""
    # Upwind flux
    driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

# Step 1: Solve potential equation implicitly from vorticity
F_phi = (
    + inner(grad(phi), grad(v_phi)) * dx
    + w * v_phi * dx
)

# Step 2: Update vorticity explicitly
# ExB drift velocity computed from potential
driftvel = as_vector([phi.dx(1), -phi.dx(0)])

F_w = (
    (w - w_old) * v_w * dx
    + DT * advection_term(w_old, v_w, driftvel)
    - DT * g * n_old.dx(1) * v_w * dx  # Curvature drift term
    + DT * alpha * phi * n_old * v_w * dx  # Sheath current loss
)

# Step 3: Update density explicitly
F_n = (
    (n - n_old) * v_n * dx
    + DT * advection_term(n_old, v_n, driftvel)
    + DT * alpha * n_old * phi * v_n * dx  # Particle loss to sheath
)

# ======================
# SOLVER
# ======================

# Solver for Poisson equation
if BOUNDARY_TYPE == "dirichlet":
    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    phi_solver = NonlinearVariationalSolver(phi_problem, solver_parameters={
        'ksp_type': 'cg',
        'pc_type': 'hypre',
        'pc_hypre_type': 'boomeramg',
        'ksp_rtol': 1e-10,
        'ksp_atol': 1e-12,
        'snes_type': 'ksponly',
    })
else:
    nullspace = VectorSpaceBasis(constant=True)

    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    phi_solver = NonlinearVariationalSolver(phi_problem, 
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=nullspace,
        solver_parameters={
            'ksp_type': 'cg',
            'pc_type': 'jacobi',
            'ksp_rtol': 1e-10,
            'ksp_atol': 1e-12,
            'ksp_max_it': 2000,
            'snes_type': 'ksponly',
            'ksp_initial_guess_nonzero': True,
        }
    )

# Solver for vorticity update
w_problem = NonlinearVariationalProblem(F_w, w)
w_solver = NonlinearVariationalSolver(w_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',  # Simple for DG mass matrix
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

# Solver for density update
n_problem = NonlinearVariationalProblem(F_n, n)
n_solver = NonlinearVariationalSolver(n_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',  # Simple for DG mass matrix
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

# ======================
# MAIN LOOP
# ======================

# Output filename based on boundary type
output_filename = f"Blob2D_explicit_{BOUNDARY_TYPE}.pvd"
output_file = VTKFile(output_filename)
start_time = time.time()

print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Track key values for diagnostics
n_max_history = []

# Save initial condition
t = 0.0
output_file.write(w, n, phi, time=t)

for step in range(TIME_STEPS):
    # Update time
    t += DT
    
    # Step 1: Solve for potential given current vorticity
    phi_solver.solve()  # Does reassembly automatically
    
    # Step 2: Update vorticity using old values and new potential
    w_solver.solve()  # Does reassembly automatically
    
    # Step 3: Update density using old values and new potential
    n_solver.solve()  # Does reassembly automatically
    
    # Check for NaNs
    if not check_for_nan(w, n, phi, step+1, t):
        print(f"Simulation stopped. Last good step: {step}")
        break

    # Update old values for next time step
    w_old.assign(w)
    n_old.assign(n)
    
    # Save output every OUTPUT_INTERVAL steps
    if (step+1) % OUTPUT_INTERVAL == 0:
        # Compute diagnostics
        n_min, n_max, w_max, phi_max = compute_field_stats(w, n, phi)
        
        # Track density maximum
        n_max_history.append(n_max)
        
        # Detect rapid growth
        check_rapid_growth(n_max_history, n_max, step+1)
        
        print(f"Saving output at t = {t}")
        print(f"  n: [{n_min}, {n_max}], |w|_max = {w_max}, |phi|_max = {phi_max}")
        output_file.write(w, n, phi, time=t)
        
    # Progress output
    print(f"Step {step+1}/{TIME_STEPS}: t = {t}/{END_TIME}")

end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
