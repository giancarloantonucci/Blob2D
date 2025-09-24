import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from firedrake import *
from irksome import Dt, TimeStepper, GaussLegendre

# ======================
# PARAMETERS
# ======================

# Parameters
g = 1.0  # curvature parameter (g = 2 * rho_s0 / R_c)
alpha = 0.1  # parallel loss parameter (alpha = rho_s0 / L_parallel)

# BCs
BOUNDARY_TYPE = "periodic" # "periodic" or "dirichlet"

# ICs
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1

# Simulation
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 128
END_TIME = 10.0
TIME_STEPS = 1000

# Printing
OUTPUT_INTERVAL = 10

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

# Function Spaces
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 1)
V = V_w * V_n * V_phi

# Fields
solution = Function(V)
w, n, phi = solution.subfunctions  # concrete references that auto-update
w_s, n_s, phi_s = split(solution)  # symbolic representations for weak forms

w.rename("vorticity")
n.rename("density")
phi.rename("potential")

# Test functions
v_w, v_n, v_phi = TestFunctions(V)

# ======================
# INITIAL CONDITIONS
# ======================

w.interpolate(0.0)

x_c = y_c = DOMAIN_SIZE / 2.0
n0 = 1.0 + BLOB_AMPLITUDE * exp(-((x - x_c)**2 + (y - y_c)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)

# ======================
# BOUNDARY CONDITIONS
# ======================

if BOUNDARY_TYPE == "dirichlet":
    # sheath-connected walls
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    bcs = []

# ======================
# WEAK FORMULATION
# ======================

# ExB drift velocity
driftvel = as_vector([phi_s.dx(1), -phi_s.dx(0)])

# Upwind flux term (for DG advection)
driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))

def advection_term(w, v_w, driftvel, driftvel_n):
    """Discontinuous Galerkin advection term with upwinding."""
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

F = (
    # Vorticity equation
    Dt(w_s) * v_w * dx
    + advection_term(w_s, v_w, driftvel, driftvel_n)
    - g * n_s.dx(1) * v_w * dx
    + alpha * phi_s * n_s * v_w * dx
    
    # Density equation
    + Dt(n_s) * v_n * dx
    + advection_term(n_s, v_n, driftvel, driftvel_n)
    + alpha * n_s * phi_s * v_n * dx
    
    # Potential equation
    + inner(grad(phi_s), grad(v_phi)) * dx
    + w_s * v_phi * dx
)

# ======================
# SOLVER
# ======================

# https://petsc.org/release/manualpages/SNES/

solver_parameters = {
    # PETSc's nonlinear solver (SNES) settings
    'snes_type': 'newtonls',  # Newton's method with line search
    'snes_monitor': None,  # Print convergence information
    'snes_max_it': 100,  # Maximum Newton iterations
    'snes_rtol': 1e-8,  # Relative tolerance for convergence
    'snes_linesearch_type': 'bt',  # Backtracking line search for robustness
    
    # Linear solver settings
    'mat_type': 'aij',  # Standard sparse matrix format
    'ksp_type': 'fgmres',  # Flexible GMRES for variable preconditioning
    
    # Fieldsplit preconditioner to handle the saddle-point structure
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',  # Schur complement method
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',  # Self-precondition Schur
    
    # Define block structure: [[w, n], [phi]]
    'pc_fieldsplit_0_fields': '0,1',  # w and n (hyperbolic)
    'pc_fieldsplit_1_fields': '2',  # phi (elliptic)
    
    # Solver for the (w, n) block
    'fieldsplit_0_ksp_type': 'gmres',
    'fieldsplit_0_ksp_rtol': 1e-6,
    'fieldsplit_0_pc_type': 'bjacobi',  # Block Jacobi for parallelism
    'fieldsplit_0_sub_pc_type': 'ilu',  # ILU on each block
    
    # Solver for the phi block
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'hypre',  # Algebraic multigrid for efficiency
}

# ======================
# TIME STEPPING
# ======================

V_t = FunctionSpace(mesh, "R", 0)

t = Function(V_t)  # current time
dt = Function(V_t)  # time step

t.assign(0.0)
dt.assign(END_TIME / TIME_STEPS)

# Time integrator (implicit midpoint rule)
butcher_tableau = GaussLegendre(1) 

# Create time stepper
stepper = TimeStepper(F, butcher_tableau, t, dt, solution, solver_parameters=solver_parameters, bcs=bcs)

# ======================
# MAIN LOOP
# ======================

# Output filename based on boundary type
output_filename = f"Blob2D_implicit_{BOUNDARY_TYPE}.pvd"
output_file = VTKFile(output_filename)
start_time = time.time()

print(f"Running with dt = {float(dt)}, {BOUNDARY_TYPE} BCs")

# Save initial condition
w, n, phi = solution.subfunctions
output_file.write(w, n, phi, time=float(t))

step_counter = 0
while step_counter < TIME_STEPS:
    # Advance solution in time
    step_counter += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    
    # Save output every OUTPUT_INTERVAL steps
    if step_counter % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {float(t)}")
        w, n, phi = solution.subfunctions
        output_file.write(w, n, phi, time=float(t))
        
    # Progress output
    print(f"Step {step_counter}/{TIME_STEPS}: t = {float(t)}/{END_TIME}")

end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
