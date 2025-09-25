import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from firedrake import *
from irksome import Dt, TimeStepper, GaussLegendre

# ======================
# PARAMETERS
# ======================

# Parameters
g = 1.0  # Curvature parameter (g = 2 * rho_s0 / R_c)
alpha = 0.1  # Parallel loss parameter (alpha = rho_s0 / L_parallel)

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
    bcs = [DirichletBC(V.sub(2), 0, 'on_boundary')]
else:
    bcs = []

if BOUNDARY_TYPE == "periodic":
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
else:
    nullspace = None

# ======================
# WEAK FORMULATION
# ======================

driftvel = as_vector([phi_s.dx(1), -phi_s.dx(0)])

def advection_term(w, v_w, driftvel):
    """Discontinuous Galerkin advection term with upwinding."""
    driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

F = (
    # Vorticity equation
    Dt(w_s) * v_w * dx
    + advection_term(w_s, v_w, driftvel)
    - g * n_s.dx(1) * v_w * dx
    + alpha * phi_s * n_s * v_w * dx
    
    # Density equation
    + Dt(n_s) * v_n * dx
    + advection_term(n_s, v_n, driftvel)
    + alpha * n_s * phi_s * v_n * dx
    
    # Potential equation
    + inner(grad(phi_s), grad(v_phi)) * dx
    + w_s * v_phi * dx
)

# ======================
# SOLVER
# ======================

solver_parameters = {
    # PETSc's nonlinear solver (SNES)
    'snes_type': 'newtonls',  # Newton's method with line search
    'snes_monitor': None,  # Print convergence information
    'snes_max_it': 100,
    'snes_rtol': 1e-8,
    'snes_linesearch_type': 'bt',  # Backtracking line search for robustness
    
    # Linear solver
    'mat_type': 'aij',  # Sparse matrix format
    'ksp_type': 'fgmres',  # Flexible GMRES for variable preconditioning
    
    # Fieldsplit preconditioner to handle the saddle-point structure
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',  # Self-precondition Schur
    
    # Block structure: [[w, n], [phi]]
    'pc_fieldsplit_0_fields': '0,1',  # w and n (hyperbolic)
    'pc_fieldsplit_1_fields': '2',  # phi (elliptic)
    
    # Solver for the (w, n) block
    'fieldsplit_0_ksp_type': 'gmres',
    'fieldsplit_0_ksp_rtol': 1e-6,
    'fieldsplit_0_pc_type': 'bjacobi',  # Block Jacobi for parallelism
    'fieldsplit_0_sub_pc_type': 'ilu',  # ILU on each block
    
    # Solver for the phi block
    'fieldsplit_1_ksp_type': 'preonly',
}

# Preconditioner for phi based on boundary conditions
if BOUNDARY_TYPE == "dirichlet":
    solver_parameters['fieldsplit_1_pc_type'] = 'hypre'
else:
    solver_parameters['fieldsplit_1_pc_type'] = 'gamg'

# Alternative direct solver (LU with MUMPS)
# solver_params = {
#     'snes_monitor': None,
#     'snes_max_it': 100,
#     'snes_linesearch_type': 'l2',
#     'mat_type': 'aij',
#     'ksp_type': 'preonly',
#     'pc_type': 'lu',
#     'pc_factor_mat_solver_type': 'mumps',
# }

# ======================
# TIME STEPPING
# ======================

V_t = FunctionSpace(mesh, "R", 0)

t = Function(V_t)
t.assign(0.0)

dt = Function(V_t)
dt.assign(END_TIME / TIME_STEPS)

butcher_tableau = GaussLegendre(1)  # Implicit midpoint rule

stepper = TimeStepper(F, butcher_tableau, t, dt, solution,
                    solver_parameters=solver_parameters,
                    bcs=bcs, nullspace=nullspace)

# ======================
# MAIN LOOP
# ======================

output_file = VTKFile(f"Blob2D_implicit_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {float(dt)}, {BOUNDARY_TYPE} BCs")

# Save ICs
w, n, phi = solution.subfunctions
output_file.write(w, n, phi, time=float(t))

step_counter = 0
while step_counter < TIME_STEPS:
    step_counter += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(f"Step {step_counter}/{TIME_STEPS}: t = {float(t):.4f}/{END_TIME}")
    
    if step_counter % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {float(t)}")
        w, n, phi = solution.subfunctions
        output_file.write(w, n, phi, time=float(t))
        
end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
