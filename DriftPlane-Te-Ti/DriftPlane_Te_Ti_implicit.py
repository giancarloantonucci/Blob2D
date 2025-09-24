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
delta_e = 6.5  # sheath heat transmission coefficient for electrons
delta_i = 2.0  # sheath heat transmission coefficient for ions
m_i_norm = 1.0  # normalised ion mass

# Sources
SOURCE_AMP_n = 0.01
SOURCE_AMP_p_e = 0.01
SOURCE_AMP_p_i = 0.01
SOURCE_WIDTH = 0.05
SOURCE_POS = 0.25

# BCs
BOUNDARY_TYPE = "periodic" # "periodic" or "dirichlet"

# ICs
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1
INITIAL_Te = 1.0
INITIAL_Ti = 0.1

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
V_p_e = FunctionSpace(mesh, "DQ", 1)
V_p_i = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 1)
V = V_w * V_n * V_p_e * V_p_i * V_phi

# Fields
solution = Function(V)
w, n, p_e, p_i, phi = solution.subfunctions  # concrete references that auto-update
w_s, n_s, p_e_s, p_i_s, phi_s = split(solution)  # symbolic representations for weak forms

w.rename("vorticity")
n.rename("density")
p_e.rename("electron_pressure")
p_i.rename("ion_pressure")
phi.rename("potential")

# Test functions
v_w, v_n, v_p_e, v_p_i, v_phi = TestFunctions(V)

# ======================
# INITIAL CONDITIONS
# ======================

w.interpolate(0.0)

x_c = y_c = DOMAIN_SIZE / 2.0
n0 = 1.0 + BLOB_AMPLITUDE * exp(-((x - x_c)**2 + (y - y_c)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)

p_e.interpolate(n0 * INITIAL_Te)
p_i.interpolate(n0 * INITIAL_Ti)

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

h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# Non-dimensional electron temperature
# T_e_s = p_e_s / n_s
T_e_s = Constant(1.0) # T_e ≈ (1 + δp_e) (1 - δn) ≈ 1

# Dynamic sound speed
# c_s = sqrt((p_e_s + p_i_s) / n_s)
c_s = Constant(1.0) # T_e ≈ sqrt( ((1 + δp_e) + (p_i0 + δp_i)) / (1 + δn) ) ≈ sqrt(1 + p_i0) ≈ 1

# Radially localised Gaussian source
source_profile = exp(-((x - SOURCE_POS)**2) / SOURCE_WIDTH**2)

F = (
    # Vorticity equation
    Dt(w_s) * v_w * dx
    + advection_term(w_s, v_w, driftvel, driftvel_n)
    - g * (p_e_s + p_i_s).dx(1) * v_w * dx
    + alpha * (n_s * c_s / T_e_s) * phi_s * v_w * dx
    
    # Density equation
    + Dt(n_s) * v_n * dx
    + advection_term(n_s, v_n, driftvel, driftvel_n)
    + alpha * (n_s * c_s / T_e_s) * phi_s * v_n * dx
    + alpha * n_s * c_s * v_n * dx
    - SOURCE_AMP_n * source_profile * v_n * dx
    
    # Electron pressure equation
    + Dt(p_e_s) * v_p_e * dx
    + advection_term(p_e_s, v_p_e, driftvel, driftvel_n)
    + alpha * delta_e * p_e_s * c_s * v_p_e * dx
    - SOURCE_AMP_p_e * source_profile * v_p_e * dx
    
    # Ion pressure equation
    + Dt(p_i_s) * v_p_i * dx
    + advection_term(p_i_s, v_p_i, driftvel, driftvel_n)
    + alpha * delta_i * p_i_s * c_s * v_p_i * dx
    - SOURCE_AMP_p_i * source_profile * v_p_i * dx
    
    # Potential equation
    + inner(grad(phi_s), grad(v_phi)) * dx
    + inner((1.0 / n_s) * grad(p_i_s), grad(v_phi)) * dx
    + w_s * v_phi * dx
    # SIPG terms for p_i
    - dot(jump(p_i_s, normal), avg((1.0 / n_s) * grad(v_phi))) * dS  # Consistency term
    - dot(avg((1.0 / n_s) * grad(p_i_s)), jump(v_phi, normal)) * dS  # Symmetry term
    + (Constant(10.0)/h_avg) * dot(jump(p_i_s, normal), jump(v_phi, normal)) * dS  # Penalty term
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
    
    # Define block structure: [[w, n, p_e, p_i], [phi]]
    'pc_fieldsplit_0_fields': '0, 1, 2, 3',  # w and n (hyperbolic)
    'pc_fieldsplit_1_fields': '4',  # phi (elliptic)
    
    # Solver for the (w, n) block
    'fieldsplit_0_ksp_type': 'gmres',
    'fieldsplit_0_ksp_rtol': 1e-6,
    'fieldsplit_0_pc_type': 'bjacobi',  # Block Jacobi for parallelism
    'fieldsplit_0_sub_pc_type': 'ilu',  # ILU on each block
    
    # Solver for the phi block
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'hypre',  # Algebraic multigrid for efficiency
}

# solver_params = {
#     'snes_monitor': None, # Print SNES convergence
#     'snes_max_it': 100, # Maximum nonlinear iterations
#     'snes_linesearch_type': 'l2', # Line search algorithm
#     'mat_type': 'aij', # Matrix type
#     'ksp_type': 'preonly', # Only use preconditioner
#     'pc_type': 'lu', # LU factorization
#     'pc_factor_mat_solver_type': 'mumps', # Parallel solver
# }

# ======================
# TIME STEPPING
# ======================

V_t = FunctionSpace(mesh, "R", 0)

t = Function(V_t)  # current time
dt = Function(V_t)  # time space

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
output_filename = f"Blob2D_Te_Ti_implicit_{BOUNDARY_TYPE}.pvd"
output_file = VTKFile(output_filename)
start_time = time.time()

print(f"Running with dt = {float(dt)}, {BOUNDARY_TYPE} BCs")

# Track key values for diagnostics
# n_max_history = []

# Save initial condition
w, n, p_e, p_i, phi = solution.subfunctions
output_file.write(w, n, p_e, p_i, phi, time=float(t))

step_counter = 0
while step_counter < TIME_STEPS:
    # Advance solution in time
    step_counter += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    
    # Check for NaNs
    # if not check_for_nan(w, n, phi, step_counter, float(t)):
    #     print(f"Simulation stopped. Last good step: {step_counter-1}")
    #     break
    
    # Save output every OUTPUT_INTERVAL steps
    if step_counter % OUTPUT_INTERVAL == 0:
        # Compute diagnostics
        # n_min, n_max, w_max, phi_max = compute_field_stats(w, n, phi)
        
        # Track density maximum
        # n_max_history.append(n_max)
        
        # Detect rapid growth
        # check_rapid_growth(n_max_history, n_max, step_counter)
        
        print(f"Saving output at t = {float(t)}")
        # print(f"  n: [{n_min}, {n_max}], |w|_max = {w_max}, |phi|_max = {phi_max}")
        w, n, p_e, p_i, phi = solution.subfunctions
        output_file.write(w, n, p_e, p_i, phi, time=float(t))
        
    # Progress output
    print(f"Step {step_counter}/{TIME_STEPS}: t = {float(t)}/{END_TIME}")

end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
