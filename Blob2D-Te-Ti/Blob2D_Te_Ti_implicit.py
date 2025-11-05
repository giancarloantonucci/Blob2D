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
delta_e = 6.5  # Sheath heat-transmission coefficient for electrons
m_i_norm = 1.0  # Normalised ion mass

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
OUTPUT_INTERVAL = int(0.1 * TIME_STEPS / END_TIME)

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
    bcs = [DirichletBC(V.sub(4), 0, 'on_boundary')]
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

h = CellDiameter(mesh)
h_avg = avg(h)
# h_avg = (h('+') + h('-'))/2
sigma = Constant(12.0)  # Penalty parameter (10-20 range)

# Non-dimensional electron temperature
T_e_s = p_e_s / n_s
# T_e_s = Constant(1.0) # T_e ≈ (1 + δp_e) (1 - δn) ≈ 1

# Dynamic sound speed
c_s = sqrt((p_e_s + p_i_s) / n_s)
# c_s = Constant(1.0) # T_e ≈ sqrt( ((1 + δp_e) + (p_i0 + δp_i)) / (1 + δn) ) ≈ sqrt(1 + p_i0) ≈ 1

F = (
    # Vorticity equation
    Dt(w_s) * v_w * dx
    + advection_term(w_s, v_w, driftvel)
    - g * (p_e_s + p_i_s).dx(1) * v_w * dx
    + alpha * (n_s * c_s / T_e_s) * phi_s * v_w * dx
    
    # Density equation
    + Dt(n_s) * v_n * dx
    + advection_term(n_s, v_n, driftvel)
    + alpha * (n_s * c_s / T_e_s) * phi_s * v_n * dx
    
    # Electron pressure equation
    + Dt(p_e_s) * v_p_e * dx
    + advection_term(p_e_s, v_p_e, driftvel)
    + alpha * delta_e * p_e_s * c_s * v_p_e * dx
    
    # Ion pressure equation
    + Dt(p_i_s) * v_p_i * dx
    + advection_term(p_i_s, v_p_i, driftvel)
    
    # Potential equation
    + inner(grad(phi_s), grad(v_phi)) * dx
    + inner((1.0 / n_s) * grad(p_i_s), grad(v_phi)) * dx
    + w_s * v_phi * dx
    # SIPG terms for p_i
    - dot(jump(p_i_s, normal), avg((1.0 / n_s) * grad(v_phi))) * dS
    - dot(avg((1.0 / n_s) * grad(p_i_s)), jump(v_phi, normal)) * dS
    + sigma/h_avg * dot(jump(p_i_s, normal), jump(v_phi, normal)) * dS
)

# ======================
# SOLVER
# ======================

solver_parameters = {
    # PETSc's nonlinear solver (SNES)
    'snes_type': 'newtonls',  # Newton's method with line search
    'snes_linesearch_type': 'bt',  # Backtracking line search for robustness
    'snes_monitor': None,  # Print convergence information
    'snes_rtol': 1e-8,
    'snes_max_it': 100,
    
    # Linear solver
    'mat_type': 'aij',  # Sparse matrix format
    'ksp_type': 'fgmres',  # Flexible GMRES for variable preconditioning
    
    # Fieldsplit preconditioner to handle the saddle-point structure
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',  # Self-precondition Schur
    
    # Block structure: [[w, n, p_e, p_i], [phi]]
    'pc_fieldsplit_0_fields': '0, 1, 2, 3',
    'pc_fieldsplit_1_fields': '4',
    
    # Solver for the (w, n, p_e, p_i) block
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
# solver_parameters = {
#     'snes_type': 'newtonls',
#     'snes_linesearch_type': 'l2',
#     'snes_monitor': None,
#     'snes_rtol': 1e-8,
#     'snes_max_it': 100,
#     'ksp_type': 'preonly',
#     'pc_type': 'lu',
#     'pc_factor_mat_solver_type': 'mumps',
#     'mat_type': 'aij',
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

output_file = VTKFile(f"Blob2D_Te_Ti_implicit_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {float(dt)}, {BOUNDARY_TYPE} BCs")

# Save ICs
w, n, p_e, p_i, phi = solution.subfunctions
output_file.write(w, n, p_e, p_i, phi, time=float(t))

step_counter = 0
while step_counter < TIME_STEPS:
    step_counter += 1
    stepper.advance()
    t.assign(float(t) + float(dt))
    print(f"Step {step_counter}/{TIME_STEPS}: t = {float(t):.4f}/{END_TIME}")
    
    if step_counter % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {float(t)}")
        w, n, p_e, p_i, phi = solution.subfunctions
        output_file.write(w, n, p_e, p_i, phi, time=float(t))
        
end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
