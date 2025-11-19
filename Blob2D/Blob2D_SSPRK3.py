import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from firedrake import *

# ======================
# CONFIGURATION
# ======================

# Physics parameters
g = 1.0      # Curvature (g = 2 * rho_s0 / R_c)
alpha = 0.1  # Parallel loss (alpha = rho_s0 / L_parallel)

# BCs
# BOUNDARY_TYPE = "Dirichlet"
BOUNDARY_TYPE = "Periodic"

# ICs
BACKGROUND_PLASMA = 0.0
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1

# Simulation parameters
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 64
END_TIME = 10.0
TIME_STEPS = 1000
DT = END_TIME / TIME_STEPS

# Printing
OUTPUT_INTERVAL = int(0.1 * TIME_STEPS / END_TIME)

# ======================
# MESH & FUNCTION SPACES
# ======================

# Mesh
if BOUNDARY_TYPE == "Periodic":
    mesh = PeriodicSquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)
else:
    mesh = SquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)

# Function spaces
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 2)

# Solution fields
w = Function(V_w, name="vorticity")
n = Function(V_n, name="density")
phi = Function(V_phi, name="potential")

# Placeholders for the input state in weak form
w_old = Function(V_w)
n_old = Function(V_n)

# Storage for SSPRK3 steps
w_0 = Function(V_w)  # State at t^n
n_0 = Function(V_n)
w_1 = Function(V_w)  # Intermediate stage 1
n_1 = Function(V_n)
w_2 = Function(V_w)  # Intermediate stage 2
n_2 = Function(V_n)

# Test functions
v_w = TestFunction(V_w)
v_n = TestFunction(V_n)
v_phi = TestFunction(V_phi)

# Trial functions
w_trial = TrialFunction(V_w)
n_trial = TrialFunction(V_n)
phi_trial = TrialFunction(V_phi)

# ======================
# BOUNDARY CONDITIONS
# ======================

if BOUNDARY_TYPE == "Dirichlet":
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    bcs = []

# ======================
# INITIAL CONDITIONS
# ======================

x, y = SpatialCoordinate(mesh)

w.interpolate(0.0)

centre = DOMAIN_SIZE / 2.0
r2 = (x - centre)**2 + (y - centre)**2
n.interpolate(BACKGROUND_PLASMA + BLOB_AMPLITUDE * exp(-r2 / (BLOB_WIDTH**2)))

# ======================
# WEAK FORMULATION
# ======================

v_ExB = as_vector([phi.dx(1), -phi.dx(0)])

normal = FacetNormal(mesh)

def advection_term(q, v_q, v_ExB):
    # Average normal velocity (to find flow direction)
    v_ExB_n_avg = dot(avg(v_ExB), normal('+'))
    # Upwinding step: select q from the upstream side
    # If v_ExB_n_avg > 0: Flow is (+) -> (-). We carry q(+)
    # If v_ExB_n_avg < 0: Flow is (-) -> (+). We carry q(-)
    flux_upwind = conditional(v_ExB_n_avg > 0, v_ExB_n_avg * q('+'), v_ExB_n_avg * q('-'))
    facet_term = (v_q('+') - v_q('-')) * flux_upwind * dS
    interior_term = q * div(v_q * v_ExB) * dx
    return facet_term - interior_term

# Potential equation
a_phi = inner(grad(phi_trial), grad(v_phi)) * dx
L_phi = -w_old * v_phi * dx

# Vorticity equation
a_w = w_trial * v_w * dx
L_w = (
    w_old * v_w * dx 
    - DT * advection_term(w_old, v_w, v_ExB)
    + DT * g * n_old.dx(1) * v_w * dx
    - DT * alpha * phi * n_old * v_w * dx
)

# Density equation
a_n = n_trial * v_n * dx
L_n = (
    n_old * v_n * dx
    - DT * advection_term(n_old, v_n, v_ExB)
    - DT * alpha * n_old * phi * v_n * dx
)

# ======================
# SOLVER
# ======================

# Elliptic solver
if BOUNDARY_TYPE == "Dirichlet":
    # Dirichlet BCs. Solution is unique
    phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, bcs=bcs)
    phi_solver = LinearVariationalSolver(phi_problem, solver_parameters={
        'ksp_type': 'cg',
        # Algebraic Multigrid, from HYPRE, is best for elliptic
        'pc_type': 'hypre',
        'pc_hypre_type': 'boomeramg',
    })
else:
    # Periodic BCs. Solution is unique only up to a constant
    # Use nullspace to make the problem solvable
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, bcs=bcs)
    phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace, solver_parameters={
        'ksp_type': 'cg',
        # Geometric Multigrid handles nullspaces well
        'pc_type': 'gamg',
    })

# Hyperbolic solvers
transport_parameters = {
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'lu',  # Invert the local blocks
}

w_problem = LinearVariationalProblem(a_w, L_w, w)
w_solver = LinearVariationalSolver(w_problem, solver_parameters=transport_parameters)

n_problem = LinearVariationalProblem(a_n, L_n, n)
n_solver = LinearVariationalSolver(n_problem, solver_parameters=transport_parameters)

# ======================
# TIME STEPPING
# ======================

output_file = VTKFile(f"Blob2D_SSPRK3_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Save ICs
t = 0.0
output_file.write(w, n, phi, time=t)

def solve_stage(w_input, n_input, w_output, n_output):
    "Takes one Forward Euler step"
    # Load input into weak form
    w_old.assign(w_input)
    n_old.assign(n_input)
    
    # Enforce solvability for periodic potential (zero mean vorticity)
    if BOUNDARY_TYPE == "Periodic":
        w_mean = assemble(w_old * dx) / (DOMAIN_SIZE**2)
        w_old.assign(w_old - w_mean)
    
    # Solve
    phi_solver.solve()
    w_solver.solve()
    n_solver.solve()
    
    # Write to output
    w_output.assign(w)
    n_output.assign(n)

def take_step():
    "Takes one SSPRK3 step"
    
    # Save the starting state U^n
    w_0.assign(w)
    n_0.assign(n)
    
    # === STAGE 1 ===
    # U(1) = U^n + dt * L(U^n)
    solve_stage(w_0, n_0, w_1, n_1)
    
    # === STAGE 2 ===
    # U(2) = 3/4 U^n + 1/4 (U(1) + dt * L(U(1)))
    solve_stage(w_1, n_1, w, n) # Use w, n as temporary storage
    w_2.assign(0.75 * w_0 + 0.25 * w)
    n_2.assign(0.75 * n_0 + 0.25 * n)
    
    # === STAGE 3 ===
    # U^{n+1} = 1/3 U^n + 2/3 (U(2) + dt * L(U(2)))
    solve_stage(w_2, n_2, w, n) # Use w, n as temporary storage
    w.assign((1.0/3.0) * w_0 + (2.0/3.0) * w)
    n.assign((1.0/3.0) * n_0 + (2.0/3.0) * n)

for step in range(1, TIME_STEPS + 1):
    t += DT
    
    take_step()
    
    if step % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.2f}/{END_TIME}")
        output_file.write(w, n, phi, time=t)

print(f"Done in {time.time() - start_time} seconds")
