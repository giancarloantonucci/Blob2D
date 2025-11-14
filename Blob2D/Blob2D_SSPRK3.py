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
BOUNDARY_TYPE = "dirichlet"
# BOUNDARY_TYPE = "periodic"

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
if BOUNDARY_TYPE == "periodic":
    mesh = PeriodicSquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)
else:
    mesh = SquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)

# Function spaces
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 2)

# Fields at current time step
w = Function(V_w, name="vorticity")
n = Function(V_n, name="density")
phi = Function(V_phi, name="potential")

# Fields at previous time step
w_old = Function(V_w)
n_old = Function(V_n)

# Intermediate stages
w_1 = Function(V_w)
n_1 = Function(V_n)
w_2 = Function(V_w)
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
# INITIAL CONDITIONS
# ======================

x, y = SpatialCoordinate(mesh)

w.interpolate(0.0)

centre = DOMAIN_SIZE / 2.0
r2 = (x - centre)**2 + (y - centre)**2
n.interpolate(BACKGROUND_PLASMA + BLOB_AMPLITUDE * exp(-r2 / (BLOB_WIDTH**2)))

# ======================
# BOUNDARY CONDITIONS
# ======================

if BOUNDARY_TYPE == "dirichlet":
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    bcs = []

# ======================
# WEAK FORMULATION
# ======================

v_ExB = as_vector([phi.dx(1), -phi.dx(0)])

normal = FacetNormal(mesh)

def advection_term(q, v_q, v_ExB):
    "From https://www.firedrakeproject.org/demos/DG_advection.py.html"
    # Conservation step
    # Calculate the normal velocity using the averaged field
    v_ExB_n_avg = dot(avg(v_ExB), normal('+'))
    # Upwinding step
    # If v_ExB_n_avg > 0: Flow is (+) -> (-). We carry q(+) info
    # If v_ExB_n_avg < 0: Flow is (-) -> (+). We carry q(-) info
    flux_upwind = conditional(v_ExB_n_avg > 0, v_ExB_n_avg * q('+'), v_ExB_n_avg * q('-'))
    # Facet term
    # (Jump in test function) * (Upwinded Flux)
    flux_term = (v_q('+') - v_q('-')) * flux_upwind * dS
    # Interior term
    # Integration by parts (standard DG)
    interior_term = q * div(v_q * v_ExB) * dx
    return flux_term - interior_term

# Potential equation
a_phi = inner(grad(phi_trial), grad(v_phi)) * dx
L_phi = -w_old * v_phi * dx

a_w = w_trial * v_w * dx
L_w = (
    w_old * v_w * dx 
    - DT * advection_term(w_old, v_w, v_ExB)
    + DT * g * n_old.dx(1) * v_w * dx
    - DT * alpha * phi * n_old * v_w * dx
)

a_n = n_trial * v_n * dx
L_n = (
    n_old * v_n * dx
    - DT * advection_term(n_old, v_n, v_ExB)
    - DT * alpha * n_old * phi * v_n * dx
)

# ======================
# SOLVER
# ======================

if BOUNDARY_TYPE == "dirichlet":
    # Dirichlet BCs. Solution is unique
    phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, bcs=bcs)
    phi_solver = LinearVariationalSolver(phi_problem, solver_parameters={
        'ksp_type': 'cg',  # Fastest for SPD linear systems
        'pc_type': 'hypre',  # Use HYPRE library
        'pc_hypre_type': 'boomeramg',  # Algebraic Multigrid, best for elliptic
    })
else:
    # Periodic BCs. Solution is unique only up to a constant
    # Define the nullspace to make the problem solvable
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, bcs=bcs)
    phi_solver = LinearVariationalSolver(phi_problem, nullspace=nullspace, solver_parameters={
        'ksp_type': 'cg',  # Fastest for SPD linear systems
        'pc_type': 'gamg',  # Geometric Multigrid, handles nullspaces well
    })

w_problem = LinearVariationalProblem(a_w, L_w, w)
w_solver = LinearVariationalSolver(w_problem, solver_parameters={
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'lu',  # Invert the local blocks
})

n_problem = LinearVariationalProblem(a_n, L_n, n)
n_solver = LinearVariationalSolver(n_problem, solver_parameters={
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'lu',  # Invert the local blocks
})

# ======================
# MAIN LOOP
# ======================

output_file = VTKFile(f"Blob2D_SSPRK3_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Save ICs
t = 0.0
output_file.write(w, n, phi, time=t)

def euler_step(w_in, n_in, w_out, n_out):
    """
    Performs one Forward Euler step: out = in + dt * RHS(in)
    """
    # 1. Load the state "in" into the variables used by the weak forms (w_old, n_old)
    w_old.assign(w_in)
    n_old.assign(n_in)
    # Enforce solvability condition for potential equation if periodic BCs
    # The RHS (w) must have zero mean
    if BOUNDARY_TYPE == "periodic":
        w_old_avg = assemble(w_old * dx) / (DOMAIN_SIZE**2)
        w_old.assign(w_old - w_old_avg)
    # 3. Update Phi (required for v_ExB in the next steps)
    phi_solver.solve()
    # 4. Solve continuity equations
    # The solvers compute the projection of the RHS, effectively doing the Euler step
    w_solver.solve()
    n_solver.solve()
    # 5. Copy results to "out"
    w_out.assign(w)
    n_out.assign(n)

for step in range(TIME_STEPS):
    t += DT
    
    # Save state at t^n
    w_old.assign(w)
    n_old.assign(n)
    
    # === STAGE 1 ===
    # w_1 = w_n + dt * L(w_n)
    euler_step(w_old, n_old, w_1, n_1)
    
    # === STAGE 2 ===
    # w_2 = 3/4 w_n + 1/4 (w_1 + dt * L(w_1))
    # First, compute the Euler step from w_1. Use 'w' as temporary storage.
    euler_step(w_1, n_1, w, n)
    
    # Combine
    w_2.assign(0.75 * w_old + 0.25 * w)
    n_2.assign(0.75 * n_old + 0.25 * n)
    
    # === STAGE 3 ===
    # w_{n+1} = 1/3 w_n + 2/3 (w_2 + dt * L(w_2))
    # First, compute the Euler step from w_2. Use 'w' as temporary storage.
    euler_step(w_2, n_2, w, n)
    
    # Combine to get final result
    w.assign((1.0/3.0) * w_old + (2.0/3.0) * w)
    n.assign((1.0/3.0) * n_old + (2.0/3.0) * n)
    
    print(f"Step {step+1}/{TIME_STEPS}: t = {t:.4f}/{END_TIME}")
    
    if (step+1) % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.4f}")
        output_file.write(w, n, phi, time=t)
        
print(f"Done in {time.time() - start_time} seconds")
