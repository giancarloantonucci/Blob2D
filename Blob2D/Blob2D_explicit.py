import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from firedrake import *

# ======================
# PARAMETERS
# ======================

# Parameters
g = 1.0  # Curvature parameter (g = 2 * rho_s0 / R_c)
alpha = 0.1  # Parallel loss parameter (alpha = rho_s0 / L_parallel)

# BCs
# BOUNDARY_TYPE = "periodic"
BOUNDARY_TYPE = "dirichlet"

# ICs
BACKGROUND_PLASMA = 1.0
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1

# Simulation
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 64
END_TIME = 10.0
TIME_STEPS = 10000
DT = END_TIME / TIME_STEPS

# Printing
OUTPUT_INTERVAL = int(0.1 * TIME_STEPS / END_TIME)

# =================
# SETUP
# =================

# Create mesh (quadrilaterals because more efficient on squares)
if BOUNDARY_TYPE == "periodic":
    mesh = PeriodicSquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)
else:
    mesh = SquareMesh(MESH_RESOLUTION, MESH_RESOLUTION, DOMAIN_SIZE, quadrilateral=True)

x, y = SpatialCoordinate(mesh)
normal = FacetNormal(mesh)

# Function Spaces
V_w = FunctionSpace(mesh, "DQ", 0)  # to satisfy LLB condition for F_phi
V_n = FunctionSpace(mesh, "DQ", 1)  # not 0 to avoid 1st-order upwind (too diffusive)
V_phi = FunctionSpace(mesh, "CG", 1)  # for grad in F_phi

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

w.interpolate(0.0)
w_old.assign(w)

x_c = y_c = DOMAIN_SIZE / 2.0
n0 = BACKGROUND_PLASMA + BLOB_AMPLITUDE * exp(-((x - x_c)**2 + (y - y_c)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)
n_old.assign(n)

# ======================
# BOUNDARY CONDITIONS
# ======================

if BOUNDARY_TYPE == "dirichlet":
    # Sheath-connected walls
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    bcs = []

# ======================
# WEAK FORMULATION
# ======================

# E x B drift velocity
driftvel = as_vector([phi.dx(1), -phi.dx(0)])

def advection_term(w, v_w, driftvel):
    """Discontinuous Galerkin advection term with upwinding."""
    driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

F_phi = (
    inner(grad(phi), grad(v_phi)) * dx
    + w * v_phi * dx
)

F_w = (
    (w - w_old) * v_w * dx
    + DT * advection_term(w_old, v_w, driftvel)
    - DT * g * n_old.dx(1) * v_w * dx
    + DT * alpha * phi * n_old * v_w * dx
)

F_n = (
    (n - n_old) * v_n * dx
    + DT * advection_term(n_old, v_n, driftvel)
    + DT * alpha * n_old * phi * v_n * dx
)

# ======================
# SOLVER
# ======================

if BOUNDARY_TYPE == "dirichlet":
    # Dirichlet BCs. Solution is unique
    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    phi_solver = NonlinearVariationalSolver(phi_problem, solver_parameters={
        'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
        'ksp_type': 'cg',  # Fastest for SPD linear systems
        'ksp_rtol': 1e-10,
        'ksp_atol': 1e-12,
        'pc_type': 'hypre',  # Use HYPRE library
        'pc_hypre_type': 'boomeramg',  # Algebraic Multigrid, best for elliptic
    })
else:
    # Periodic BCs. Solution is unique only up to a constant
    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    # Define the nullspace to make the problem solvable
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    phi_solver = NonlinearVariationalSolver(phi_problem, 
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=nullspace,
        solver_parameters={
            'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
            'ksp_type': 'cg',  # Fastest for SPD linear systems
            'ksp_rtol': 1e-10,
            'ksp_atol': 1e-12,
            'pc_type': 'gamg',  # Geometric Multigrid, handles nullspaces well
        }
    )

w_problem = NonlinearVariationalProblem(F_w, w)
w_solver = NonlinearVariationalSolver(w_problem, solver_parameters={
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
    'ksp_type': 'preonly',  # Apply preconditioner once, no iteration
    'pc_type': 'jacobi',  # Use matrix diagonal
})

n_problem = NonlinearVariationalProblem(F_n, n)
n_solver = NonlinearVariationalSolver(n_problem, solver_parameters={
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
    'ksp_type': 'preonly',  # Apply preconditioner once, no iteration
    'pc_type': 'jacobi',  # Use matrix diagonal
})

# ======================
# MAIN LOOP
# ======================

output_file = VTKFile(f"Blob2D_explicit_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Save ICs
t = 0.0
output_file.write(w, n, phi, time=t)

for step in range(TIME_STEPS):
    t += DT
    
    # Solve
    phi_solver.solve()  # Does reassembly automatically
    w_solver.solve()  # Does reassembly automatically
    n_solver.solve()  # Does reassembly automatically

    # Update fields for next time step
    w_old.assign(w)
    n_old.assign(n)
    
    print(f"Step {step+1}/{TIME_STEPS}: t = {t:.4f}/{END_TIME}")
    
    if (step+1) % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.4f}")
        output_file.write(w, n, phi, time=t)
        
end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
