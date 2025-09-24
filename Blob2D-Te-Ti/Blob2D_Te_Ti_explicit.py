import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from firedrake import *

# ======================
# PARAMETERS
# ======================

# Parameters
g = 1.0  # curvature parameter (g = 2 * rho_s0 / R_c)
alpha = 0.1  # parallel loss parameter (alpha = rho_s0 / L_parallel)
delta_e = 6.5  # sheath heat transmission coefficient for electrons
m_i_norm = 1.0  # normalised ion mass

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
TIME_STEPS = 10000

# DT ≤ C * (DX / V_max)
# * C = Courant number. For DG(p), C ≤ 1/(2p+1)
# * DX = DOMAIN_SIZE / MESH_RESOLUTION
# * V_max = max ExB drift velocity ≈ sqrt(g * BLOB_AMPLITUDE * BLOB_WIDTH)
# 
# For p = 1 (C = 1/3), DX = 1.0 / 128 (≈ 0.008), V_max = sqrt(1 * 0.5 * 0.1) ≈ 0.3: DT ≤ 0.009
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

# Function Spaces
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_p_e = FunctionSpace(mesh, "DQ", 1)
V_p_i = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 1)

# Fields at current time step
w = Function(V_w, name="vorticity")
n = Function(V_n, name="density")
p_e = Function(V_p_e, name="electron_pressure")
p_i = Function(V_p_i, name="ion_pressure")
phi = Function(V_phi, name="potential")

# Fields at previous time step
w_old = Function(V_w)
n_old = Function(V_n)
p_e_old = Function(V_p_e)
p_i_old = Function(V_p_i)

# Test functions
v_w = TestFunction(V_w)
v_n = TestFunction(V_n)
v_p_e = TestFunction(V_p_e)
v_p_i = TestFunction(V_p_i)
v_phi = TestFunction(V_phi)

# ======================
# INITIAL CONDITIONS
# ======================

w.interpolate(0.0)
w_old.assign(w)

x_c = y_c = DOMAIN_SIZE / 2.0
n0 = 1.0 + BLOB_AMPLITUDE * exp(-((x - x_c)**2 + (y - y_c)**2) / (BLOB_WIDTH**2))
n.interpolate(n0)
n_old.assign(n)

p_e.interpolate(n0 * INITIAL_Te)
p_e_old.assign(p_e)

p_i.interpolate(n0 * INITIAL_Ti)
p_i_old.assign(p_i)

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

def advection_term(w, v_w, driftvel):
    """Discontinuous Galerkin advection term with upwinding."""
    # Upwind flux
    driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

h = CellDiameter(mesh)
h_avg = (h('+') + h('-'))/2

# Step 1: Solve potential equation implicitly from vorticity
F_phi = (
    + inner(grad(phi), grad(v_phi)) * dx
    + inner((1.0 / n) * grad(p_i), grad(v_phi)) * dx
    + w * v_phi * dx
    # SIPG terms for p_i
    - dot(jump(p_i, normal), avg((1.0 / n) * grad(v_phi))) * dS  # Consistency term
    - dot(avg((1.0 / n) * grad(p_i)), jump(v_phi, normal)) * dS  # Symmetry term
    + (Constant(10.0)/h_avg) * dot(jump(p_i, normal), jump(v_phi, normal)) * dS  # Penalty term
)

# Step 2: Update vorticity explicitly
# ExB drift velocity computed from potential
driftvel = as_vector([phi.dx(1), -phi.dx(0)])

# Non-dimensional electron temperature
# T_e_old = p_e_old / n_old
T_e_old = Constant(1.0) # T_e ≈ (1 + δp_e) (1 - δn) ≈ 1

# Dynamic sound speed
# c_s = sqrt((p_e_old + p_i_old) / n_old)
c_s = Constant(1.0) # T_e ≈ sqrt( ((1 + δp_e) + (p_i0 + δp_i)) / (1 + δn) ) ≈ sqrt(1 + p_i0) ≈ 1

F_w = (
    (w - w_old) * v_w * dx
    + DT * advection_term(w_old, v_w, driftvel)
    - DT * g * (p_e_old + p_i_old).dx(1) * v_w * dx  # Curvature drift term
    + DT * alpha * (n_old * c_s / T_e_old) * phi * v_w * dx  # Sheath current loss    
)

# Step 3: Update density explicitly
F_n = (
    (n - n_old) * v_n * dx
    + DT * advection_term(n_old, v_n, driftvel)
    + DT * alpha * (n_old * c_s / T_e_old) * phi * v_n * dx  # Particle loss to sheath
)

# Step 3: Update pressures explicitly
F_p_e = (
    + (p_e - p_e_old) * v_p_e * dx
    + DT * advection_term(p_e_old, v_p_e, driftvel)
    + DT * alpha * delta_e * p_e_old * c_s * v_p_e * dx
)

F_p_i = (
    + (p_i - p_i_old) * v_p_i * dx
    + DT * advection_term(p_i_old, v_p_i, driftvel)
)

# ======================
# SOLVER
# ======================

# Solver for Poisson equation
if BOUNDARY_TYPE == "dirichlet":
    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    phi_solver = NonlinearVariationalSolver(phi_problem, solver_parameters={
        'snes_type': 'ksponly',
        'ksp_type': 'cg',
        'ksp_rtol': 1e-10,
        'ksp_atol': 1e-12,
        'pc_type': 'hypre',
        'pc_hypre_type': 'boomeramg',
    })
else:
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)

    phi_problem = NonlinearVariationalProblem(F_phi, phi, bcs=bcs)
    phi_solver = NonlinearVariationalSolver(phi_problem, 
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        near_nullspace=nullspace,
        solver_parameters={
            'snes_type': 'ksponly',
            'ksp_type': 'cg',
            'ksp_rtol': 1e-10,
            'ksp_atol': 1e-12,
            'ksp_initial_guess_nonzero': True,
            'pc_type': 'gamg',
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

# Solver for electron-pressure update
p_e_problem = NonlinearVariationalProblem(F_p_e, p_e)
p_e_solver = NonlinearVariationalSolver(p_e_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',  # Simple for DG mass matrix
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

# Solver for ion-pressure update
p_i_problem = NonlinearVariationalProblem(F_p_i, p_i)
p_i_solver = NonlinearVariationalSolver(p_i_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',  # Simple for DG mass matrix
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

# ======================
# MAIN LOOP
# ======================

# Output filename based on boundary type
output_filename = f"Blob2D_Te_Ti_explicit_{BOUNDARY_TYPE}.pvd"
output_file = VTKFile(output_filename)
start_time = time.time()

print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Save initial condition
t = 0.0
output_file.write(w, n, p_e, p_i, phi, time=t)

for step in range(TIME_STEPS):
    # Update time
    t += DT
    
    # Step 1: Solve for potential given current vorticity
    phi_solver.solve()  # Does reassembly automatically
    
    # Step 2: Update vorticity using old values and new potential
    w_solver.solve()  # Does reassembly automatically
    
    # Step 3: Update density using old values and new potential
    n_solver.solve()  # Does reassembly automatically

    p_e_solver.solve()
    p_i_solver.solve()

    # Update old values for next time step
    w_old.assign(w)
    n_old.assign(n)
    p_e_old.assign(p_e)
    p_i_old.assign(p_i)
    
    # Save output every OUTPUT_INTERVAL steps
    if (step+1) % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t}")
        output_file.write(w, n, p_e, p_i, phi, time=t)
        
    # Progress output
    print(f"Step {step+1}/{TIME_STEPS}: t = {t}/{END_TIME}")

end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
