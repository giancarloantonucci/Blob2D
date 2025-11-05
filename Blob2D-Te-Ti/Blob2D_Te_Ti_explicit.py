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
delta_e = 6.5  # Sheath heat-transmission coefficient for electrons

# BCs
# BOUNDARY_TYPE = "periodic"
BOUNDARY_TYPE = "dirichlet"

# ICs
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1
INITIAL_Te = 1.0
INITIAL_Ti = 0.1

# Simulation
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 128
END_TIME = 10.0
TIME_STEPS = 20000
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
V_w = FunctionSpace(mesh, "DQ", 0) # to satisfy LLB condition for F_phi
V_n = FunctionSpace(mesh, "DQ", 1) # not 0 to avoid 1st-order upwind (too diffusive)
V_p_e = FunctionSpace(mesh, "DQ", 1) # not 0 to avoid 1st-order upwind (too diffusive)
V_p_i = FunctionSpace(mesh, "DQ", 1) # for grad in F_phi
V_phi = FunctionSpace(mesh, "CG", 1) # for grad in F_phi

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
    # Sheath-connected walls
    bcs = [DirichletBC(V_phi, 0, 'on_boundary')]
else:
    bcs = []

# ======================
# WEAK FORMULATION
# ======================

driftvel = as_vector([phi.dx(1), -phi.dx(0)])

def advection_term(w, v_w, driftvel):
    """Discontinuous Galerkin advection term with upwinding."""
    driftvel_n = 0.5 * (dot(driftvel, normal) + abs(dot(driftvel, normal)))
    return (
        (v_w('+') - v_w('-')) * (driftvel_n('+') * w('+') - driftvel_n('-') * w('-')) * dS
        - w * dot(driftvel, grad(v_w)) * dx
    )

# Electron temperature
# T_e_old = p_e_old / n_old
n_floor = conditional(n_old > 1e-6, n_old, 1e-6)  # to avoid division by zero
T_e_old = p_e_old / n_floor
# T_e_old = Constant(1.0) # T_e ≈ (1 + δp_e) (1 - δn) ≈ 1

# Ion sound speed
# c_s = sqrt((p_e_old + p_i_old) / n_old)
p_total_old = p_e_old + p_i_old
p_total_floor = conditional(p_total_old > 0, p_total_old, 0)  # to avoid sqrt(negative)
c_s = sqrt(p_total_floor / n_floor) 
# c_s = Constant(1.0) # T_e ≈ sqrt( ((1 + δp_e) + (p_i0 + δp_i)) / (1 + δn) ) ≈ sqrt(1 + p_i0) ≈ 1

ion_pressure_flux = (1.0 / n0) * grad(p_i)
F_phi = (
    inner(grad(phi), grad(v_phi)) * dx
    + inner(ion_pressure_flux, grad(v_phi)) * dx
    + w * v_phi * dx
)

F_w = (
    (w - w_old) * v_w * dx
    + DT * advection_term(w_old, v_w, driftvel)
    - DT * g * (p_e_old + p_i_old).dx(1) * v_w * dx  # Curvature drift term
    + DT * alpha * (n_old * c_s / T_e_old) * phi * v_w * dx  # Sheath current loss    
)

F_n = (
    (n - n_old) * v_n * dx
    + DT * advection_term(n_old, v_n, driftvel)
    + DT * alpha * (n_old * c_s / T_e_old) * phi * v_n * dx  # Particle loss to sheath
)

epsilon = Constant(1.0e-3)
h = CellDiameter(mesh)
h_avg = (h('+') + h('-')) / 2.0

F_p_e = (
    + (p_e - p_e_old) * v_p_e * dx
    + DT * advection_term(p_e_old, v_p_e, driftvel)
    + DT * alpha * delta_e * p_e_old * c_s * v_p_e * dx
    # --- Artificial Viscosity using SIPG ---
    + DT * epsilon * (
        inner(grad(p_e_old), grad(v_p_e)) * dx
        - inner(avg(grad(p_e_old)), jump(v_p_e, normal)) * dS
        - inner(jump(p_e_old, normal), avg(grad(v_p_e))) * dS
        + (Constant(10.0) / h_avg) * inner(jump(p_e_old), jump(v_p_e)) * dS
    )
)

F_p_i = (
    + (p_i - p_i_old) * v_p_i * dx
    + DT * advection_term(p_i_old, v_p_i, driftvel)
    # --- Artificial Viscosity using SIPG ---
    + DT * epsilon * (
        inner(grad(p_i_old), grad(v_p_i)) * dx
        - inner(avg(grad(p_i_old)), jump(v_p_i, normal)) * dS
        - inner(jump(p_i_old, normal), avg(grad(v_p_i))) * dS
        + (Constant(10.0) / h_avg) * inner(jump(p_i_old), jump(v_p_i)) * dS
    )
)

# ======================
# SOLVER
# ======================

if BOUNDARY_TYPE == "periodic":
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
else:
    nullspace = None

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

w_problem = NonlinearVariationalProblem(F_w, w)
w_solver = NonlinearVariationalSolver(w_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

n_problem = NonlinearVariationalProblem(F_n, n)
n_solver = NonlinearVariationalSolver(n_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

p_e_problem = NonlinearVariationalProblem(F_p_e, p_e)
p_e_solver = NonlinearVariationalSolver(p_e_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

p_i_problem = NonlinearVariationalProblem(F_p_i, p_i)
p_i_solver = NonlinearVariationalSolver(p_i_problem, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',
    'snes_type': 'ksponly',  # Skip Newton, go straight to linear solve
})

# ======================
# MAIN LOOP
# ======================

output_file = VTKFile(f"Blob2D_Te_Ti_explicit_{BOUNDARY_TYPE}.pvd")
start_time = time.time()
print(f"Running with dt = {DT}, {BOUNDARY_TYPE} BCs")

# Save ICs
t = 0.0
output_file.write(w, n, p_e, p_i, phi, time=t)

for step in range(TIME_STEPS):
    t += DT
    
    # Solve in sequence
    phi_solver.solve()  # Does reassembly automatically
    w_solver.solve()  # Does reassembly automatically
    n_solver.solve()  # Does reassembly automatically
    p_e_solver.solve()  # Does reassembly automatically
    p_i_solver.solve()  # Does reassembly automatically

    # Update fields for next time step
    w_old.assign(w)
    n_old.assign(n)
    p_e_old.assign(p_e)
    p_i_old.assign(p_i)
    
    print(f"Step {step+1}/{TIME_STEPS}: t = {t:.4f}/{END_TIME}")
    
    if (step+1) % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.4f}")
        output_file.write(w, n, p_e, p_i, phi, time=t)
        
end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
