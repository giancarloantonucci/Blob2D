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
BACKGROUND_PLASMA = 0.0
BLOB_AMPLITUDE = 0.5
BLOB_WIDTH = 0.1
INITIAL_Te = 1.0
INITIAL_Ti = 0.01  # Lower this to reduce the ion-pressure brake

# Simulation
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 64
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
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_p_e = FunctionSpace(mesh, "DQ", 1)
V_p_i = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 2)

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

# Trial functions
u_w = TrialFunction(V_w)
u_n = TrialFunction(V_n)
u_p_e = TrialFunction(V_p_e)
u_p_i = TrialFunction(V_p_i)
u_phi = TrialFunction(V_phi)

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
n0 = BACKGROUND_PLASMA + BLOB_AMPLITUDE * exp(-((x - x_c)**2 + (y - y_c)**2) / (BLOB_WIDTH**2))
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

# E x B drift velocity
v_ExB = as_vector([phi.dx(1), -phi.dx(0)])

def advection_term(q, v_q, v_ExB):
    """
    DG advection: - int(q * v_ExB . grad(v_q)) dx + int(flux) dS
    """
    # Conservation step
    # We calculate the normal velocity using the averaged field
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
    interior_term = q * dot(v_ExB, grad(v_q)) * dx
    return flux_term - interior_term

# Electron temperature
n_floor = conditional(n_old > 1e-6, n_old, 1e-6)  # Avoid division by zero
T_e_old = p_e_old / n_floor
T_e_floor = conditional(T_e_old > 1e-4, T_e_old, 1e-4)  # Avoid division by zero in the loss terms

# Ion sound speed
p_total_old = p_e_old + p_i_old
p_total_floor = conditional(p_total_old > 0, p_total_old, 0)  # Avoid sqrt(negative)
c_s = sqrt(p_total_floor / n_floor) 

a_phi = inner(grad(u_phi), grad(v_phi)) * dx
L_phi = (
    - w_old * v_phi * dx
    - inner(grad(p_i_old), grad(v_phi)) * dx
    - inner(jump(p_i_old, normal), avg(grad(v_phi))) * dS
)

a_w = u_w * v_w * dx
L_w = (
    w_old * v_w * dx
    - DT * advection_term(w_old, v_w, v_ExB)
    + DT * g * (p_e_old + p_i_old).dx(1) * v_w * dx
    - DT * alpha * (n_old * c_s / T_e_floor) * phi * v_w * dx
)

a_n = u_n * v_n * dx
L_n = (
    n_old * v_n * dx
    - DT * advection_term(n_old, v_n, v_ExB)
    - DT * alpha * (n_old * c_s / T_e_floor) * phi * v_n * dx
)

a_p_e = u_p_e * v_p_e * dx
L_p_e = (
    p_e_old * v_p_e * dx
    - DT * advection_term(p_e_old, v_p_e, v_ExB)
    - DT * alpha * delta_e * p_e_old * c_s * v_p_e * dx
)

# Small parameter to control the amount of diffusion
epsilon = Constant(1.0e-4)
# Cell diameter, needed for SIPG
h = CellDiameter(mesh)
# Average cell diameter on a facet
h_avg = (h('+') + h('-')) / 2.0

a_p_i = u_p_i * v_p_i * dx
L_p_i = (
    p_i_old * v_p_i * dx
    - DT * advection_term(p_i_old, v_p_i, v_ExB)
    # The advection term creates numerical wiggles that can cause p_i to go
    # negative. F_p_i is pure advection and has no physical damping to counteract
    # this. We add artificial viscosity (SIPG) to dampen these wiggles
    - DT * epsilon * (
        inner(grad(p_i_old), grad(v_p_i)) * dx
        - inner(avg(grad(p_i_old)), jump(v_p_i, normal)) * dS
        - inner(jump(p_i_old, normal), avg(grad(v_p_i))) * dS
        + (Constant(10.0) / h_avg) * inner(jump(p_i_old), jump(v_p_i)) * dS
    )
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
    'sub_pc_type': 'ilu',  # Invert the local blocks
})

n_problem = LinearVariationalProblem(a_n, L_n, n)
n_solver = LinearVariationalSolver(n_problem, solver_parameters={
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'ilu',  # Invert the local blocks
})

p_e_problem = LinearVariationalProblem(a_p_e, L_p_e, p_e)
p_e_solver = LinearVariationalSolver(p_e_problem, solver_parameters={
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'ilu',  # Invert the local blocks
})

p_i_problem = LinearVariationalProblem(a_p_i, L_p_i, p_i)
p_i_solver = LinearVariationalSolver(p_i_problem, solver_parameters={
    'ksp_type': 'preonly',  # Apply preconditioner once
    'pc_type': 'bjacobi',  # Block Jacobi (unlike DG0, the basis functions for DG1 elements on a quadrilateral are not orthogonal)
    'sub_pc_type': 'ilu',  # Invert the local blocks
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
    
    # Update fields with the results from the last step
    w_old.assign(w)
    n_old.assign(n)
    p_e_old.assign(p_e)
    p_i_old.assign(p_i)
    
    # Enforce solvability condition for potential equation
    # The RHS (w) must have zero mean
    # Both pressure terms vanish automatically when tested against the nullspace
    if BOUNDARY_TYPE == "periodic":
        w_old_integral = assemble(w_old * dx)
        area = DOMAIN_SIZE * DOMAIN_SIZE
        w_old_avg = w_old_integral / area
        w_old.assign(w_old - w_old_avg)
    
    # Solve
    # solve() here re-assembles the RHS vectors L
    phi_solver.solve()
    w_solver.solve()
    n_solver.solve()
    p_e_solver.solve()
    p_i_solver.solve()
    
    print(f"Step {step+1}/{TIME_STEPS}: t = {t:.4f}/{END_TIME}")
    
    if (step+1) % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.4f}")
        output_file.write(w, n, p_e, p_i, phi, time=t)
        
end_time = time.time()
print(f"Done. Total wall-clock time: {end_time - start_time} seconds")
