import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
from firedrake import *

# ======================
# PARAMETERS
# ======================

# Physics
g = 2.0        # Curvature (g = 2 * rho_s0 / R_c)
alpha = 0.05   # Parallel loss (alpha = rho_s0 / L_parallel)
delta_e = 6.5  # Sheath heat-transmission coefficient for electrons
delta_i = 2.0  # Sheath heat-transmission coefficient for ions

# Sources
SOURCE_DECAY_LENGTH = 0.05
SOURCE_AMPLITUDE_n = 1.0
SOURCE_AMPLITUDE_p_e = (2.0 / 3.0) * 1.0
SOURCE_AMPLITUDE_p_i = (2.0 / 3.0) * 0.1
RAMP_TIME = 0.5

# Diffusion parameters
NU_VISCOSITY = 1.0e-4

# ICs
BACKGROUND_PLASMA = 0.2  # Low density in the "SOL" (Right side)
CORE_DENSITY = 1.2  # High density in the "Core" (Left side)
INITIAL_Te = 1.0
INITIAL_Ti = 0.01

# Simulation
DOMAIN_SIZE = 1.0
MESH_RESOLUTION = 128
END_TIME = 10.0
TIME_STEPS = 10000
DT = END_TIME / TIME_STEPS
OUTPUT_INTERVAL = int(0.1 * TIME_STEPS / END_TIME)

# ======================
# MESH & FUNCTION SPACES
# ======================

# Mesh
# 1. Create the Periodic Base (The Poloidal Y-direction)
# 2. Extrude it to create the Bounded Radial X-direction
# "layers" defines the number of cells in the extruded direction
base_mesh = PeriodicIntervalMesh(MESH_RESOLUTION, length=DOMAIN_SIZE)
mesh = ExtrudedMesh(base_mesh, layers=MESH_RESOLUTION, layer_height=DOMAIN_SIZE/MESH_RESOLUTION)

# 2. ROTATE THE MESH COORDINATES
# We map (Periodic, Bounded) -> (Bounded_Y, Periodic_X) 
# So that the new X is Bounded (Left/Right) and new Y is Periodic (Top/Bottom)
Vc = mesh.coordinates.function_space()
x_old, y_old = SpatialCoordinate(mesh)
# We swap: New X = Old Y (Bounded), New Y = Old X (Periodic)
new_coords = Function(Vc).interpolate(as_vector([y_old, x_old]))
mesh.coordinates.assign(new_coords)

# Now we define coordinates based on the NEW rotated mesh
# x is now Bounded (Left-Right)
# y is now Periodic (Top-Bottom)
x, y = SpatialCoordinate(mesh)

# Function spaces
V_w = FunctionSpace(mesh, "DQ", 1)
V_n = FunctionSpace(mesh, "DQ", 1)
V_p_e = FunctionSpace(mesh, "DQ", 1)
V_p_i = FunctionSpace(mesh, "DQ", 1)
V_phi = FunctionSpace(mesh, "CG", 2)

# Solution fields
w = Function(V_w, name="vorticity")
n = Function(V_n, name="density")
p_e = Function(V_p_e, name="electron_pressure")
p_i = Function(V_p_i, name="ion_pressure")
phi = Function(V_phi, name="potential")

# Placeholders for the input state in weak form
w_old = Function(V_w)
n_old = Function(V_n)
p_e_old = Function(V_p_e)
p_i_old = Function(V_p_i)

# Storage for SSPRK3 steps
w_0 = Function(V_w)  # State at t^n
n_0 = Function(V_n)
p_e_0 = Function(V_p_e)
p_i_0 = Function(V_p_i)
w_1 = Function(V_w)  # Intermediate stage 1
n_1 = Function(V_n)
p_e_1 = Function(V_p_e)
p_i_1 = Function(V_p_i)
w_2 = Function(V_w)  # Intermediate stage 2
n_2 = Function(V_n)
p_e_2 = Function(V_p_e)
p_i_2 = Function(V_p_i)

# Test functions
v_w = TestFunction(V_w)
v_n = TestFunction(V_n)
v_p_e = TestFunction(V_p_e)
v_p_i = TestFunction(V_p_i)
v_phi = TestFunction(V_phi)

# Trial functions
w_trial = TrialFunction(V_w)
n_trial = TrialFunction(V_n)
p_e_trial = TrialFunction(V_p_e)
p_i_trial = TrialFunction(V_p_i)
phi_trial = TrialFunction(V_phi)

# ======================
# BOUNDARY CONDITIONS
# ======================

# Because we rotated the mesh:
# 'bottom' (start of extrusion) is now x=0 (Left Wall)
# 'top' (end of extrusion) is now x=L (Right Wall)
bcs = [
    DirichletBC(V_phi, 0, 'bottom'), 
    DirichletBC(V_phi, 0, 'top')
]

# ======================
# INITIAL CONDITIONS
# ======================

w.interpolate(0.0)

# Front setup (Vertical front at x = 0.25)
front_pos = 0.25
steepness = 0.05

# Density Profile (High on Left, Low on Right)
n0 = BACKGROUND_PLASMA + (CORE_DENSITY - BACKGROUND_PLASMA) * 0.5 * (1.0 - tanh((x - front_pos)/steepness))
n.interpolate(n0)

# Explicitly create function for coordinates
x_coord_func = Function(V_n)
x_coord_func.interpolate(x)
x_vals = x_coord_func.dat.data

# Apply random noise
np.random.seed(42)
noise_strength = 0.05
noise = np.random.normal(0, noise_strength, n.dat.data.shape)
# Mask noise to only appear near the front
mask = np.abs(x_vals - front_pos) < 0.1 
n.dat.data[:] += noise * mask

p_e.interpolate(n * INITIAL_Te)
p_i.interpolate(n * INITIAL_Ti)

# ======================
# SOURCE PROFILE
# ======================

# A (core) source entering from the left and decaying exponentially (into the SOL)
source_profile = 0.5 * (1.0 - tanh((x - front_pos)/steepness))

# Control variable for source strength (starts at 0.0 or t/t_R)
source_scaling = Constant(0.0)

# ======================
# WEAK FORMULATION
# ======================

# Velocity Definition with swapped coords
# v_y (Periodic) is parallel to base -> index 0
# v_x (Radial) is parallel to extrusion -> index 1
# ExB: v = (B x Grad Phi) / B^2. 
# v_x = -dPhi/dy = -phi.dx(0)
# v_y =  dPhi/dx =  phi.dx(1)
# Vector in Firedrake (Base, Extruded) -> (y, x)
# v_ExB = [v_y, v_x] = [phi.dx(1), -phi.dx(0)]
v_ExB = as_vector([phi.dx(1), -phi.dx(0)])

normal = FacetNormal(mesh)
# The docs state: "Interior facet integrals are no longer denoted by dS."
# They are split into dS_h (between layers) and dS_v (between columns).
# We combine them so our advection and SIPG terms cover all internal boundaries.
# dS_v = vertical facets (between columns)
# dS_h = horizontal facets (between layers)
dS = dS_v + dS_h

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
L_phi = (
    - w_old * v_phi * dx
    # - inner(grad(p_i_old), grad(v_phi)) * dx
    # - inner(jump(p_i_old, normal), avg(grad(v_phi))) * dS
)

# Floors
n_floor = conditional(n_old > 1e-6, n_old, 1e-6)  # Avoid division by zero
p_total_old = p_e_old + p_i_old
p_total_floor = conditional(p_total_old > 0, p_total_old, 0)  # Avoid sqrt(negative)

# Electron temperature (reference)
# if BACKGROUND_PLASMA == 0.0:
#     # Avoid 0/0 instabilities if starting from a vacuum state
#     T_e = Constant(1.0)
# else:
#     # Dynamic calculation derived from the equation of state
#     T_e_old = p_e_old / n_floor
#     T_e = conditional(T_e_old > 1e-4, T_e_old, 1e-4)
T_e = Constant(1.0)

# Ion sound speed
c_s = sqrt(p_total_floor / n_floor)

# Diffusion Constant
D_num = Constant(NU_VISCOSITY)

# Cell diameter, needed for SIPG
h = CellDiameter(mesh)
# Average cell diameter on a facet
h_avg = (h('+') + h('-')) / 2.0
# Penalty parameter (approx 10 * order^2)
sigma = Constant(10.0)

def sipg_term(q, v_q):
    return (
        inner(grad(q), grad(v_q)) * dx
        - inner(avg(grad(q)), jump(v_q, normal)) * dS
        - inner(jump(q, normal), avg(grad(v_q))) * dS
        + (sigma / h_avg) * inner(jump(q), jump(v_q)) * dS
    )

# Vorticity equation
a_w = w_trial * v_w * dx
L_w = (
    w_old * v_w * dx
    - DT * advection_term(w_old, v_w, v_ExB)
    + DT * g * (p_e_old + p_i_old).dx(1) * v_w * dx
    - DT * alpha * (n_old * c_s / T_e) * phi * v_w * dx
    - DT * D_num * sipg_term(w_old, v_w)
)

# Density equation
a_n = n_trial * v_n * dx
L_n = (
    n_old * v_n * dx
    - DT * advection_term(n_old, v_n, v_ExB)
    - DT * alpha * (n_old * c_s / T_e) * phi * v_n * dx
    - DT * alpha * n_old * c_s * v_n * dx
    + DT * source_scaling * SOURCE_AMPLITUDE_n * source_profile * v_n * dx
    - DT * D_num * sipg_term(n_old, v_n)
)

# Electron pressure equation
a_p_e = p_e_trial * v_p_e * dx
L_p_e = (
    p_e_old * v_p_e * dx
    - DT * advection_term(p_e_old, v_p_e, v_ExB)
    - DT * alpha * delta_e * p_e_old * c_s * v_p_e * dx
    + DT * source_scaling * SOURCE_AMPLITUDE_p_e * source_profile * v_p_e * dx
    - DT * D_num * sipg_term(p_e_old, v_p_e)
)

# Ion pressure equation
a_p_i = p_i_trial * v_p_i * dx
L_p_i = (
    p_i_old * v_p_i * dx
    - DT * advection_term(p_i_old, v_p_i, v_ExB)
    - DT * alpha * delta_i * p_i_old * c_s * v_p_i * dx
    + DT * source_scaling * SOURCE_AMPLITUDE_p_i * source_profile * v_p_i * dx
    - DT * D_num * sipg_term(p_i_old, v_p_i)
)

# ======================
# SOLVER
# ======================

# Elliptic solver
# Dirichlet BCs. Solution is unique
phi_problem = LinearVariationalProblem(a_phi, L_phi, phi, bcs=bcs)
phi_solver = LinearVariationalSolver(phi_problem, solver_parameters={
    'ksp_type': 'cg',
    # Algebraic Multigrid, from HYPRE, is best for elliptic
    'pc_type': 'hypre',
    'pc_hypre_type': 'boomeramg',
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

p_e_problem = LinearVariationalProblem(a_p_e, L_p_e, p_e)
p_e_solver = LinearVariationalSolver(p_e_problem, solver_parameters=transport_parameters)

p_i_problem = LinearVariationalProblem(a_p_i, L_p_i, p_i)
p_i_solver = LinearVariationalSolver(p_i_problem, solver_parameters=transport_parameters)

# ======================
# TIME STEPPING
# ======================

output_file = VTKFile(f"DriftPlane_Te_Ti_SSPRK3.pvd")
start_time = time.time()
print(f"Running with dt = {float(DT)}")

# Save ICs
t = 0.0
output_file.write(w, n, p_e, p_i, phi, time=t)

def solve_stage(w_input, n_input, p_e_input, p_i_input,
                w_output, n_output, p_e_output, p_i_output):
    "Takes one Forward Euler step"
    # Load input into weak form
    w_old.assign(w_input)
    n_old.assign(n_input)
    p_e_old.assign(p_e_input)
    p_i_old.assign(p_i_input)
    
    # Solve
    phi_solver.solve()
    w_solver.solve()
    n_solver.solve()
    p_e_solver.solve()
    p_i_solver.solve()
    
    # Write to output
    w_output.assign(w)
    n_output.assign(n)
    p_e_output.assign(p_e)
    p_i_output.assign(p_i)

def take_step():
    "Takes one SSPRK3 step"
    
    # Save the starting state U^n
    w_0.assign(w)
    n_0.assign(n)
    p_e_0.assign(p_e)
    p_i_0.assign(p_i)
    
    # === STAGE 1 ===
    # U(1) = U^n + dt * L(U^n)
    solve_stage(w_0, n_0, p_e_0, p_i_0, w_1, n_1, p_e_1, p_i_1)
    
    # === STAGE 2 ===
    # U(2) = 3/4 U^n + 1/4 (U(1) + dt * L(U(1)))
    solve_stage(w_1, n_1, p_e_1, p_i_1, w, n, p_e, p_i) # Use w, n, p_e, p_i as temporary storage
    w_2.assign(0.75 * w_0 + 0.25 * w)
    n_2.assign(0.75 * n_0 + 0.25 * n)
    p_e_2.assign(0.75 * p_e_0 + 0.25 * p_e)
    p_i_2.assign(0.75 * p_i_0 + 0.25 * p_i)
    
    # === STAGE 3 ===
    # U^{n+1} = 1/3 U^n + 2/3 (U(2) + dt * L(U(2)))
    solve_stage(w_2, n_2, p_e_2, p_i_2, w, n, p_e, p_i) # Use w, n, p_e, p_i as temporary storage
    w.assign((1.0/3.0) * w_0 + (2.0/3.0) * w)
    n.assign((1.0/3.0) * n_0 + (2.0/3.0) * n)
    p_e.assign((1.0/3.0) * p_e_0 + (2.0/3.0) * p_e)
    p_i.assign((1.0/3.0) * p_i_0 + (2.0/3.0) * p_i)

for step in range(1, TIME_STEPS + 1):
    t += DT
    
    if t < RAMP_TIME:
        # Linear ramp from 0 to 1
        source_scaling.assign(t / RAMP_TIME)
    else:
        # Full strength
        source_scaling.assign(1.0)
    
    take_step()
    
    # CFL diagnostics
    # if step % 100 == 0:
    #     # Compute max velocity
    #     # Fix: Create the function explicitly, then interpolate into it
    #     V_cfl = FunctionSpace(mesh, "DG", 0)
    #     v_cfl_func = Function(V_cfl)
        
    #     v_mag = sqrt(dot(v_ExB, v_ExB))
    #     v_cfl_func.interpolate(v_mag) # Forces computation
        
    #     max_v = v_cfl_func.dat.data.max()
        
    #     # Minimum cell size estimate
    #     min_h = DOMAIN_SIZE / MESH_RESOLUTION
    #     # CFL Number
    #     cfl_now = max_v * DT / min_h
    #     print(f"Step {step}: t = {t:.2f}, Source = {float(source_scaling):.2f}, CFL = {cfl_now:.3f}")
        
    #     if cfl_now > 0.5:
    #         print("WARNING: CFL > 0.5. Consider reducing DT.")
    
    if step % OUTPUT_INTERVAL == 0:
        print(f"Saving output at t = {t:.2f}/{END_TIME}")
        output_file.write(w, n, p_e, p_i, phi, time=t)

print(f"Done in {time.time() - start_time} seconds")
