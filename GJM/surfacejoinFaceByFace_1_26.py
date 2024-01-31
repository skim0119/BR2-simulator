import warnings
import numpy as np
import elastica as ea
from numpy import cos, sin, sqrt
from numba import njit
from elastica import integrate, PositionVerlet,CallBackBaseClass,defaultdict
from surfacejoinFaceByFace_Preprocess_1_26 import glue_rods_surface_connection
# Import Boundary Condition Classes
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces

#plot video
from plot_video_GJM import plot_video_2D, plot_video

class ParallelRodRodConnect(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
):
    pass

#define simulator
Parallel_rod_rod_connect_sim = ParallelRodRodConnect()

# Simulation parameters
dt = 5e-4
final_time = 1
total_steps = int(final_time / dt)
time_step = np.float64(final_time / total_steps)
rendering_fps = 20
step_skip = int(1.0 / (rendering_fps * time_step))

# Rod parameters
n_elem_rod_one = 50
n_elem_rod_two = 50
base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
nu = 0.0
E = 3e5
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

# Rod orientations
start_one = np.zeros(3,)
start_two = np.array((0.02,0,0))
inclination = np.deg2rad(0)
direction = np.array([0.0, np.cos(inclination), np.sin(inclination)])
normal = np.array([0.0, -np.sin(inclination), np.cos(inclination)])

#try to add a streching force to an SurfaceJointSideBySide soft arm
#not sure which value should I put
k = 1e3 #弹性系数或者刚度
nu = 10 #粘性阻尼系数
kt = 1e3 #旋转刚度或者旋转弹性系数

rod_one = ea.CosseratRod.straight_rod(
    n_elem_rod_one,
    start_one,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

Parallel_rod_rod_connect_sim.append(rod_one)

rod_two = ea.CosseratRod.straight_rod(
    n_elem_rod_two,
    start_two,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

Parallel_rod_rod_connect_sim.append(rod_two)

# add damping
damping_constant = 2e-4
Parallel_rod_rod_connect_sim.dampen(rod_one).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)
Parallel_rod_rod_connect_sim.dampen(rod_two).using(
    ea.AnalyticalLinearDamper,
    damping_constant=damping_constant,
    time_step=dt,
)

# add constrain
Parallel_rod_rod_connect_sim.constrain(rod_one).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)#这里的idx是索引的意思
)
Parallel_rod_rod_connect_sim.constrain(rod_two).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)#这里的idx是索引的意思
)

glue_rods_surface_connection(Parallel_rod_rod_connect_sim,rod_one,rod_two,k,nu,kt)
        
Parallel_rod_rod_connect_sim.finalize()

#start simulation
timestepper = ea.PositionVerlet()
ea.integrate(timestepper, Parallel_rod_rod_connect_sim, final_time, total_steps)
