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
from plot_video_GJM import plot_video

class ParallelRodRodConnect(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
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
step_skip = int(1.0 / (15*rendering_fps * time_step))

# Rod parameters
n_elem_rod = 50
base_length = 0.5
base_radius = 0.01
base_area = np.pi * base_radius ** 2
density = 1750
nu = 0.0
E = 1e6
poisson_ratio = 0.5
shear_modulus = E / (poisson_ratio + 1.0)

# Rod orientations
start_one = np.zeros(3,)
start_two = np.array((0.02,0,0))
start_three = np.array((0.04,0,0))
inclination = np.deg2rad(0)
direction = np.array([0.0, np.cos(inclination), np.sin(inclination)])
normal = np.array([0.0, -np.sin(inclination), np.cos(inclination)])

#try to add a streching force to an SurfaceJointSideBySide soft arm
#not sure which value should I put
k = 1e3 #弹性系数或者刚度
nu = 10 #粘性阻尼系数
kt = 1e3 #旋转刚度或者旋转弹性系数

rod_one = ea.CosseratRod.straight_rod(
    n_elem_rod,
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
    n_elem_rod,
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

rod_three = ea.CosseratRod.straight_rod(
    n_elem_rod,
    start_three,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

Parallel_rod_rod_connect_sim.append(rod_three)

#glue this two rods parallely
glue_rods_surface_connection(Parallel_rod_rod_connect_sim,rod_one,rod_two,k,nu,kt)
glue_rods_surface_connection(Parallel_rod_rod_connect_sim,rod_two,rod_three,k,nu,kt)

# add damping
damping_constant = 5000
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
Parallel_rod_rod_connect_sim.dampen(rod_three).using(
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
Parallel_rod_rod_connect_sim.constrain(rod_three).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)#这里的idx是索引的意思
)

#add forces
# try to add forces to simulator
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([0.0, 0.0, 1000.0])
ramp_up_time = 1.0
Parallel_rod_rod_connect_sim.add_forcing_to(rod_one).using(
    ea.EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
Parallel_rod_rod_connect_sim.add_forcing_to(rod_two).using(
    ea.EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
Parallel_rod_rod_connect_sim.add_forcing_to(rod_three).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
        

# Add call backs
class GJMCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):

        if current_step % self.every == 0:

            self.callback_params["time"].append(time)
            # Collect x
            self.callback_params["position"].append(system.position_collection.copy())
            # Collect radius
            self.callback_params["radius"].append(system.radius.copy())
            return

post_processing_dict_rod1 = ea.defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
Parallel_rod_rod_connect_sim.collect_diagnostics(rod_one).using(
    GJMCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_rod1,
)

post_processing_dict_rod2 = ea.defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
Parallel_rod_rod_connect_sim.collect_diagnostics(rod_two).using(
    GJMCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_rod2,
)

post_processing_dict_rod3 = ea.defaultdict(
    list
)  # list which collected data will be append
# set the diagnostics for rod and collect data
Parallel_rod_rod_connect_sim.collect_diagnostics(rod_three).using(
    GJMCallBack,
    step_skip=step_skip,
    callback_params=post_processing_dict_rod3,
)


Parallel_rod_rod_connect_sim.finalize()

#start simulation
timestepper = ea.PositionVerlet()
ea.integrate(timestepper, Parallel_rod_rod_connect_sim, final_time, total_steps)

#plotting videos
filename_video = 'surfacejoinFaceByFace_1_26'
plot_video(
    [post_processing_dict_rod1, post_processing_dict_rod2, post_processing_dict_rod3],
    video_name="3d_" + filename_video + ".mp4",
    fps=50,
    step=1,
    x_limits=(-0.2, 0.2),
    y_limits=(-0.2, 0.2),
    z_limits=(0, 0.5),
    dpi=100,
)