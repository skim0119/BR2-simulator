import warnings
import numpy as np
import elastica as ea
from numpy import cos, sin, sqrt
from numba import njit
from elastica import integrate, PositionVerlet,CallBackBaseClass,defaultdict
from free_simulator import FreeAssembly,FreeCallback
# Import Boundary Condition Classes
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces

#plot video
from plot_video_GJM import plot_video_2D, plot_video

# Join the two rods
from elastica._linalg import _batch_norm, _batch_cross, _batch_matvec, _batch_dot, _batch_matmul, _batch_matrix_transpose
from elastica.interaction import (
    elements_to_nodes_inplace,
    node_to_element_position,
    node_to_element_velocity,
)

from elastica._rotations import _inv_skew_symmetrize

#10.24GJM
#try to add a streching force to an SurfaceJointSideBySide soft arm
#not sure which value should I put
k = 1e3 #弹性系数或者刚度
nu = 10 #粘性阻尼系数
kt = 1e3 #旋转刚度或者旋转弹性系数
# rd1_local = 5
# rd2_local = 5
# SurfaceJointSideBySide_try_1 = SurfaceJointSideBySide(k,nu,kt,rd1_local,rd2_local)

rod_one_spec = {
    'n_elements' : 40,
    'start' : np.zeros((3,)),
    'direction' : np.array([0.0,0.0,1.0]), #Q matrix
    'normal' : np.array([0.0,1.0,0.0]), #法向量
    'base_length' : 0.15,
    'base_radius' : 0.005,
    'density' : 1000,
    'nu' : None,
    'youngs_modulus' : 1e6,
    'outer_radius' : 0.005,
    'inner_radius' : 0.002,
    'damping_constant' : 100
           }

rod_two_spec = {
    'n_elements' : 40,
    'start' : np.array([0.01,0,0]),
    'direction' : np.array([0.0,0.0,1.0]),
    'normal' : np.array([0.0,1.0,0.0]),
    'base_length' : 0.15,
    'base_radius' : 0.005,
    'density' : 1000,
    'nu' : None,
    'youngs_modulus' : 1e6,
    'outer_radius' : 0.005,#现在暂时没用
    'inner_radius' : 0.002,
    'damping_constant' : 100
           }

rod_three_spec = {
    'n_elements' : 40,
    'start' : np.array([0.02,0,0]),
    'direction' : np.array([0.0,0.0,1.0]),
    'normal' : np.array([0.0,1.0,0.0]),
    'base_length' : 0.15,
    'base_radius' : 0.005,
    'density' : 1000,
    'nu' : None,
    'youngs_modulus' : 1e6,
    'outer_radius' : 0.005,#现在暂时没用
    'inner_radius' : 0.002,
    'damping_constant' : 100
           }

time_step = 1e-4
class Environment:
    def __init__(self,time_step):
        self.time_step = time_step

env = Environment(time_step)
assembly = FreeAssembly(env)
#这里之后就和timoshenko beam应该是一样的
simulator = assembly.simulator

# rod_one = assembly.create_rod(name='A_01_rodone',is_first_segment=True, verbose=False, **rod_one_spec)
# rod_two = assembly.create_rod(name='A_02_rodtwo',is_first_segment=True, verbose=False, **rod_two_spec)

#尝试按照正常的pyelstica的方式定义一下
rod_one = ea.CosseratRod.straight_rod(**rod_one_spec)
rod_two = ea.CosseratRod.straight_rod(**rod_two_spec)
rod_three = ea.CosseratRod.straight_rod(**rod_three_spec)


rod_one.outer_radius = rod_one_spec['outer_radius']
rod_one.inner_radius = rod_one_spec['inner_radius']

rod_two.outer_radius = rod_two_spec['outer_radius']
rod_two.inner_radius = rod_two_spec['inner_radius']

rod_three.outer_radius = rod_three_spec['outer_radius']
rod_three.inner_radius = rod_three_spec['inner_radius']

simulator.append(rod_one)
simulator.append(rod_two)
simulator.append(rod_three)

#ablation test
assembly.glue_rods_surface_connection(rod_one,rod_two,k,nu,kt)#这两根rods依旧是分开的，所以仍旧需要对两根都施加力，同时用callback
assembly.glue_rods_surface_connection(rod_two,rod_three,k,nu,kt)#这两根rods依旧是分开的，所以仍旧需要对两根都施加力，同时用callback

#11.10GJM
#try to add BC to simulator
# 固定用函数（也可以模仿timoshenko）给两个rods固定  def tip_to_base_connection(self, rod1, rod2, k, nu, kt):
simulator.constrain(rod_one).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
simulator.constrain(rod_two).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
simulator.constrain(rod_three).using(
    ea.OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
)

# try to add forces to simulator
origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([0.0, 0.15, 0.0])
ramp_up_time = 1.0
simulator.add_forcing_to(rod_one).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
simulator.add_forcing_to(rod_two).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
simulator.add_forcing_to(rod_three).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)

#callback 参考butterfly 两个rods都要写 def make_callback(self, system, time, current_step: int):
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

recorded_history_one = defaultdict(list)
recorded_history_two = defaultdict(list)
recorded_history_three = defaultdict(list)


simulator.collect_diagnostics(rod_one).using(
    GJMCallBack, step_skip=10, callback_params=recorded_history_one
)

simulator.collect_diagnostics(rod_two).using(
    GJMCallBack, step_skip=10, callback_params=recorded_history_two
)

simulator.collect_diagnostics(rod_three).using(
    GJMCallBack, step_skip=10, callback_params=recorded_history_three
)


simulator.finalize()

final_time = 1 #仿真时间10s
time_steps = PositionVerlet()
dl = rod_one_spec['base_length'] / rod_one_spec['n_elements']
dt = 1e-4
n_steps = int(final_time / dt)
integrate(time_steps,simulator,final_time,n_steps)#n_steps是总步数


#def plot_video_2D
#making vedio
filename_video = 'threerods_SurfaceJointSidebySide'
plot_video(
    [recorded_history_one, recorded_history_two, recorded_history_three],
    video_name="3d_" + filename_video + ".mp4",
    fps=50,
    step=1,
    x_limits=(-0.2, 0.2),
    y_limits=(-0.2, 0.2),
    z_limits=(0, 0.2),
    dpi=100,
)



