# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage

import os, sys
from copy import deepcopy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, euler_from_quat
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.human import load_target_jt_new
from .g1_config import G1RoughCfg
import IPython; e = IPython.embed
import imageio
import torch
import numpy as np

def quaternion_to_angular_velocity(R0, R1, delta_t):
    """
    Compute angular velocity from two quaternions and time difference.
    
    Parameters:
    R0 (torch.Tensor): Quaternion at time t0 (shape: [4]).
    R1 (torch.Tensor): Quaternion at time t1 (shape: [4]).
    delta_t (float): Time interval between t0 and t1.
    
    Returns:
    torch.Tensor: Angular velocity vector (shape: [3]).
    """
    # Ensure the quaternions are normalized
    # R0 = R0 / R0.norm(dim=-1)
    # R1 = R1 / R1.norm(dim=-1)
    
    # Compute relative quaternion: Δq = R1 * conj(R0)
    R0_conj = -R0
    R0_conj[:,3] = R0[:,3]
    delta_q = R0.clone()
    
    delta_q[:,0] = R1[:,3] * R0_conj[:,0] + R1[:,0] * R0_conj[:,3] + R1[:,1] * R0_conj[:,2] - R1[:,2] * R0_conj[:,1]
    delta_q[:,1] = R1[:,3] * R0_conj[:,1] - R1[:,0] * R0_conj[:,2] + R1[:,1] * R0_conj[:,3] + R1[:,2] * R0_conj[:,0]
    delta_q[:,2] = R1[:,3] * R0_conj[:,2] + R1[:,0] * R0_conj[:,1] - R1[:,1] * R0_conj[:,0] + R1[:,2] * R0_conj[:,3]
    delta_q[:,3] = R1[:,3] * R0_conj[:,3] - R1[:,0] * R0_conj[:,0] - R1[:,1] * R0_conj[:,1] - R1[:,2] * R0_conj[:,2]
    
    # Extract the imaginary part (x, y, z) and normalize
    delta_q_imag = delta_q[:,:3]  # Extract imaginary components
    angular_velocity = 2 * delta_q_imag / delta_t
    
    return angular_velocity


def sample_int_from_float(x):
    if int(x) == x:
        return int(x)
    return int(x)+1 if np.random.rand() < (x - int(x)) else int(x)

def create_wireframe_cone_geometry(radius=1.0, height=1.0, num_segments=16):
    """
    创建线框圆锥体的顶点和线段
    
    参数:
        radius (float): 底面圆的半径
        height (float): 圆锥体的高度
        num_segments (int): 底面圆的分段数
        
    返回:
        vertices (list): 顶点列表
        lines (list): 线段索引列表
    """
    import math
    
    vertices = []
    lines = []
    
    # 添加顶点
    # 圆锥顶点
    vertices.append([0, height, 0])
    
    # 底面圆的顶点
    for i in range(num_segments):
        angle = 2.0 * math.pi * i / num_segments
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        vertices.append([x, 0, z])
    
    # 添加线段
    # 从顶点到底面的线段
    for i in range(num_segments):
        lines.append([0, i + 1])  # 顶点(索引0)到底面顶点的连线
    
    # 底面圆的线段
    for i in range(num_segments):
        lines.append([i + 1, ((i + 1) % num_segments) + 1])
    
    return vertices, lines

def create_visual_wireframe_cone(gym, env, pose, radius=1.0, height=1.0, num_segments=16, color=(1, 1, 1)):
    """
    在Isaac Gym中创建并显示线框圆锥体
    
    参数:
        gym: Isaac Gym实例
        env: 环境实例
        pose: 圆锥体的位姿
        radius (float): 底面圆的半径
        height (float): 圆锥体的高度
        num_segments (int): 底面圆的分段数
        color (tuple): RGB颜色值 (0-1范围)
    """
    vertices, lines = create_wireframe_cone_geometry(radius, height, num_segments)
    
    # 创建可视化对象
    for line in lines:
        start = vertices[line[0]]
        end = vertices[line[1]]
        
        # 应用位姿变换
        start_transformed = [
            start[0] + pose.p.x,
            start[1] + pose.p.y,
            start[2] + pose.p.z
        ]
        end_transformed = [
            end[0] + pose.p.x,
            end[1] + pose.p.y,
            end[2] + pose.p.z
        ]
        
        # 使用gym.add_lines()添加线段
        gym.add_lines(env, start_transformed, end_transformed, color)

# 使用示例:
"""
# 在你的Isaac Gym代码中使用:
from isaacgym import gymapi

gym = gymapi.acquire_gym()
# ... 创建环境等其他设置 ...

# 创建一个位姿
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)  # 位置
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # 旋转

# 创建线框圆锥体
create_visual_wireframe_cone(
    gym=gym,
    env=env,
    pose=pose,
    radius=0.5,
    height=1.0,
    num_segments=16,
    color=(1, 0, 0)  # 红色
)
"""

def create_arrow(gym, sim, env, base_position, arrow_length=1.0, arrow_radius=0.05):
    """
    Create an arrow using a cylinder and a cone in Isaac Gym.
    
    Parameters:
    gym (gymapi.Gym): Isaac Gym instance.
    sim (gymapi.Sim): Simulation instance.
    env (gymapi.Env): Environment instance.
    base_position (list or np.array): [x, y, z] position of the arrow base.
    arrow_length (float): Total length of the arrow.
    arrow_radius (float): Radius of the arrow's shaft.
    
    Returns:
    arrow_handles: A tuple of actor handles (cylinder, cone).
    """
    # Create cone (tip of the arrow)
    cone_options = gymapi.AssetOptions()
    cone = gym.create_capsule(sim, arrow_radius, arrow_length, cone_options)

    # Position the cone
    cone_pose = gymapi.Transform()
    cone_pose.p = gymapi.Vec3(base_position[0], base_position[1], base_position[2] + arrow_length * 0.9)

    # Create the cylinder and cone actors in the environment
    cone_handle = gym.create_actor(env, cone, cone_pose, "cone", -1, 0)

    return cone_handle

class G1():
    def __init__(self, cfg: G1RoughCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.video_step = -1
        self._parse_cfg(self.cfg)
        self._super_init(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        
        # human retargeted poses
        self._init_target_jt()
        self.init_done = True
    
    def _super_init(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        # if self.headless == True:
        #     self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.obs_context_len = cfg.env.obs_context_len

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
        
        self.obs_history_buf = torch.zeros(self.num_envs, self.obs_context_len, self.cfg.env.num_observations, device=self.device, dtype=torch.float)
        self.action_history_buf = torch.zeros(self.num_envs, self.action_delay + 2, self.num_actions, device=self.device, dtype=torch.float)

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.create_camera()
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_history_buf
    
    def get_privileged_observations(self):
        return None


    def _init_target_jt(self):
        if self.cfg.commands.command_type == "target":
            self.target_jt_seq = load_target_jt_new(self.device, self.cfg.human.filename, self.default_dof_pos, self.cfg.human.freq)
            self.num_target_jt_seq = self.target_jt_seq.shape[0]
            print(f"Loaded target joint trajectories of shape {self.target_jt_seq.shape}")
            self.target_jt_dt = 1 / self.cfg.human.freq
            self.target_jt_update_steps = self.target_jt_dt / self.dt # not necessary integer
            assert(self.dt <= self.target_jt_dt)
            self.target_jt_i = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.target_jt_j = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.target_jt_exp = torch.tensor(0., dtype=torch.long, device=self.device)
            self.target_jt_update_steps_int = sample_int_from_float(self.target_jt_update_steps)
            self.delayed_obs_target_jt = None
            self.delayed_obs_target_jt_steps = self.cfg.human.delay / self.target_jt_dt
            self.delayed_obs_target_jt_steps_int = sample_int_from_float(self.delayed_obs_target_jt_steps)
            self._reset_jt(torch.arange(self.num_envs, device=self.device))



    def update_target_jt(self):
        if self.cfg.commands.command_type == "target":
            self.target_jt = self.target_jt_seq[self.target_jt_i, self.target_jt_j]
            self.target_jt_next = self.target_jt_seq[self.target_jt_i, self.target_jt_j+1]
            self.target_bp = self.target_jt[:,36:].view(self.num_envs,-1,3)
            self.target_bp_next = self.target_jt_next[:,36:].view(self.num_envs,-1,3)
            self.delayed_obs_target_jt = self.target_jt_seq[self.target_jt_i, torch.maximum(self.target_jt_j - self.delayed_obs_target_jt_steps_int, torch.tensor(0)),7:36]

            base_quat = self.target_jt[:, 3:7]
            lin_vel = (self.target_jt_next[:,:3] - self.target_jt[:,:3])*self.cfg.human.freq
            ang_vel = quaternion_to_angular_velocity(self.target_jt[:,3:7],self.target_jt_next[:,3:7],  1/self.cfg.human.freq)
            body_pos = self.body_pos[:,self.target_pos_idx] - self.root_states[:,:3].unsqueeze(1)
            for i in range(body_pos.shape[1]):
                body_pos[:,i] = quat_rotate_inverse(self.root_states[:,3:7], body_pos[:,i])
            body_vel = (self.target_bp - body_pos)*(self.cfg.human.freq*4)
            vel_norm = body_vel.norm(dim=-1) + 1e-6 #防止除零
            body_vel = body_vel/vel_norm.sqrt().unsqueeze(-1)
            
            self.commands[:,:3] = quat_rotate_inverse(base_quat, lin_vel)
            self.commands[:,3:6] = quat_rotate_inverse(base_quat, ang_vel)
            self.commands[:,6:] = body_vel.view(-1,12)

            if (self.common_step_counter +1)% self.target_jt_update_steps_int == 0:
                self.target_jt_j += 1
            
            
            #self.__upd_root() # view


    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # step physics and render each frame
        self.render()
        
        if self.action_delay != -1:
            self.action_history_buf = torch.cat([self.action_history_buf[:, 1:], actions[:, None, :]], dim=1)
            actions = self.action_history_buf[:, -self.action_delay - 1] # delay for 1/50=20ms
        self.actions = actions.clone()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        # clip_obs = self.cfg.normalization.clip_observations
        # self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        # if self.privileged_obs_buf is not None:
        #     self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_history_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_lin_acc[:] = (self.base_lin_vel - self.last_base_lin_vel)/self.dt
        self.base_orn_rp[:] = self.get_body_orientation()
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.update_target_jt()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # left_leg = quat_rotate_inverse(self.base_quat, self.body_pos[:,5]-self.root_states[:,:3]) #left ankle: 5 right ankle:11
        # right_leg = quat_rotate_inverse(self.base_quat, self.body_pos[:,11]-self.root_states[:,:3]) #left ankle: 5 right ankle:11
        # cross_leg = ((left_leg[:,1]<0) | (right_leg[:,1] > 0))
        # leg_air = ((self.body_pos[:,5,2]>0.20) | (self.body_pos[:,11,2] > 0.20))
        # hands_up = ((self.body_pos[:,22,2]>1.1) | (self.body_pos[:,29,2] > 1.1))

        # yaw_leg = ((torch.abs(self.dof_pos[:, 2]) > 0.5) | (torch.abs(self.dof_pos[:, 8] > 0.5)))

        # knee_pos = ((self.dof_pos[:, 3] < 0.1) | (self.dof_pos[:, 9] < 0.1))

        too_far = torch.norm(self.root_states[:, :2] - (self.init_pos[:,:2] + self.target_jt[:,:2] - self.init_target_pos[:,:2]), dim=-1) > 1.

        r, p = self.base_orn_rp[:, 0], self.base_orn_rp[:, 1]
        z = self.root_states[:, 2]

        r_threshold_buff = r.abs() > self.cfg.termination.r_threshold
        p_threshold_buff = p.abs() > self.cfg.termination.p_threshold
        z_threshold_buff = z < self.cfg.termination.z_threshold
        

        self.time_out_buf = (self.episode_length_buf > self.max_episode_length) 
        if self.cfg.commands.command_type == 'target':
            self.time_out_buf = self.time_out_buf | (self.target_jt_j >= self.target_jt_seq[0].shape[0]-2) # no terminal reward for time-outs

        # self.reset_triggers = torch.stack([termination_contact_buf, r_threshold_buff, p_threshold_buff, z_threshold_buff, self.time_out_buf], dim=-1).nonzero(as_tuple=False)
        # if len(self.reset_triggers) > 0:
        #     print('reset_triggers: ', self.reset_triggers)

        self.reset_buf = too_far | termination_contact_buf | r_threshold_buff | p_threshold_buff | z_threshold_buff | self.time_out_buf
    

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_jt(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode_metrics"] = deepcopy(self.episode_metrics)
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            unscaled_rew, metric = self.reward_functions[i]()
            rew = unscaled_rew * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.episode_metrics[name] = metric.mean().item()
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_orn_rp * self.obs_scales.orn,  # [0:2]
                                    self.base_ang_vel * self.obs_scales.ang_vel,  # [2:5]
                                    self.base_lin_acc * self.obs_scales.lin_acc,
                                    self.commands * self.commands_scale,  # [5:8]
                                    self.dof_pos * self.obs_scales.dof_pos,  # [8:8+num_dofs]
                                    self.dof_vel * self.obs_scales.dof_vel,  # [8+num_dofs:8+2*num_dofs]
                                    self.actions,  # [8+2*num_dofs:8+3*num_dofs]
                                    ),dim=-1)
        # print(self.target_jt_j[:3], self.target_jt_i[:3])
        if self.cfg.commands.command_type == 'target':
            obs_target = self.delayed_obs_target_jt * self.obs_scales.dof_pos
            if self.cfg.domain_rand.drop_target:
                obs_target *= (torch.rand(self.dof_pos.shape[1],device=self.device)>self.target_jt_exp)
        else:
            obs_target = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float32, device=self.device)
        self.obs_buf = torch.cat([self.obs_buf, obs_target], dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.cfg.noise.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            self.obs_buf *= (torch.rand_like(self.obs_buf)-0.5)*(2*self.cfg.noise.noise_ratio) + 1.0

        self.obs_history_buf = torch.cat([
            self.obs_history_buf[:, 1:],
            self.obs_buf.unsqueeze(1)
        ], dim=1)

    def get_body_orientation(self, return_yaw=False):
        r, p, y = euler_from_quat(self.base_quat)
        if return_yaw:
            return torch.stack([r, p, y], dim=-1)
        else:
            return torch.stack([r, p], dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_powlim = pow(1.2,25.)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.command_type == 'heading':
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 5] = torch.clip(0.5*wrap_to_pi(self.heading - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.cfg.commands.command_type == 'heading':
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.heading[env_ids] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # set small commands to zero
        elif self.cfg.commands.command_type == 'vel_ang':
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 5] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)            
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        
        self.actions_scaled[:,self.execute_dof] = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            target_dof_pos = self.actions_scaled + self.default_dof_pos
            if self.cfg.control.clip_actions:
                target_dof_pos = torch.clip(target_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            torques = self.p_gains*(target_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(self.actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = self.actions_scaled
        elif control_type=='C':
            target_dof_pos = self.target_jt[7:36]
            if self.cfg.control.clip_actions:
                target_dof_pos = torch.clip(target_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
            torques = self.p_gains*(target_dof_pos - self.dof_pos + self.actions_scaled) - self.d_gains*self.dof_vel
        elif control_type=="D": #discrete control
            self.torques[:,self.execute_dof] = (self._torque_tj(self.actions)/self.torque_table[-1])
            self.torques *= self.torque_limits
            self.torques = self.torques*0.8 + self.last_torques*0.2
            self.last_torques[:] = self.torques[:]
            return self.torques
        elif control_type=="W":
            torques = (torch.pow(1.2, torch.abs(self.actions_scaled*25))-1)*((self.actions_scaled<0)*-1.+(self.actions_scaled>=0))
            torques *= self.torque_limits/self.torque_powlim
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        ret = torch.clip(torques, -self.torque_limits, self.torque_limits)
        self.limit_torque[:] = torques - ret

        return ret





    def _reset_jt(self, env_ids):
        if self.cfg.commands.command_type == "target":
            self.target_jt_i[env_ids] = torch.randint(0, self.num_target_jt_seq, (env_ids.shape[0],), device=self.device)
            self.target_jt_j[env_ids] = torch.randint(0,self.target_jt_seq.shape[1] - self.cfg.human.least_time*self.cfg.human.freq, (env_ids.shape[0],), dtype=torch.long, device=self.device)
            self.init_target_pos[env_ids] = self.target_jt_seq[self.target_jt_i[env_ids], self.target_jt_j[env_ids],:3]
            self.update_target_jt()
        
        #self.target_jt[env_ids] = self.target_jt_seq[self.target_jt_i[env_ids], self.target_jt_j[env_ids]]
        #self.target_jt_j[env_ids] = torch.randint(0,1, (env_ids.shape[0],), dtype=torch.long, device=self.device)


    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.cfg.commands.command_type == "target":
            self.dof_pos[env_ids] = self.target_jt[env_ids,7:36] + torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = (self.target_jt_next[env_ids, 7:36] - self.target_jt[env_ids, 7:36]) * self.cfg.human.freq * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dofs), device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dofs), device=self.device)
            self.dof_vel[env_ids] = torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dofs), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.init_pos[env_ids] = self.root_states[env_ids, :3]

        # base velocities
        #self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.commands.command_type == 'target':
            self.root_states[env_ids, 2] += self.target_jt[env_ids,2]
            self.root_states[env_ids, 3:7] = self.target_jt[env_ids,3:7]
            self.root_states[env_ids, 7:10] = (self.target_jt_next[env_ids,:3] - self.target_jt[env_ids,:3])*self.cfg.human.freq*torch_rand_float(0.9, 1.1, (len(env_ids), 3), device=self.device) 
            self.root_states[env_ids, 10:13] = quaternion_to_angular_velocity(self.target_jt[env_ids,3:7],self.target_jt_next[env_ids,3:7],  1/self.cfg.human.freq) #dq/dt = 1/2*w*q
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def __upd_root(self):
        self.root_states[:, :2] = self.init_pos[:, :2] + self.target_jt[:,:2]-self.init_target_pos[:,:2]
        self.root_states[:, 3:7] = self.target_jt[:,3:7]
        self.root_states[:, 7:10] = (self.target_jt_next[:,:3] - self.target_jt[:,:3])*self.cfg.human.freq
        self.root_states[:, 10:13] = quaternion_to_angular_velocity(self.target_jt[:,3:7],self.target_jt_next[:,3:7],  1/self.cfg.human.freq) #dq/dt = 1/2*w*q

        self.dof_pos[:] = self.target_jt[:,7:36]
        self.dof_vel[:] = (self.target_jt_next[:, 7:36] - self.target_jt[:, 7:36])
        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] += torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:2] = noise_scales.orn * noise_level * self.obs_scales.orn
        noise_vec[2:5] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[5:23] = 0. # commands
        noise_vec[23: 23 + self.num_dofs] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[23 + self.num_dofs: 23 + 2 * self.num_dofs] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[23 + 2 * self.num_dofs: 23 + 3 * self.num_dofs] = 0 # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[-self.terrain.num_height_points:] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) #p,o,v,a
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.body_state = gymtorch.wrap_tensor(body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, -1)[...,:3]
        self.body_vel = self.body_state.view(self.num_envs, self.num_bodies, -1)[...,7:10]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        if self.cfg.noise.add_noise:
            self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.limit_torque = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_scaled = torch.zeros((self.num_envs,self.num_dofs), dtype=torch.float32, device=self.device) #在这里补齐空缺
        self.execute_dof = torch.zeros(self.num_actions, dtype=torch.long, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(self.cfg.normalization.commands_scale, device=self.device, requires_grad=False,) # TODO change this
        self.heading = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_lin_acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.last_base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.base_orn_rp = self.get_body_orientation() # [r, p]
        self.act_table = torch.zeros(self.cfg.control.discrete_lev*2+1,dtype=torch.float, device=self.device)
        self.torque_table = torch.zeros(self.cfg.control.discrete_lev*2+1,dtype=torch.float, device=self.device)
        self.init_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.init_target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        
        self._init_discrete_table()
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        assert(np.all([name1 == name2 for name1, name2 in zip(self.dof_names, self.cfg.init_state.default_joint_angles.keys())]))
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_weights = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        ndof = 0
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            weights = self.cfg.init_state.dof_weights[name]
            self.default_dof_pos[i] = angle
            self.dof_weights[i] = weights
            found = False
            for dof_name in self.cfg.asset.exclude_dof:
                if dof_name in name:
                    found=True
            if not found:
                self.execute_dof[ndof]=i
                ndof += 1
            
            found=False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    found=True
            if not found:
                self.p_gains[i] = self.cfg.control.stiffness["default"]
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD stiffness of joint {name} were not defined, setting them to default")

            found=False
            for dof_name in self.cfg.control.damping.keys():
                if dof_name in name:
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.d_gains[i] = self.cfg.control.damping["default"]
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD damping of joint {name} were not defined, setting them to default")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    def _init_discrete_table(self):
        tblen = self.act_table.shape[0]-1
        self.act_table = torch.tensor([i/tblen for i in range(tblen+1)], dtype=torch.float, device=self.device)
        self.act_table = self.act_table - self.cfg.control.discrete_lev/tblen
        self.act_table *= 2
        self.act_table[-1] = torch.inf
        rg = torch.arange(1, self.cfg.control.discrete_lev+1, dtype=torch.float, device=self.device)
        rg = torch.pow(1.2, rg)-1.
        self.torque_table[[i for i in range(self.cfg.control.discrete_lev-1,-1,-1)]] = -rg
        self.torque_table[self.cfg.control.discrete_lev+1:self.cfg.control.discrete_lev*2+1] = rg
    def _torque_tj(self, act):
        return self.torque_table[torch.searchsorted(self.act_table, act)]
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.episode_metrics = {name: 0 for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        assert(self.num_bodies == len(body_names))
        assert(self.num_dofs == len(self.dof_names))
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.symmetry_zero_idx = []
        for name in self.cfg.rewards.symmetry_zero:
            self.symmetry_zero_idx.extend([self.dof_names.index(s) for s in self.dof_names if name in s])
        self.symmetry_same_idx_l=[self.dof_names.index(s) for s in self.dof_names if 'left' in s and 'yaw' not in s and 'roll' not in s]
        self.symmetry_same_idx_r=[self.dof_names.index(s) for s in self.dof_names if 'right' in s and 'yaw' not in s and 'roll' not in s]
        self.symmetry_opp_idx_l=[self.dof_names.index(s) for s in self.dof_names if 'left' in s and ('yaw' in s or 'roll' in s)]
        self.symmetry_opp_idx_r=[self.dof_names.index(s) for s in self.dof_names if 'right' in s and ('yaw' in s or 'roll' in s)]

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0, -0., -0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.arrows = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            # if self.debug_viz:
            #     # Create cone (tip of the arrow)    
            #     cone = self.gym.create_capsule(self.sim, 0.02, 0.1)

            #     # Create the cylinder and cone actors in the environment
            #     self.arrows.append(self.gym.create_actor(env_handle, cone, gymapi.Transform()))

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.target_pos_idx = torch.zeros(len(self.cfg.asset.target_pos), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.target_pos)):
            self.target_pos_idx[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.target_pos[i])

        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        

        print('penalized_contact_indices: {}'.format(self.penalized_contact_indices))
        print('termination_contact_indices: {}'.format(self.termination_contact_indices))
        print('feet_indices: {}'.format(self.feet_indices))

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.action_delay = self.cfg.env.action_delay

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if hasattr(self, 'terrain') and self.terrain.cfg.measure_heights:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
        if self.cfg.commands.command_type == 'target':
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                lin_vel = quat_rotate(self.root_states[[i], 3:7], self.commands[[i],:3])
                pos = self.root_states[i, :3]
                # norm = np.linalg.norm(lin_vel)
                # if norm > 0:
                #     direction = lin_vel / norm
                #     z_axis = np.array([0, 0, 1])
                #     cross_prod = np.cross(z_axis, direction)
                #     dot_prod = np.dot(z_axis, direction)
                #     angle = np.arccos(dot_prod)
                #     if np.linalg.norm(cross_prod) != 0:
                #         cross_prod = cross_prod / np.linalg.norm(cross_prod)
                    
                #     # Create a quaternion from the axis-angle representation
                #     rotation_quat = gymapi.Quat.from_axis_angle(cross_prod, angle)    
                #     self.gym.set_actor_pose(self.arrows[i], gymapi.Pose(pos, rotation_quat))
                p1 = (lin_vel[0]+pos).cpu().numpy()
                p2 = pos.cpu().numpy()
                gymutil.draw_line(gymapi.Vec3(p1[0],p1[1],p1[2]),gymapi.Vec3(p2[0],p2[1],p2[2]),gymapi.Vec3(1,0,0),self.gym, self.viewer, self.envs[i])
                ang_vel = quat_rotate(self.root_states[[i], 3:7], self.commands[[i],3:6])
                p1 = (ang_vel[0]*2+pos).cpu().numpy()
                gymutil.draw_line(gymapi.Vec3(p1[0],p1[1],p1[2]),gymapi.Vec3(p2[0],p2[1],p2[2]),gymapi.Vec3(0,1,0),self.gym, self.viewer, self.envs[i])


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def create_camera(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1024
        camera_props.height = 768
        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)
        self.gym.set_camera_location(self.camera_handle, self.envs[0], gymapi.Vec3(2,1.2,1.5), gymapi.Vec3(0.,0.,0.8))

    def save_video(self, path):
        self.video_frames = []
        self.video_step = 0
        self.video_path = path

    def refresh_video(self):
        if self.video_step < 0:
            return
        if self.video_step >= int(1/self.dt)*self.cfg.viewer.video_len:
            writer = imageio.get_writer(self.video_path, fps=int(1/self.dt))
            for frame in self.video_frames:
                writer.append_data(frame)
            writer.close()
            self.video_frames = []
            self.video_step = -1
        else:
            self.gym.render_all_camera_sensors(self.sim)
            rgb_image = self.gym.get_camera_image(self.sim,self.envs[0],self.camera_handle,gymapi.ImageType.IMAGE_COLOR)
            rgb_image_np = rgb_image.reshape(768, 1024, 4)
            self.video_frames.append(rgb_image_np[..., :3])  #
            self.video_step += 1

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            # if self.gym.query_viewer_has_closed(self.viewer):
            #     sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results


            # step graphics
            if self.enable_viewer_sync:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.refresh_video()
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                if self.video_step >= 0:
                    self.gym.fetch_results(self.sim, True)
                    self.gym.step_graphics(self.sim)
                    self.refresh_video()
                    if sync_frame_time:
                        self.gym.sync_frame_time(self.sim)
                self.gym.poll_viewer_events(self.viewer)

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        err = torch.sum(torch.square(self.torques), dim=1)
        return torch.exp(-err/40000), err
    
    def _reward_torques_limit(self):
        # Penalize torques
        err = torch.sum(torch.abs(self.limit_torque), dim=1)
        return torch.exp(-err/200), err

    def _reward_dof_vel(self):
        # Penalize dof velocities
        dof_err = torch.sum(torch.square((self.dof_vel)), dim=1)
        return torch.exp(-dof_err/2500), dof_err
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        dof_err = torch.sum(torch.square((self.last_dof_vel - self.dof_vel)), dim=1)
        return torch.exp(-dof_err/10000), dof_err
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        err = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        return torch.exp(-err/100), err
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        err = torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1), dim=1)
        return torch.exp(-err/50), err
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return (self.reset_buf * ~self.time_out_buf)*(self.episode_length_buf+100.).sqrt()/10.
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        err = torch.sum(out_of_limits, dim=1)
        return torch.exp(-0.1*err), err

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        err = torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
        return torch.exp(-0.1*err), err
    
    def _reward_tracking_pos(self):
        err = torch.norm(self.root_states[:, :2] - (self.init_pos[:,:2] + self.target_jt[:,:2] - self.init_target_pos[:,:2]), dim=-1)
        return torch.exp(-1.*err), err

    def _reward_tracking_ornt(self):
        err_quat = torch.abs(quat_mul(self.target_jt[:,3:7], quat_conjugate(self.root_states[:, 3:7]))[:,3])    # 旋转偏差角度
        return torch.exp(-1.*err_quat), err_quat

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :3] - self.base_lin_vel[:, :3]), dim=1)
        return torch.exp(-lin_vel_error/0.2), lin_vel_error
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.sum(torch.square(self.commands[:, 3:6] - self.base_ang_vel), dim=1)
        return torch.exp(-ang_vel_error/5), ang_vel_error

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime,rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_err = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return torch.exp(-0.1 * dof_err),dof_err

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_target_jt(self):
        target_jt_error = torch.sum(torch.abs(self.dof_pos - self.target_jt[:,7:36])*self.dof_weights, dim=1)/self.dof_weights.sum()
        target_jt_exp = torch.exp(-4 * target_jt_error)
        self.target_jt_exp = torch.mean(target_jt_exp, dim=-1)
        return target_jt_exp, target_jt_error

    def _reward_target_vel(self):
        #Penalize body vel error
        body_vel = self.body_vel[:,self.target_pos_idx]
        for i in range(body_vel.shape[1]):
            body_vel[:,i] = quat_rotate_inverse(self.base_quat, body_vel[:,i])
        vel_cmd = self.commands[:,6:18].view(-1,4,3)
        target_pos_err = torch.sum(torch.norm(body_vel-vel_cmd, dim=-1)*(vel_cmd.norm(dim=-1) > 0.2), dim=-1)
        return torch.exp(-0.1 * target_pos_err), target_pos_err
    
    def _reward_feet_slide(self) -> torch.Tensor:
        """Penalize feet sliding.

        This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
        norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
        agent is penalized only when the feet are in contact with the ground.
        """
        # Penalize feet sliding
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        body_vel = self.body_vel[:, self.feet_indices]
        reward = torch.sum(body_vel.norm(dim=-1) * contact, dim=1)
        return reward, reward
    
    def _reward_height(self) -> torch.Tensor:
        height_err = 0.65 - self.root_states[:,2]
        height_err = height_err*(height_err>0)
        return height_err, height_err

    def _reward_torso_stable(self) -> torch.Tensor:
        err = torch.sum(torch.square(self.body_vel[:,15]), dim=-1)
        return torch.exp(-25 * err),err

    def _reward_symmetry(self) -> torch.Tensor:
        err1 = torch.abs(self.dof_pos[:,self.symmetry_opp_idx_l]+self.dof_pos[:,self.symmetry_opp_idx_r])
        err2 = torch.abs(self.dof_pos[:,self.symmetry_same_idx_l]-self.dof_pos[:,self.symmetry_same_idx_r])
        err3 = torch.abs(self.dof_pos[:,self.symmetry_zero_idx])
        err = err1.sum(dim=-1)+err2.sum(dim=-1)+err3.sum(dim=-1)
        return torch.exp(-1 * err), err
 