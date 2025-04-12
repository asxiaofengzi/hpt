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

from legged_gym.envs.base.base_config import BaseConfig

class G1RoughCfg( BaseConfig ):
    class human:
        delay = 0.0 # delay in seconds
        freq = 20
        resample_on_env_reset = True
        filename = '/mnt/data1/xiaofengzi/hpt/HST/legged_gym/data/motion.csv'
        least_time = 2.5 #至少要学习10秒
    class env:
        num_envs = 1
        num_dofs = 29
        num_observations = 29*3 # TODO
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 29*3
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

        action_delay = -1  # -1 for no delay
        obs_context_len = 4

    class terrain:
        mesh_type = "plane" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 0.8
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 18 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        command_type = 'target' #'heading', 'vel_ang','target'
        class ranges:
            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class init_state:
        pos = [0.0, 0.0, 0.893] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        dof_weights = {
            'left_hip_pitch_joint':1.0,
            'left_hip_roll_joint':1.0, 
            'left_hip_yaw_joint':1.0, 
            'left_knee_joint':0.5, 
            'left_ankle_pitch_joint':0.1, 
            'left_ankle_roll_joint':0.1, 
            'right_hip_pitch_joint':1.0, 
            'right_hip_roll_joint':1.0, 
            'right_hip_yaw_joint':1.0, 
            'right_knee_joint':0.5, 
            'right_ankle_pitch_joint':0.1, 
            'right_ankle_roll_joint':0.1, 
            'waist_yaw_joint':1.0, 
            'waist_roll_joint':1.0, 
            'waist_pitch_joint':1.0, 
            'left_shoulder_pitch_joint':1.0, 
            'left_shoulder_roll_joint':1.0, 
            'left_shoulder_yaw_joint':1.0, 
            'left_elbow_joint':1.0, 
            'left_wrist_roll_joint':1.0, 
            'left_wrist_pitch_joint':1.0, 
            'left_wrist_yaw_joint':1.0, 
            'right_shoulder_pitch_joint':1.0, 
            'right_shoulder_roll_joint':1.0, 
            'right_shoulder_yaw_joint':1.0, 
            'right_elbow_joint':1.0, 
            'right_wrist_roll_joint':1.0, 
            'right_wrist_pitch_joint':1.0, 
            'right_wrist_yaw_joint':1.0
        }

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_pitch_joint':-0.28,   #0
            'left_hip_roll_joint':0.0,      #1
            'left_hip_yaw_joint':0.0,       #2
            'left_knee_joint':0.69,         #3
            'left_ankle_pitch_joint':-0.42, #4
            'left_ankle_roll_joint':0.0,    #5
            'right_hip_pitch_joint':-0.28,  #6
            'right_hip_roll_joint':0.0,     #7
            'right_hip_yaw_joint':0.0,      #8
            'right_knee_joint':0.69,        #9
            'right_ankle_pitch_joint':-0.42, #10
            'right_ankle_roll_joint':0.0,   #11
            'waist_yaw_joint':0.0,          #12
            'waist_roll_joint':0.0,         #13
            'waist_pitch_joint':0.0,        #14
            'left_shoulder_pitch_joint':0.05, #15
            'left_shoulder_roll_joint':0.16, #16
            'left_shoulder_yaw_joint':-0.3, #17
            'left_elbow_joint':0.80,        #18
            'left_wrist_roll_joint':0.0,    #19
            'left_wrist_pitch_joint':0.0,   #20
            'left_wrist_yaw_joint':0.0,     #21
            'right_shoulder_pitch_joint':0.05, #22
            'right_shoulder_roll_joint':-0.16, #23
            'right_shoulder_yaw_joint':0.3, #24
            'right_elbow_joint':0.80,       #25
            'right_wrist_roll_joint':0.0,   #26
            'right_wrist_pitch_joint':0.0,  #27
            'right_wrist_yaw_joint':0.0     #28
        }

    class control:
        control_type = 'VPID'
        # PD Drive parameters:
        stiffness = {'default': 100.,
                    'hip_yaw_joint':150.,
                    "hip_roll_joint": 150.0,
                    "hip_pitch_joint": 200.0,
                    "knee_joint": 200.0,
                    "waist_pitch_joint": 200.0,
                    "_ankle_": 40.,
                    "_shoulder_": 40.,
                    "_elbow_": 40.
                     }  # [N*m/rad]
        damping = {'default': 5.,
                   "_ankle_": 2.,
                   "_shoulder_": 10.,
                    "_elbow_": 10.
                   }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1
        clip_actions = False
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        discrete_lev = 25
        
    class asset:
        file = '/mnt/data1/xiaofengzi/hpt/HST/legged_gym/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        arrow=''
        name = "g1"
        foot_name = '_ankle_roll_link'
        penalize_contacts_on = [""]
        terminate_after_contacts_on = ["knee"]
        target_pos = ['left_ankle_pitch_link','right_ankle_pitch_link','left_wrist_yaw_link','right_wrist_yaw_link']
        exclude_dof = []#['wrist','ankle','knee','hip']
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = True # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.01
        thickness = 0.01

    class domain_rand:
        drop_target=True
        randomize_friction = True
        friction_range = [0.4, 0.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        motion_package_loss=False
        package_loss_interval_s=0

    class rewards:
        class scales:
            termination = -0
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0
            ang_vel_xy = -0
            orientation = -0.
            tracking_pos = 0.
            tracking_ornt = 0.
            torques = 0.0
            torques_limit = 0.01
            dof_vel = 0.0
            dof_acc = 0.0
            feet_air_time = 0.
            collision = 0.0
            feet_stumble = -0.0 
            action_rate = 0.0
            stand_still = 0
            dof_pos_limits = 0.1
            dof_vel_limits = 0.1
            target_jt = 50
            target_pos = 0
            target_jt_vel = 1.
            feet_slide = -0.0
            height = -0.0
            torso_stable = 0.
            symmetry = 0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        symmetry_zero = ['waist_roll_joint','waist_yaw_joint']

    class termination:
        r_threshold = 0.7
        p_threshold = 0.7
        z_threshold = 0.3

    class normalization:
        class obs_scales:
            # lin_vel = 1.0
            ang_vel = 0.25
            orn = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            lin_vel = 0.05
            lin_acc = 0.2
            height_measurements = 5.0
        commands_scale = [1., 1., 1., 1., 1.0, 1.0,
                          1., 1., 1., 1., 1.0, 1.0,
                          1., 1., 1., 1., 1.0, 1.0]
        # clip_observations = 100.
        # clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 0.1 # scales other values
        noise_ratio = 0.002
        class noise_scales:
            dof_pos = 0.02
            dof_vel = 0.05
            lin_vel = 0.05
            orn = 0.05
            ang_vel = 0.05
            # gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        video = True
        video_len = 10.

    class sim:
        dt =  0.005
        substeps = 2
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class G1RoughCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.9
        #transformer ppo 已不用这些参数
        actor_hidden_dims = [128, 128, 64]   
        critic_hidden_dims = [128, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        #only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 128
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 1e-5
        num_learning_epochs = 2
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5e-5
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32 # per iteration
        max_iterations = 1000000 # number of policy updates

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'rough_g1_pos'
        run_name = "g1_pos"
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
