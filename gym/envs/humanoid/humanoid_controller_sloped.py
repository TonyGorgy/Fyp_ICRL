import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from gym.envs.humanoid.humanoid_controller_sloped_config import HumanoidControllerSlopedCfg
from gym.utils.math import *
from gym.envs import LeggedRobot
from isaacgym import gymapi, gymutil
import numpy as np
from typing import Tuple, Dict
from .humanoid_utils import (
    FootStepGeometry, SimpleLineGeometry, VelCommandGeometry,
    smart_sort, FootStepGeometry3D
)
from gym.utils import XCoMKeyboardInterface
from .jacobian import apply_coupling
from scipy.signal import correlate
import torch.nn.functional as F
import random, os


class HumanoidControllerSloped(LeggedRobot):
    """Humanoid controller for sloped terrain environments."""

    cfg: HumanoidControllerSlopedCfg

    def __init__(self, cfg, sim_params, physics_engine, simu_device, headless):
        """
        Parse the configuration file, create the simulation (terrain + envs),
        and initialise all PyTorch buffers used during training.

        Args:
            cfg (HumanoidControllerSlopedCfg): Environment configuration.
            sim_params (gymapi.SimParams): Simulation parameters.
            physics_engine (gymapi.SimType): Gym physics backend (must be PhysX).
            simu_device (str): 'cuda' or 'cpu'.
            headless (bool): Disable rendering if True.
        """
        super().__init__(cfg, sim_params, physics_engine, simu_device, headless)

    # --------------------------------------------------------------------- #
    #                          Keyboard Interface                           #
    # --------------------------------------------------------------------- #
    def _setup_keyboard_interface(self):
        """Bind the keyboard interface for interactive velocity commands."""
        self.keyboard_interface = XCoMKeyboardInterface(self)

    # --------------------------------------------------------------------- #
    #                          Buffer Allocation                            #
    # --------------------------------------------------------------------- #
    def _init_buffers(self):
        """Allocate and register all state / command buffers."""
        super()._init_buffers()  # Parent initialisation

        # ------------------------------ Robot ----------------------------- #
        # Base state
        self.base_height = self.root_states[:, 2:3]                           # z-height of the base
        self.current_support_foot_height = torch.zeros(                      # z-height of current support foot
            self.num_envs, 1, dtype=torch.float, device=self.device
        )
        self.relative_base_height = self.base_height - self.current_support_foot_height

        # Hip positions
        self.right_hip_pos = self.rigid_body_state[:, self.rigid_body_idx['right_hip_yaw'], :3]
        self.left_hip_pos = self.rigid_body_state[:, self.rigid_body_idx['left_hip_yaw'], :3]

        # Center-of-mass position
        self.CoM = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Foot states: [pos(3) | quat(4)]
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7,
                                       dtype=torch.float, device=self.device)
        self.support_foot_cosine_average = torch.zeros(                      # Average surface normal cos similarity
            self.num_envs, 1, dtype=torch.float, device=self.device
        )

        # Individual foot state buffers [x, y, z, heading|proj-gravity]
        self.foot_states_right = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                             device=self.device)
        self.foot_states_left = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                            device=self.device)

        self.foot_heading = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.float, device=self.device)
        self.foot_projected_gravity = torch.stack((self.gravity_vec, self.gravity_vec), dim=1)
        self.foot_contact = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.bool, device=self.device)
        # Two frames of ankle velocity history (each frame: 3-D vel)
        self.ankle_vel_history = torch.zeros(self.num_envs, len(self.feet_ids), 2 * 3,
                                             dtype=torch.float, device=self.device)

        self.base_heading = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.base_lin_vel_world = torch.zeros(self.num_envs, 3, dtype=torch.float,
                                              device=self.device)

        # ------------------------------ Commands -------------------------- #
        self.step_commands = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                         dtype=torch.float, device=self.device)      # [x, y, heading]
        self.step_commands_right_foot = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                               device=self.device)
        self.step_commands_left_foot = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                              device=self.device)
        self.foot_on_motion = torch.zeros(self.num_envs, len(self.feet_ids),
                                          dtype=torch.bool, device=self.device)
        self.step_period = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.full_step_period = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.ref_foot_trajectories = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                                 dtype=torch.float, device=self.device)

        # ---------------------------- Step State -------------------------- #
        self.current_step = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                        dtype=torch.float, device=self.device)
        self.prev_step_commands = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                              dtype=torch.float, device=self.device)
        self.step_location_offset = torch.zeros(self.num_envs, len(self.feet_ids),
                                                dtype=torch.float, device=self.device)
        self.step_location_offset_projected = torch.zeros_like(self.step_location_offset)
        self.step_heading_offset = torch.zeros_like(self.step_location_offset)
        self.ankle_torque = torch.zeros(self.num_envs, len(self.feet_ids),
                                        dtype=torch.float, device=self.device)

        # Success thresholds
        self.succeed_step_radius = torch.tensor(self.cfg.commands.succeed_step_radius,
                                                dtype=torch.float, device=self.device)
        self.succeed_step_angle = torch.tensor(np.deg2rad(self.cfg.commands.succeed_step_angle),
                                               dtype=torch.float, device=self.device)

        self.semi_succeed_step = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.bool,
                                             device=self.device)
        self.succeed_step = torch.zeros_like(self.semi_succeed_step)
        self.already_succeed_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.error_contact = torch.zeros_like(self.semi_succeed_step)
        self.step_stance = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)

        # ------------------------------ Misc ------------------------------ #
        self.update_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_commands_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.update_phase_ids = torch.zeros_like(self.update_commands_ids)
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.capture_point = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

        # Raibert-style heuristic buffers
        self.raibert_heuristic = torch.zeros(self.num_envs, len(self.feet_ids), 3,
                                             dtype=torch.float, device=self.device)
        self.w = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.step_length = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.step_width = torch.zeros_like(self.step_length)
        self.dstep_length = torch.zeros_like(self.step_length)
        self.dstep_width = torch.zeros_like(self.step_length)

        self.support_foot_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.support_foot_orientation = torch.zeros(self.num_envs, len(self.feet_ids), 4,
                                                    dtype=torch.float, device=self.device)
        self.prev_support_foot_pos = torch.zeros_like(self.support_foot_pos)
        self.LIPM_CoM = torch.zeros_like(self.support_foot_pos)

        # --------------------------- Observation -------------------------- #
        self.phase_sin = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phase_cos = torch.zeros_like(self.phase_sin)
        self.contact_schedule = torch.zeros_like(self.phase_sin)
        self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)

    # --------------------------------------------------------------------- #
    #                       Environment Origin Handling                     #
    # --------------------------------------------------------------------- #
    def _get_env_origins(self):
        """
        Set environment origins.  
        Heightfield / trimesh terrains use pre-baked origins from the terrain
        object; otherwise a regular grid is generated.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            # Use custom origins defined by the terrain
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1

            # Randomly assign terrain levels (rows) and evenly assign terrain types (cols)
            self.terrain_levels = torch.randint(0, max_init_level + 1,
                                                (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode='floor'
            ).long()

            total_tiles = self.cfg.terrain.num_rows * self.cfg.terrain.num_cols
            if self.num_envs > total_tiles:
                raise ValueError(f"Num envs ({self.num_envs}) exceeds terrain tiles ({total_tiles})")

            # Lookup table of origins defined by the terrain generator
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).float()

            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

        else:
            # Flat grid placement
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)

            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')

            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            # z stays at 0

    # --------------------------------------------------------------------- #
    #                           Torque Computation                          #
    # --------------------------------------------------------------------- #
    def _compute_torques(self):
        """Joint PD controller with optional Jacobian coupling."""
        self.desired_pos_target = self.dof_pos_target + self.default_dof_pos
        q, qd = self.dof_pos.clone(), self.dof_vel.clone()
        q_des, qd_des = self.desired_pos_target.clone(), torch.zeros_like(self.dof_pos_target)
        tau_ff = torch.zeros_like(self.dof_pos_target)
        kp, kd = self.p_gains.clone(), self.d_gains.clone()

        if self.cfg.asset.apply_humanoid_jacobian:
            torques = apply_coupling(q, qd, q_des, qd_des, kp, kd, tau_ff)
        else:
            torques = kp * (q_des - q) + kd * (qd_des - qd) + tau_ff

        torques = torch.clip(torques, -self.torque_limits, self.torque_limits)

        # Expose ankle torques for reward terms / debugging
        self.ankle_torque[:, 0] = torques[:, self.dof_names.index('05_right_ankle')]
        self.ankle_torque[:, 1] = torques[:, self.dof_names.index('10_left_ankle')]
        self.torque_left_ankle = self.torques[:, self.dof_names.index('05_right_ankle')]
        self.torque_right_ankle = self.torques[:, self.dof_names.index('10_left_ankle')]
        if self.cfg.DEBUG.PRINT_ANKLR_TORQUES:
            print("Left-ankle torque:", self.torque_left_ankle)
            print("Right-ankle torque:", self.torque_right_ankle)

        return torques.view(self.torques.shape)

    # --------------------------------------------------------------------- #
    #                         Command Resampling                            #
    # --------------------------------------------------------------------- #
    def _resample_commands(self, env_ids):
        """Randomly sample footstep commands and step periods."""
        super()._resample_commands(env_ids)

        self.step_period[env_ids] = torch.randint(
            low=self.command_ranges["sample_period"][0],
            high=self.command_ranges["sample_period"][1],
            size=(len(env_ids), 1),
            device=self.device
        )
        self.full_step_period = 2 * self.step_period
        self.step_stance[env_ids] = self.step_period[env_ids]

        # Desired step width
        self.dstep_width[env_ids] = torch_rand_float(
            self.command_ranges["dstep_width"][0],
            self.command_ranges["dstep_width"][1],
            (len(env_ids), 1), self.device
        )

    # --------------------------------------------------------------------- #
    #                           Environment Reset                           #
    # --------------------------------------------------------------------- #
    def _reset_system(self, env_ids):
        """Reset episode-specific buffers and initial commands."""
        super()._reset_system(env_ids)

        # Robot state initialisation
        self.foot_states[env_ids] = self._calculate_foot_states(
            self.rigid_body_state[:, self.feet_ids, :7])[env_ids]
        self.foot_projected_gravity[env_ids, :] = self.gravity_vec[env_ids, None, :]

        # Initial footstep commands (place feet shoulder-width apart)
        self.step_commands[env_ids, 0] = self.env_origins[env_ids] + torch.tensor([0., -0.15, 0.],
                                                                                  device=self.device)
        self.step_commands[env_ids, 1] = self.env_origins[env_ids] + torch.tensor([0., 0.15, 0.],
                                                                                  device=self.device)
        self.foot_on_motion[env_ids, 0] = False
        self.foot_on_motion[env_ids, 1] = True   # Left foot swings first

        # Step-related buffers
        self.current_step[env_ids] = self.step_commands[env_ids]
        self.prev_step_commands[env_ids] = self.step_commands[env_ids]
        self.semi_succeed_step[env_ids] = False
        self.succeed_step[env_ids] = False
        self.already_succeed_step[env_ids] = False
        self.error_contact[env_ids] = False

        # Misc counters
        self.update_count[env_ids] = 0
        self.update_commands_ids[env_ids] = False
        self.phase_count[env_ids] = 0
        self.update_phase_ids[env_ids] = False
        self.phase[env_ids] = 0.
        self.capture_point[env_ids] = 0.
        self.raibert_heuristic[env_ids] = 0.
        self.w[env_ids] = 0.
        self.dstep_length[env_ids] = self.cfg.commands.dstep_length
        self.dstep_width[env_ids] = self.cfg.commands.dstep_width

    # --------------------------------------------------------------------- #
    #                       Physics Step Callback                           #
    # --------------------------------------------------------------------- #
    def _post_physics_step_callback(self):
        """Hook executed after each physics simulation step."""
        super()._post_physics_step_callback()

        self._update_robot_states()
        self._calculate_CoM()
        self._calculate_raibert_heuristic()
        self._calculate_capture_point()
        self._measure_success_rate()
        self._update_commands()

        if self.cfg.DEBUG.PRINT_SUPPORTFOOT_HEIGHT:
            print("Support-foot height:", self.current_support_foot_height)
            print("Base-support error:", self.base_height - self.current_support_foot_height)

    # --------------------------------------------------------------------- #
    #                         State-Update Helpers                          #
    # --------------------------------------------------------------------- #
    def _update_robot_states(self):
        """Refresh base, foot and phase-related state variables."""
        self.base_height[:] = self.root_states[:, 2:3]

        forward = quat_apply(self.base_quat, self.forward_vec)
        self.base_heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)

        self.right_hip_pos = self.rigid_body_state[:, self.rigid_body_idx['right_hip_yaw'], :3]
        self.left_hip_pos = self.rigid_body_state[:, self.rigid_body_idx['left_hip_yaw'], :3]

        self.foot_states = self._calculate_foot_states(self.rigid_body_state[:, self.feet_ids, :7])

        # Foot headings
        right_fwd = quat_apply(self.foot_states[:, 0, 3:7], self.forward_vec)
        left_fwd = quat_apply(self.foot_states[:, 1, 3:7], self.forward_vec)
        self.foot_heading[:, 0] = wrap_to_pi(torch.atan2(right_fwd[:, 1], right_fwd[:, 0]))
        self.foot_heading[:, 1] = wrap_to_pi(torch.atan2(left_fwd[:, 1], left_fwd[:, 0]))

        # Project gravity into each foot frame
        self.foot_projected_gravity[:, 0] = quat_rotate_inverse(self.foot_states[:, 0, 3:7],
                                                                self.gravity_vec)
        self.foot_projected_gravity[:, 1] = quat_rotate_inverse(self.foot_states[:, 1, 3:7],
                                                                self.gravity_vec)

        # Phase counters
        self.update_count += 1
        self.phase_count += 1
        self.phase += 1 / self.full_step_period

        # Ground-truth contact
        self.foot_contact = self.contact_forces[:, self.feet_ids, 2] > 0

        # Smooth square-wave contact schedule (phase-based)
        self.contact_schedule = self.smooth_sqr_wave(self.phase)

        # Update current step when contact established
        current_step_masked = self.current_step[self.foot_contact]
        current_step_masked[:, :3] = self.foot_states[self.foot_contact][:, :3]
        self.current_step[self.foot_contact] = current_step_masked

        # Ankle angular velocity history (two-frame rolling buffer)
        n_axes = 3
        self.ankle_vel_history[:, 0, n_axes:] = self.ankle_vel_history[:, 0, :n_axes]
        self.ankle_vel_history[:, 0, :n_axes] = self.rigid_body_state[:, self.rigid_body_idx['right_foot'], 7:10]
        self.ankle_vel_history[:, 1, n_axes:] = self.ankle_vel_history[:, 1, :n_axes]
        self.ankle_vel_history[:, 1, :n_axes] = self.rigid_body_state[:, self.rigid_body_idx['left_foot'], 7:10]

    def _calculate_foot_states(self, foot_states):
        """
        Adjust foot positions by adding sole-to-ankle offsets (-z in foot frame).
        """
        foot_height_vec = torch.tensor([0., 0., -0.04], device=self.device).repeat(self.num_envs, 1)
        r_foot_offset = quat_apply(foot_states[:, 0, 3:7], foot_height_vec)
        l_foot_offset = quat_apply(foot_states[:, 1, 3:7], foot_height_vec)
        foot_states[:, 0, :3] += r_foot_offset
        foot_states[:, 1, :3] += l_foot_offset
        return foot_states

    # -------------------------- Utility (static) -------------------------- #
    def compute_foot_cos_theta(self, foot_orientations: torch.Tensor,
                               angle_threshold_deg: float) -> torch.Tensor:
        """
        Compute cosine similarity between each foot's surface normal and the world z-axis.
        Invalid quaternions (all zeros) return cos(angle_threshold).

        Args:
            foot_orientations (Tensor): shape (env, 2, 4) quaternion per foot.
            angle_threshold_deg (float): Tolerance angle for flat-foot contact.

        Returns:
            Tensor shape (env, 2) containing cos(theta) values.
        """
        num_envs, num_feet = foot_orientations.shape[:2]

        world_up = torch.tensor([0., 0., 1.], device=foot_orientations.device)
        local_z = world_up.expand(num_envs, num_feet, 3)

        # Validity mask
        is_valid = foot_orientations.norm(dim=-1) > 1e-6

        foot_normals = torch.zeros_like(local_z)
        foot_normals[is_valid] = quat_apply(foot_orientations[is_valid], local_z[is_valid])

        cos_theta = (foot_normals * world_up).sum(dim=-1)

        # Replace invalid results with threshold
        angle_threshold_rad = torch.tensor(angle_threshold_deg * torch.pi / 180.0,
                                           device=foot_orientations.device)
        cos_threshold = torch.cos(angle_threshold_rad)
        cos_theta[~is_valid] = cos_threshold
        cos_theta = cos_theta - cos_threshold  # Normalise to [0, 1]

        return cos_theta

    # --------------------------------------------------------------------- #
    #                     Center-of-Mass & Capture Point                    #
    # --------------------------------------------------------------------- #
    def _calculate_CoM(self):
        """Compute center of mass in world frame (optionally height-compensated)."""
        adjusted_state = self.rigid_body_state.clone()
        adjusted_state[..., 2] += self.current_support_foot_height
        self.CoM = (self.rigid_body_state[:, :, :3] *
                    self.rigid_body_mass.unsqueeze(1)).sum(dim=1) / self.mass_total

        if self.cfg.DEBUG.PRINT_CoM:
            print("CoM (compensated):", self.CoM)

    def _calculate_capture_point(self):
        """Instantaneous Capture Point: x_ic = x + ẋ / ω with ω = √(g / z)."""
        g = -self.sim_params.gravity.z
        self.w = torch.sqrt(g / self.CoM[:, 2:3])
        self.capture_point[:, :2] = self.CoM[:, :2] + self.root_states[:, 7:9] / self.w

    def _calculate_raibert_heuristic(self):
        """
        Compute Raibert heuristic positions for each foot:
            r = p_hip + p_sym + p_cent
        where p_sym ~ stance time * velocity error,
              p_cent ~ centrifugal term (ignored here except heading).
        """
        g = -self.sim_params.gravity.z
        k = torch.sqrt(self.CoM[:, 2:3] / g)

        p_sym = 0.5 * self.step_stance * self.dt * self.base_lin_vel_world[:, :2] + \
                k * (self.base_lin_vel_world[:, :2] - self.commands[:, :2])

        self.raibert_heuristic[:, 0, :2] = self.right_hip_pos[:, :2] + p_sym
        self.raibert_heuristic[:, 1, :2] = self.left_hip_pos[:, :2] + p_sym

    # --------------------------------------------------------------------- #
    #                     Step Success & Command Update                     #
    # --------------------------------------------------------------------- #
    def _measure_success_rate(self):
        """
        Evaluate foot placement accuracy and update success flags.
        """
        # Horizontal distance between actual foot landing and target (xy only)
        self.step_location_offset = torch.norm(
            self.foot_states[:, :, :3] -
            torch.cat((self.step_commands[:, :, :2],
                       torch.zeros((self.num_envs, len(self.feet_ids), 1), device=self.device)),
                      dim=2),
            dim=2
        )
        # 3-D distance
        self.step_location_offset_projected = torch.norm(
            self.foot_states[:, :, :3] - self.step_commands[:, :, :3], dim=2
        )

        self.step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.step_commands[:, :, 2])
        )

        self.semi_succeed_step = (self.step_location_offset < self.succeed_step_radius) & \
                                 (self.step_heading_offset < self.succeed_step_angle)

        # Evaluate previous commands (for contact validation)
        self.prev_step_location_offset = torch.norm(
            self.foot_states[:, :, :3] -
            torch.cat((self.prev_step_commands[:, :, :2],
                       torch.zeros((self.num_envs, len(self.feet_ids), 1), device=self.device)),
                      dim=2),
            dim=2
        )
        self.prev_step_heading_offset = torch.abs(
            wrap_to_pi(self.foot_heading - self.prev_step_commands[:, :, 2])
        )
        self.prev_semi_succeed_step = (self.prev_step_location_offset < self.succeed_step_radius) & \
                                      (self.prev_step_heading_offset < self.succeed_step_angle)

        # Contact error mask
        self.error_contact |= self.foot_contact & ~self.semi_succeed_step & ~self.prev_semi_succeed_step

        # Final success flag
        self.succeed_step = self.semi_succeed_step & ~self.error_contact
        self.succeed_step_ids = (self.succeed_step.sum(dim=1) == 2)
        self.already_succeed_step[self.succeed_step_ids] = True

        if self.cfg.DEBUG.PRINT_SUCCEED_STEP:
            print("Succeeded env IDs:", torch.where(self.succeed_step_ids)[0])

    def _update_commands(self):
        """Phase update + generation of new step commands."""
        # ---------------- Update phase counters ---------------- #
        self.update_phase_ids = self.phase_count >= self.full_step_period.squeeze(1)
        self.phase_count[self.update_phase_ids] = 0
        self.phase[self.update_phase_ids] = 0

        # ---------------- Determine which envs resample -------- #
        self.update_commands_ids = self.update_count >= self.step_period.squeeze(1)
        self.already_succeed_step[self.update_commands_ids] = False
        self.error_contact[self.update_commands_ids] = False
        self.update_count[self.update_commands_ids] = 0
        self.step_stance[self.update_commands_ids] = self.step_period[self.update_commands_ids].clone()

        # Toggle swing/support flags
        self.foot_on_motion[self.update_commands_ids] = ~self.foot_on_motion[self.update_commands_ids]

        # Shortcut handles
        update_cmd_mask = self.step_commands[self.update_commands_ids]
        self.prev_step_commands[self.update_commands_ids] = update_cmd_mask.clone()

        # Generate new steps with 3-D LIPM + XCoM
        update_cmd_mask[self.foot_on_motion[self.update_commands_ids]] = \
            self._generate_3dlipm_steps(self.update_commands_ids)

        # Update CoM prediction for logging
        self._update_LIPM_CoM(self.update_commands_ids)

        # Avoid self-collision (feet too close)
        collision_ids = (update_cmd_mask[:, 0, :2] - update_cmd_mask[:, 1, :2]).norm(dim=1) < 0.2
        update_cmd_mask[collision_ids, :, :2] = self._adjust_foot_collision(
            update_cmd_mask[collision_ids, :, :2],
            self.foot_on_motion[self.update_commands_ids][collision_ids]
        )

        # Optional terrain-adaptation of step height
        if self.cfg.terrain.measure_heights:
            update_cmd_mask[self.foot_on_motion[self.update_commands_ids]] = \
                self._adjust_step_command_in_rough_terrain(
                    self.update_commands_ids,
                    update_cmd_mask
                )

        if self.cfg.DEBUG.PRINT_STEP_COMMANDS:
            print("New step commands:", update_cmd_mask)

        self.step_commands[self.update_commands_ids] = update_cmd_mask

    # -------------------- 3-D LIPM Step Generator ------------------------ #
    def _generate_3dlipm_steps(self, update_commands_ids):
        """
        Generate footstep targets using the analytic 3-D LIPM solution combined
        with XCoM heuristics. Only swing feet are updated.
        """
        foot_on_motion = self.foot_on_motion[update_commands_ids]
        step_period = self.step_period[update_commands_ids]
        commands = self.commands[update_commands_ids]
        current_step = self.current_step[update_commands_ids]
        CoM = self.CoM[update_commands_ids]

        T = step_period * self.dt
        w = self.w[update_commands_ids]

        dstep_length = torch.norm(commands[:, :2], dim=1, keepdim=True) * T
        dstep_width = self.dstep_width[update_commands_ids]
        theta = torch.atan2(commands[:, 1:2], commands[:, 0:1])

        right_ids = torch.where(torch.where(foot_on_motion)[1] == 0)[0]
        left_ids = torch.where(torch.where(foot_on_motion)[1] == 1)[0]

        root_states = self.root_states[update_commands_ids]
        support_foot_pos = self.support_foot_pos[update_commands_ids]
        support_foot_pos[right_ids] = current_step[right_ids, 1, :3]  # Left foot supports
        support_foot_pos[left_ids] = current_step[left_ids, 0, :3]   # Right foot supports

        # For logging
        rfx = torch.cos(theta) * current_step[:, 0, 0:1] + torch.sin(theta) * current_step[:, 0, 1:2]
        rfy = -torch.sin(theta) * current_step[:, 0, 0:1] + torch.cos(theta) * current_step[:, 0, 1:2]
        lfx = torch.cos(theta) * current_step[:, 1, 0:1] + torch.sin(theta) * current_step[:, 1, 1:2]
        lfy = -torch.sin(theta) * current_step[:, 1, 0:1] + torch.cos(theta) * current_step[:, 1, 1:2]

        self.step_length[update_commands_ids] = torch.abs(rfx - lfx)
        self.step_width[update_commands_ids] = torch.abs(rfy - lfy)
        self.dstep_length[update_commands_ids] = dstep_length
        self.dstep_width[update_commands_ids] = dstep_width

        # Allocate output
        new_step_cmd = torch.zeros(foot_on_motion.sum(), 3, dtype=torch.float, device=self.device)

        x_0 = CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = root_states[:, 7:8]
        vy_0 = root_states[:, 8:9]

        # Analytic LIPM propagation
        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        vx_f = x_0 * w * torch.sinh(T * w) + vx_0 * torch.cosh(T * w)
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w
        vy_f = y_0 * w * torch.sinh(T * w) + vy_0 * torch.cosh(T * w)

        x_f_world = x_f + support_foot_pos[:, 0:1]
        y_f_world = y_f + support_foot_pos[:, 1:2]
        ecapture_point_x = x_f_world + vx_f / w
        ecapture_point_y = y_f_world + vy_f / w

        b_x = dstep_length / (torch.exp(T * w) - 1)
        b_y = dstep_width / (torch.exp(T * w) + 1)

        original_offset_x = -b_x
        original_offset_y = -b_y
        original_offset_y[left_ids] = b_y[left_ids]

        offset_x = torch.cos(theta) * original_offset_x - torch.sin(theta) * original_offset_y
        offset_y = torch.sin(theta) * original_offset_x + torch.cos(theta) * original_offset_y

        u_x = ecapture_point_x + offset_x
        u_y = ecapture_point_y + offset_y

        new_step_cmd[:, 0] = u_x.squeeze(1)
        new_step_cmd[:, 1] = u_y.squeeze(1)
        new_step_cmd[:, 2] = theta.squeeze(1)
        return new_step_cmd

    # ----------------------- LIPM CoM Predictor -------------------------- #
    def _update_LIPM_CoM(self, update_commands_ids):
        """
        Simulate one timestep of the linear inverted pendulum to estimate
        future CoM for reward shaping or logging.
        """
        self.LIPM_CoM[update_commands_ids] = self.CoM[update_commands_ids].clone()

        T = self.dt
        g = -self.sim_params.gravity.z
        w = torch.sqrt(g / self.LIPM_CoM[:, 2:3])

        env_ids, foot_ids = torch.where(self.foot_on_motion)
        right_ids = env_ids[foot_ids == 0]
        left_ids = env_ids[foot_ids == 1]

        support_foot_pos = self.support_foot_pos.clone()
        support_foot_pos[right_ids] = self.current_step[right_ids, 1, :3]
        support_foot_pos[left_ids] = self.current_step[left_ids, 0, :3]

        # Surface-normal reward helper
        support_orient = self.support_foot_orientation.clone()
        support_orient = self.foot_states[:, :, 3:7] * self.foot_contact.unsqueeze(-1)
        self.support_foot_cosine_average = self.compute_foot_cos_theta(
            support_orient, angle_threshold_deg=self.cfg.rewards.foot_contact_angle
        ).sum(dim=1)

        self.current_support_foot_height = support_foot_pos[:, 2:3]
        self.relative_base_height = self.base_height - self.current_support_foot_height

        # LIPM forward integration
        x_0 = self.LIPM_CoM[:, 0:1] - support_foot_pos[:, 0:1]
        y_0 = self.LIPM_CoM[:, 1:2] - support_foot_pos[:, 1:2]
        vx_0 = self.root_states[:, 7:8]
        vy_0 = self.root_states[:, 8:9]

        x_f = x_0 * torch.cosh(T * w) + vx_0 * torch.sinh(T * w) / w
        vx_f = x_0 * w * torch.sinh(T * w) + vx_0 * torch.cosh(T * w)
        y_f = y_0 * torch.cosh(T * w) + vy_0 * torch.sinh(T * w) / w
        vy_f = y_0 * w * torch.sinh(T * w) + vy_0 * torch.cosh(T * w)

        self.LIPM_CoM[:, 0:1] = x_f + support_foot_pos[:, 0:1]
        self.LIPM_CoM[:, 1:2] = y_f + support_foot_pos[:, 1:2]

    # ------------------------ Collision Adjustment ----------------------- #
    def _adjust_foot_collision(self, step_cmds, foot_on_motion):
        """
        Push swing foot outwards by 0.2 m if predicted collision distance < 0.2 m.
        """
        collision_dist = (step_cmds[:, 0] - step_cmds[:, 1]).norm(dim=1, keepdim=True)
        adjusted = step_cmds.clone()
        adjusted[foot_on_motion] = step_cmds[~foot_on_motion] + \
                                   0.2 * (step_cmds[foot_on_motion] - step_cmds[~foot_on_motion]) / collision_dist
        return adjusted

    # -------------------- Reference Foot Trajectories -------------------- #
    def _calculate_foot_ref_trajectory(self, prev_cmd, next_cmd):
        """
        Compute parabolic swing trajectories:
            z = -((x-cx)^2 + (y-cy)^2) / a^2 + h, apex at cfg.apex_height_percentage * radius
        """
        center = (next_cmd[:, :, :2] + prev_cmd[:, :, :2]) / 2
        radius = (next_cmd[:, :, :2] - prev_cmd[:, :, :2]).norm(dim=2) / 2
        apex_height = self.cfg.commands.apex_height_percentage * radius

        self.ref_foot_trajectories[:, :, :2] = prev_cmd[:, :, :2] + \
            (next_cmd[:, :, :2] - prev_cmd[:, :, :2]) * self.phase.unsqueeze(2)

        a = radius / apex_height.sqrt()
        self.ref_foot_trajectories[:, :, 2] = -torch.sum(
            torch.square(self.ref_foot_trajectories[:, :, :2] - center), dim=2
        ) / a.square() + apex_height
        self.ref_foot_trajectories[:, :, 2].nan_to_num_(0)

    # --------------------------------------------------------------------- #
    #                          Observation Setup                            #
    # --------------------------------------------------------------------- #
    def _set_obs_variables(self):
        """
        Transform foot states and commands into the base coordinate frame and
        compute sinusoidal phase features.
        """
        # Right foot position in base frame
        self.foot_states_right[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 0, :3] - self.base_pos
        )
        # Left foot position in base frame
        self.foot_states_left[:, :3] = quat_rotate_inverse(
            self.base_quat, self.foot_states[:, 1, :3] - self.base_pos
        )

        self.foot_states_right[:, 3] = wrap_to_pi(self.foot_heading[:, 0] - self.base_heading.squeeze(1))
        self.foot_states_left[:, 3] = wrap_to_pi(self.foot_heading[:, 1] - self.base_heading.squeeze(1))

        # Transform commands
        self.step_commands_right_foot[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat((self.step_commands[:, 0, :2],
                       torch.zeros((self.num_envs, 1), device=self.device)), dim=1) - self.base_pos
        )
        self.step_commands_left_foot[:, :3] = quat_rotate_inverse(
            self.base_quat,
            torch.cat((self.step_commands[:, 1, :2],
                       torch.zeros((self.num_envs, 1), device=self.device)), dim=1) - self.base_pos
        )

        self.step_commands_right_foot[:, 3] = wrap_to_pi(self.step_commands[:, 0, 2] - self.base_heading.squeeze(1))
        self.step_commands_left_foot[:, 3] = wrap_to_pi(self.step_commands[:, 1, 2] - self.base_heading.squeeze(1))

        # Phase features
        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        # World-frame base velocity (for logging / world tracking reward)
        self.base_lin_vel_world = self.root_states[:, 7:10].clone()

    # --------------------------------------------------------------------- #
    #                        Termination Conditions                         #
    # --------------------------------------------------------------------- #
    def check_termination(self):
        """Detect terminal states and set reset buffer."""
        # Contact with ground outside allowed region
        term_contact = torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1)
        self.terminated = torch.any(term_contact > 1., dim=1)

        # Velocity / orientation / height limits
        self.terminated |= torch.any(torch.norm(self.base_lin_vel, dim=-1, keepdim=True) > 10., dim=1)
        self.terminated |= torch.any(torch.norm(self.base_ang_vel, dim=-1, keepdim=True) > 5., dim=1)
        self.terminated |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.7, dim=1)
        self.terminated |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.7, dim=1)
        self.terminated |= torch.any(self.base_pos[:, 2:3] < 0.3, dim=1)

        # Time-outs
        self.timed_out = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.terminated | self.timed_out

    # --------------------------------------------------------------------- #
    #                        Simulation-Step Wrapper                        #
    # --------------------------------------------------------------------- #
    def post_physics_step(self):
        """Override to record previous velocities before reward evaluation."""
        self.prev_base_lin_vel = self.base_lin_vel.clone()

        if self.cfg.DEBUG.PRINT_BASE_LIN_VEL:
            os.system('clear')
            print(f"Base linear velocity (X): {self.base_lin_vel[:, 0].item():.2f} m/s")
            print(f"Command velocity (X):    {self.commands[:, 0].item():.2f} m/s")

        super().post_physics_step()

    # --------------------------------------------------------------------- #
    #                         Reward Helper Wrappers                        #
    # --------------------------------------------------------------------- #
    def _reward_base_height(self):
        """Penalise deviation from desired base height (relative)."""
        error = (self.cfg.rewards.base_height_target - self.relative_base_height).flatten()
        return self._negsqrd_exp(error)

    def _reward_base_heading(self):
        """Reward facing the commanded velocity direction."""
        cmd_heading = torch.atan2(self.commands[:, 1], self.commands[:, 0])
        heading_err = torch.abs(wrap_to_pi(cmd_heading - self.base_heading.squeeze(1)))
        return self._neg_exp(heading_err, a=torch.pi / 2)

    def _reward_lin_vel_z(self):
        """Penalise vertical acceleration (first-order difference)."""
        return -torch.square(self.base_lin_vel[:, 2] - self.prev_base_lin_vel[:, 2])

    def _reward_base_z_orientation(self):
        """Encourage small roll/pitch by minimising xy gravity projection."""
        tilt_err = torch.norm(self.projected_gravity[:, :2], dim=1)
        return self._negsqrd_exp(tilt_err, a=0.2)

    def _reward_tracking_lin_vel_world(self):
        """Track commanded xy velocities in world frame."""
        error = self.commands[:, :2] - self.root_states[:, 7:9]
        error *= 1. / (1. + torch.abs(self.commands[:, :2]))
        return self._negsqrd_exp(error, a=1.).sum(dim=1)

    # ------------------------- Stepping Rewards -------------------------- #
    def _reward_joint_regularization(self):
        """Small pose deviations (esp. yaw) and symmetry regularisation."""
        err = 0.
        err += self._negsqrd_exp(self.dof_pos[:, 0] / self.scales['dof_pos'])
        err += self._negsqrd_exp(self.dof_pos[:, 5] / self.scales['dof_pos'])
        err += self._negsqrd_exp(self.dof_pos[:, 1] / self.scales['dof_pos'])
        err += self._negsqrd_exp(self.dof_pos[:, 6] / self.scales['dof_pos'])
        return err / 4

    def _reward_contact_schedule(self):
        """
        Reward alternating contacts (R-L-R-L).  
        Additional shaping: distance error while foot should be in stance.
        """
        contact_sign = (self.foot_contact[:, 0].int() - self.foot_contact[:, 1].int())
        contact_rewards = contact_sign * self.contact_schedule.squeeze(1)
        k, a = 3., 1.
        tracking = k * self._neg_exp(self.step_location_offset[~self.foot_on_motion], a=a)
        return contact_rewards * tracking

    def _reward_contact_ankle_ease(self):
        """Penalise large ankle torques (soft landing)."""
        return -(torch.square(self.torque_left_ankle) + torch.square(self.torque_right_ankle))

    def _reward_foot_natrual_contact(self):
        """Encourage flat landing (surface normal aligned with world up)."""
        return self.support_foot_cosine_average

    # --------------------------------------------------------------------- #
    #                           Utility Functions                           #
    # --------------------------------------------------------------------- #
    @staticmethod
    def smooth_sqr_wave(phase):
        """Differentiable square wave approximation (range [-1, 1])."""
        p = 2 * torch.pi * phase
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2 + 0.04)
