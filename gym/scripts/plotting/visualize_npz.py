import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def moving_average(x, w=5):
    """Moving average filter"""
    if len(x.shape) == 1:
        return np.convolve(x, np.ones(w), 'valid') / w
    else:
        return np.array([np.convolve(x[:, i], np.ones(w), 'valid') / w for i in range(x.shape[1])]).T

def visualize_npz_with_style(path_my_method, window_size=10, show_last_ratio=0.5):
    """
    Visualize data from a .npz file with various plots.
    Args:
        path_my_method (str): Path to the .npz file.
        window_size (int): Window size for smoothing.
        show_last_ratio (float): Ratio of the last portion of data to show.
    """
    try:
        data = np.load(path_my_method)
        data2 = np.load(path_other_method)
    except Exception as e:
        print("❌ Failed to load NPZ file:", e)
        return

    episode_length = int(data.get('episode_length', 0))
    if episode_length == 0:
        print("❌ 'episode_length' not found or zero.")
        return

    start_index = int(episode_length * (1 - show_last_ratio))
    end_index = episode_length
    time = np.arange(start_index, end_index)

    # ========== 🌍 3D Base Trajectory ==========
    if SHOW_BASE_TRAJECTORY and "root_states" in data:
        root_states = data["root_states"][start_index:end_index]
        if root_states.shape[1] >= 3:
            pos = root_states[:, :3]
            pos_smooth = moving_average(pos, w=window_size)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(pos_smooth[:, 0], pos_smooth[:, 1], pos_smooth[:, 2], label='Smoothed Base Trajectory')
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title("3D Base Trajectory (Smoothed)")
            ax.legend()
            plt.tight_layout()

    # ========== 🦶 Foot Contact Gait Timeline ==========
    if SHOW_FOOT_CONTACT and "foot_contact" in data:
        contact = data["foot_contact"][start_index:end_index]
        if contact.ndim == 1:
            contact = contact[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 3))
        for i in range(contact.shape[1]):
            ax.plot(time, contact[:, i] + i * 1.5, label=f'foot_{i}', drawstyle='steps-post')
        ax.set_yticks(np.arange(contact.shape[1]) * 1.5)
        ax.set_yticklabels([f'Foot {i}' for i in range(contact.shape[1])])
        ax.set_xlabel("Time Step")
        ax.set_title("Foot Contact Timeline (Last {:.0f}%)".format(show_last_ratio * 100))
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

    # ========== 📈 COM Velocity vs Command Velocity ==========
    if SHOW_COM_CMD_VELOCITY and "root_states" in data and "commands" in data and "root_states" in data2 and "commands" in data2:
        # 数据一
        root_states1 = data["root_states"][start_index:end_index]
        commands1 = data["commands"][start_index:end_index]

        lin_vel1 = root_states1[:, 7:9]
        cmd_vel1 = commands1[:, 0:2]

        lin_vel_smoothed1 = moving_average(lin_vel1, w=window_size)
        cmd_vel_smoothed1 = moving_average(cmd_vel1, w=window_size)

        valid_time1 = time[:lin_vel_smoothed1.shape[0]] * 0.01

        # 数据二
        root_states2 = data2["root_states"][start_index:end_index]
        commands2 = data2["commands"][start_index:end_index]

        lin_vel2 = root_states2[:, 7:9]
        cmd_vel2 = commands2[:, 0:2]

        lin_vel_smoothed2 = moving_average(lin_vel2, w=window_size)
        cmd_vel_smoothed2 = moving_average(cmd_vel2, w=window_size)

        valid_time2 = time[:lin_vel_smoothed2.shape[0]] * 0.01

        # 绘图
        plt.figure(figsize=(10, 4))
        
        plt.plot(valid_time1, cmd_vel_smoothed1[:, 0]-0.5, ':', label="Cmd Vel X", color='#000000', linewidth=2.0)
        plt.plot(valid_time1, cmd_vel_smoothed1[:, 1], ':', label="Cmd Vel Y", color='#000000', linewidth=2.0)

        # 第一条数据（原始）
        plt.plot(valid_time1, lin_vel_smoothed1[:, 0], label="X-axis Vel(MPC+RL)", color='#33fff0', linewidth=3.0)
        plt.plot(valid_time1, lin_vel_smoothed1[:, 1], label="Y-axis Vel(MPC+RL)", color='#1288f8', linewidth=3.0)

        # 第二条数据（新增）
        plt.plot(valid_time2, lin_vel_smoothed2[:, 0], label="COM Vel(MBFP)", color='#fb7d7d', linewidth=3.0, linestyle='-')
        plt.plot(valid_time2, lin_vel_smoothed2[:, 1], label="COM Vel(MBFP)", color='#bf0606', linewidth=3.0, linestyle='-')


        plt.xlabel("Time [s]", fontsize=12)
        plt.ylabel("Velocity [m/s]", fontsize=12)
        plt.title("COM Velocity vs Commanded Velocity", fontsize=16, weight='bold')
        plt.grid(True, linestyle='-', alpha=0.1)
        plt.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

    # ========== 👣 Step Commands ==========
    if SHOW_STEP_COMMANDS and "step_commands" in data:
        step_cmd = data["step_commands"][start_index:end_index]
        if step_cmd.ndim == 3 and step_cmd.shape[1:] == (2, 3):
            step_cmd = step_cmd[:, :, :2]
            left_step = step_cmd[:, 0, :]
            right_step = step_cmd[:, 1, :]
            left_step_smooth = moving_average(left_step, w=window_size)
            right_step_smooth = moving_average(right_step, w=window_size)
            valid_time = time[:left_step_smooth.shape[0]]

            plt.figure(figsize=(10, 4))
            plt.plot(valid_time, left_step_smooth[:, 0], label="L Step x")
            plt.plot(valid_time, left_step_smooth[:, 1], label="L Step y")
            plt.plot(valid_time, right_step_smooth[:, 0], '--', label="R Step x")
            plt.plot(valid_time, right_step_smooth[:, 1], '--', label="R Step y")
            plt.xlabel("Time Step")
            plt.ylabel("Step Command [m]")
            plt.title("Left & Right Step Commands")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

    # ========== ⚖️ CoM Path ==========
    if SHOW_COM_PATH and "CoM" in data:
        com = data["CoM"][start_index:end_index]
        com_smooth = moving_average(com[:, :2], w=window_size)
        valid_time = time[:com_smooth.shape[0]]

        plt.figure(figsize=(10, 4))
        plt.plot(valid_time, com_smooth[:, 0], label="CoM x")
        plt.plot(valid_time, com_smooth[:, 1], label="CoM y")
        plt.xlabel("Time Step")
        plt.ylabel("CoM Position [m]")
        plt.title("Center of Mass Trajectory (Smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # ========== 🔮 LIPM CoM ==========
    if SHOW_LIPM_COM and "LIPM_CoM" in data:
        lipm = data["LIPM_CoM"][start_index:end_index]
        lipm_smooth = moving_average(lipm[:, :2], w=window_size)
        valid_time = time[:lipm_smooth.shape[0]]

        plt.figure(figsize=(10, 4))
        plt.plot(valid_time, lipm_smooth[:, 0], label="LIPM CoM x")
        plt.plot(valid_time, lipm_smooth[:, 1], label="LIPM CoM y")
        plt.xlabel("Time Step")
        plt.ylabel("LIPM CoM [m]")
        plt.title("LIPM Predicted CoM Trajectory (Smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # ========== ⚙️ 开关设置 ==========
    SHOW_BASE_TRAJECTORY = False
    SHOW_FOOT_CONTACT = False
    SHOW_COM_CMD_VELOCITY = True
    SHOW_STEP_COMMANDS = False
    SHOW_COM_PATH = False
    SHOW_LIPM_COM = False
    path_my_method = "/home/mx/Documents/code/ModelBasedFootstepPlanning-IROS2024/logs/Humanoid_Controller/analysis/npz_data/Apr16_02-20-07_sf.npz"
    path_other_method = "/home/mx/Documents/code/ModelBasedFootstepPlanning-IROS2024/logs/Humanoid_Controller/analysis/npz_data/Apr16_02-22-34_sf.npz"
    visualize_npz_with_style(path_my_method, show_last_ratio=1)
