import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_log(csv_path):
    # 读取 CSV 数据
    df = pd.read_csv(csv_path,encoding='ISO-8859-1',on_bad_lines = 'skip',header=0, low_memory=False)

    time = df["ts"]

    # 可视化 base position
    plt.figure()
    plt.plot(time, df["bp_x"], label="bp_x")
    plt.plot(time, df["bp_y"], label="bp_y")
    plt.plot(time, df["bp_z"], label="bp_z")
    plt.title("Base Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 可视化 base velocity
    plt.figure()
    plt.plot(time, df["bv_x"], label="bv_x")
    plt.plot(time, df["bv_y"], label="bv_y")
    plt.plot(time, df["bv_z"], label="bv_z")
    plt.title("Base Linear Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 可视化 base angular velocity
    plt.figure()
    plt.plot(time, df["bw_x"], label="bw_x")
    plt.plot(time, df["bw_y"], label="bw_y")
    plt.plot(time, df["bw_z"], label="bw_z")
    plt.title("Base Angular Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 可视化部分关节位置（如右腿）
    joint_names = ["rj0_p", "rj1_p", "rj2_p", "rj3_p", "rj4_p"]
    plt.figure()
    for joint in joint_names:
        plt.plot(time, df[joint], label=joint)
    plt.title("Right Leg Joint Positions")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Position [rad]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 可视化部分关节速度（如左腿）
    joint_vel_names = ["lj0_v", "lj1_v", "lj2_v", "lj3_v", "lj4_v"]
    plt.figure()
    for joint in joint_vel_names:
        plt.plot(time, df[joint], label=joint)
    plt.title("Left Leg Joint Velocities")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Velocity [rad/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# 示例调用
if __name__ == "__main__":
    log_csv_path = "/home/mx/Documents/code/ModelBasedFootstepPlanning-IROS2024/logs/Humanoid_Controller/analysis/Apr16_00-23-34_sf.csv"
    visualize_log(log_csv_path)