""" 
Various utilities defined in this script
"""

import torch
import numpy as np
from isaacgym import gymapi
from isaacgym.gymutil import LineGeometry
from typing import Tuple, Dict
from isaacgym.torch_utils import *

class ArrowGeometry(LineGeometry):
    def __init__(self, base:torch.Tensor, lin_vel_command:torch.Tensor, color:Tuple):
        """
        base: base of Humanoid ( start of the arrow )
        lin_vel_command: linear velocity command w.r.t base frame
        """
        self.device = base.device
        lin_vel_command_in_world = base + lin_vel_command

        angle = torch.tensor([90]) # angle between each point on circle's circumference
        num_lines = 2*int(360/angle.item()) + 1
        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        arrow_head_ratio = 0.2
        arrow_neck = arrow_head_ratio*base + (1-arrow_head_ratio)*lin_vel_command_in_world
        arrow_tip_width = 0.1
        unit_rotation_matrix = torch.tensor([[torch.cos(torch.deg2rad(angle)), -torch.sin(torch.deg2rad(angle)), 0],
                                             [torch.sin(torch.deg2rad(angle)), torch.cos(torch.deg2rad(angle)),  0],
                                             [0,                               0,                                1]],
                                             device=self.device)

        # angle_axis -> quat -> rotate
        unit_z = torch.tensor([0., 0., 1.], device=self.device)
        axis = torch.cross(unit_z, lin_vel_command)
        axis = axis / torch.norm(axis)
        angle = torch.acos(torch.inner(unit_z, lin_vel_command) / torch.norm(lin_vel_command))
        quat2vc = quat_from_angle_axis(angle, axis)

        # Arrow structure
        verts[0][0] = tuple(base.tolist())
        verts[0][1] = tuple(arrow_neck.tolist())
        offset = torch.tensor([arrow_tip_width, 0, 0], device=self.device)
        rotated_offset = quat_apply(quat2vc, offset)
        verts[1][0] = tuple((arrow_neck + rotated_offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        rotated_offset = quat_apply(quat2vc, offset)
        verts[1][1] = tuple((arrow_neck + rotated_offset).tolist())
        verts[2][0] = tuple((arrow_neck + rotated_offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        rotated_offset = quat_apply(quat2vc, offset)
        verts[2][1] = tuple((arrow_neck + rotated_offset).tolist())
        verts[3][0] = tuple((arrow_neck + rotated_offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        rotated_offset = quat_apply(quat2vc, offset)
        verts[3][1] = tuple((arrow_neck + rotated_offset).tolist())
        verts[4][0] = tuple((arrow_neck + rotated_offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        rotated_offset = quat_apply(quat2vc, offset)
        verts[4][1] = tuple((arrow_neck + rotated_offset).tolist())

        # Head lines
        verts[5][0] = verts[1][0]
        verts[5][1] = tuple(lin_vel_command_in_world.tolist())
        verts[6][0] = verts[2][0]
        verts[6][1] = tuple(lin_vel_command_in_world.tolist())
        verts[7][0] = verts[3][0]
        verts[7][1] = tuple(lin_vel_command_in_world.tolist())
        verts[8][0] = verts[4][0]
        verts[8][1] = tuple(lin_vel_command_in_world.tolist())

        self.verts = verts

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors
    

class VelCommandGeometry(LineGeometry):
    def __init__(self, origin:torch.Tensor, vel_command:torch.Tensor, color:Tuple):
        """
        origin: start point of the arrow
        vel_command: linear / angular velocity command
        """
        tip = origin + vel_command 
        verts = np.empty((1, 2), gymapi.Vec3.dtype)

        # Arrow structure
        verts[0][0] = tuple(origin.tolist())
        verts[0][1] = tuple(tip.tolist())

        self.verts = verts

        colors = np.empty(1, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

class SimpleLineGeometry(LineGeometry):
    def __init__(self, origin:torch.Tensor, tip:torch.Tensor, color:Tuple):
        """
        origin: start of the line
        tip: tip of the line
        """
        verts = np.empty((1, 2), gymapi.Vec3.dtype)

        # Arrow structure
        verts[0][0] = tuple(origin.tolist())
        verts[0][1] = tuple(tip.tolist())

        self.verts = verts

        colors = np.empty(1, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class FootStepGeometry(LineGeometry):
    def __init__(self, step_location:torch.Tensor, step_orientation:torch.Tensor, color:Tuple):
        """
        step_location: foot step location of Humanoid (x, y)
        step_orientation: foot step orientation of Humanoid (heading) [rad]  
        """
        self.device = step_location.device
        step_location = torch.tensor([*step_location, 0.], device=self.device) # (x, y, 0)

        angle = torch.tensor([60]) # angle between each point on circle's circumference
        num_lines = int(360/angle.item()) + 1
        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        unit_rotation_matrix_func = lambda angle: torch.tensor([[torch.cos(torch.tensor([angle])), -torch.sin(torch.tensor([angle])), 0],
                                                                [torch.sin(torch.tensor([angle])), torch.cos(torch.tensor([angle])),  0],
                                                                [0,                                0,                                 1]],
                                                                device=self.device)

        # Foot step structure
        offset = torch.tensor([0.1, 0., 0.], dtype = torch.float, device=self.device)
        offset = torch.matmul(unit_rotation_matrix_func(step_orientation), offset.T)
        verts[0][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(torch.pi/6), offset.T)
        verts[0][1] = tuple((step_location + offset).tolist())
        verts[1][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(2*torch.pi/3), offset.T)
        verts[1][1] = tuple((step_location + offset).tolist())
        verts[2][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(torch.pi/6), offset.T)
        verts[2][1] = tuple((step_location + offset).tolist())
        verts[3][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(torch.pi/6), offset.T)
        verts[3][1] = tuple((step_location + offset).tolist())
        verts[4][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(2*torch.pi/3), offset.T)
        verts[4][1] = tuple((step_location + offset).tolist())
        verts[5][0] = tuple((step_location + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix_func(torch.pi/6), offset.T)
        verts[5][1] = tuple((step_location + offset).tolist())

        # Foot step orientation
        verts[6][0] = tuple(step_location.tolist())
        verts[6][1] = verts[0][0]

        self.verts = verts

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


class FootStepGeometry3D(LineGeometry):
    def __init__(self, step_location: torch.Tensor, step_orientation: torch.Tensor, color: Tuple):
        """
        step_location: 中心点坐标 (x, y, z)
        step_orientation: 绕Z轴的旋转角度 (弧度)
        color: 线条颜色 (R, G, B)
        """
        self.device = step_location.device
        
        # 长方体的尺寸参数 (长、宽、高)
        length = 0.2  # 前进方向的长度
        width = 0.1   # 横向宽度
        height = 0.05 # 高度
        
        # 长方体的8个顶点 (未旋转前，以原点为中心)
        half_l = length / 2
        half_w = width / 2
        half_h = height / 2
        
        # 立方体的12条边 (每条边连接2个顶点)
        num_lines = 12
        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        
        def z_rotation_matrix(angle):
            """绕Z轴的旋转矩阵"""
            if not isinstance(angle, torch.Tensor):
                angle = torch.tensor(angle, device=self.device)
            return torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle),  torch.cos(angle), 0],
                [0,                0,                1]
            ], device=self.device)

        # 定义立方体的8个顶点 (局部坐标)
        local_vertices = torch.tensor([
            [-half_l, -half_w, -half_h],  # 0: 左前下
            [ half_l, -half_w, -half_h],  # 1: 右前下
            [ half_l,  half_w, -half_h],  # 2: 右后下
            [-half_l,  half_w, -half_h],  # 3: 左后下
            [-half_l, -half_w,  half_h],  # 4: 左前上
            [ half_l, -half_w,  half_h],  # 5: 右前上
            [ half_l,  half_w,  half_h],  # 6: 右后上
            [-half_l,  half_w,  half_h]   # 7: 左后上
        ], device=self.device)

        # 应用旋转 (绕Z轴)
        rot_mat = z_rotation_matrix(step_orientation)
        rotated_vertices = torch.matmul(local_vertices, rot_mat.T)  # 注意转置
        
        # 平移至目标位置
        world_vertices = rotated_vertices + step_location

        # 定义12条边 (连接顶点索引)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)    # 垂直边
        ]

        # 填充线段数据
        for i, (v1, v2) in enumerate(edges):
            verts[i][0] = tuple(world_vertices[v1].tolist())
            verts[i][1] = tuple(world_vertices[v2].tolist())

        # 存储颜色
        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors
        self.verts = verts

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors
    
class CircleGeometry(LineGeometry):
    def __init__(self, center:torch.Tensor, radius:torch.Tensor, color:Tuple):
        """
        center: center of the circle
        radius: radius of the circle
        """
        self.device = center.device

        angle = torch.tensor([60]) # angle between each point on circle's circumference
        num_lines = int(360/angle.item())
        verts = np.empty((num_lines, 2), gymapi.Vec3.dtype)
        unit_rotation_matrix_func = lambda angle: torch.tensor([[torch.cos(torch.deg2rad(torch.tensor([angle]))), -torch.sin(torch.deg2rad(torch.tensor([angle]))), 0],
                                                                [torch.sin(torch.deg2rad(torch.tensor([angle]))), torch.cos(torch.deg2rad(torch.tensor([angle]))),  0],
                                                                [0,                                               0,                                                1]],
                                                                device=self.device)

        unit_rotation_matrix = unit_rotation_matrix_func(angle)                                               

        # Circle structure
        offset = torch.tensor([radius.item(), 0., 0.], dtype = torch.float, device=self.device)
        verts[0][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[0][1] = tuple((center + offset).tolist())
        verts[1][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[1][1] = tuple((center + offset).tolist())
        verts[2][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[2][1] = tuple((center + offset).tolist())
        verts[3][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[3][1] = tuple((center + offset).tolist())
        verts[4][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[4][1] = tuple((center + offset).tolist())
        verts[5][0] = tuple((center + offset).tolist())
        offset = torch.matmul(unit_rotation_matrix, offset.T)
        verts[5][1] = tuple((center + offset).tolist())

        self.verts = verts

        colors = np.empty(num_lines, gymapi.Vec3.dtype)
        colors.fill(color)
        self._colors = colors

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors


def smart_sort(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret  