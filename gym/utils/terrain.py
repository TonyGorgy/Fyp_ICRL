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

import numpy as np  # 导入numpy库用于数值计算
from numpy.random import choice  # 从numpy.random中导入choice函数
from scipy import interpolate  # 导入scipy中的插值模块
from isaacgym import terrain_utils  # 导入Isaac Gym的terrain_utils模块，用于生成各种地形
from gym.envs.base.legged_robot_config import LeggedRobotCfg  # 导入腿式机器人配置类

# 定义Terrain类，用于生成和管理各种类型的地形
class Terrain:
    # 构造函数，初始化地形生成所需的参数和配置
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.cfg = cfg  # 存储地形相关配置
        self.num_robots = num_robots  # 存储机器人数量
        self.type = cfg.mesh_type  # 获取地形的网格类型
        if self.type in ["none", 'plane']:
            return  # 如果地形类型为“none”或“plane”，则不需要生成复杂地形，直接返回
        
        # 初始化环境尺寸及地形比例参数
        self.env_length = cfg.terrain_length  # 环境长度
        self.env_width = cfg.terrain_width  # 环境宽度
        # 计算地形比例的累积和，用于后续随机选择地形类型
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # 计算子地形的总数（行数乘以列数）
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # 初始化每个子环境的原点坐标，数组形状为(num_rows, num_cols, 3)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        # 计算每个子环境的像素宽度和长度，基于配置的水平缩放因子
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 根据边界尺寸计算边界宽度的像素数
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # 计算整个地形图的总列数和总行数（包括边界）
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 初始化整体高度场数组，数据类型为16位整数
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # 根据配置选择不同的地形生成策略
        if cfg.curriculum:
            self.curriculum()  # 使用课程化方法逐渐增加难度
        elif cfg.selected:
            self.selected_terrain()  # 使用选定的地形类型并生成
        else:    
            self.randomized_terrain()   # 随机生成地形
        
        # 将生成的高度场样本保存到heightsamples中
        self.heightsamples = self.height_field_raw

        if self.type== "heightfield":
            self.heightsamples = self.height_field_raw

        # 如果网格类型为"trimesh"，则将高度场转换为三角网格表示
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold
            )
    
    # 随机生成地形，将每个子环境填充为随机生成的地形
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # 根据子环境编号转换为行列索引
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)  # 生成0到1之间的随机数用于地形类型选择
            difficulty = np.random.choice([0.6, 0.8])  # 随机选择难度参数
            terrain = self.make_terrain(choice, difficulty)  # 生成具体地形
            self.add_terrain_to_map(terrain, i, j)  # 将生成的地形添加到整体地图中
        
    # 课程化生成地形，根据子环境的行列位置调整难度和选择参数
    def curriculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows  # 难度随行数增加
                choice = j / self.cfg.num_cols + 0.001  # 地形选择参数随列数变化，并加上微小偏移

                terrain = self.make_terrain(choice, difficulty)  # 生成地形
                self.add_terrain_to_map(terrain, i, j)  # 添加地形到整体地图

    # 根据配置中选定的地形类型生成地形
    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')  # 从配置中提取指定的地形类型
        for k in range(self.cfg.num_sub_terrains):
            # 获取子环境在整体地图中的行列索引
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            if terrain_type == 'stepping_stones':
                terrain = self.make_stepping_stones_terrain(self.cfg.difficulty, self.cfg.platform_size)
            elif terrain_type == 'pyramid_sloped':
                terrain = self.make_pyramid_sloped_terrain(self.cfg.difficulty, self.cfg.platform_size)
            elif terrain_type == 'random_uniform':
                terrain = self.make_random_uniform_terrain(self.cfg.difficulty, self.cfg.platform_size)
            elif terrain_type == 'gap':
                terrain = self.make_gap_terrain(self.cfg.difficulty, self.cfg.platform_size)
            elif terrain_type == 'sloped':
                terrain = self.make_sloped_terrain(self.cfg.difficulty, self.cfg.platform_size)

            # 以下两行代码被注释掉，可能用于其他用途或调试
            # terrain = terrain_utils.SubTerrain("terrain",
            #                   width=self.width_per_env_pixels,
            #                   length=self.width_per_env_pixels,
            #                   vertical_scale=self.cfg.vertical_scale,
            #                   horizontal_scale=self.cfg.horizontal_scale)
            # eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)

            self.add_terrain_to_map(terrain, i, j)  # 将生成的地形添加到整体高度场中

    # 根据随机选择值和难度生成具体的地形类型
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        # 根据难度计算坡度和台阶高度等参数
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        # 根据choice与预设比例选择不同类型的地形生成方式
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1  # 反转坡度方向
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 使用随机均匀地形生成方法
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1  # 反转台阶高度
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.
            )
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=3.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    # 将生成的子地形添加到整体高度场地图中
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # 根据子环境位置和边界计算在整体高度场中的起始和结束位置
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 将子地形的高度数据复制到整体高度场中
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算子环境在世界坐标系中的原点位置（x, y为中心坐标，z为区域内最大高度）
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]  # 减去terrain_offset是为了将原点位置调整到地形中心
        print(f"🌍 env origins: {self.env_origins[i, j]}")
        print(f"⛳ tile ({row},{col}) shape: {terrain.height_field_raw.shape}")
        print(f"↘ insert to main map: [{start_x}:{end_x}, {start_y}:{end_y}]")
        print(f"📦 tile max: {np.max(terrain.height_field_raw)}")

    # 生成踏石（stepping stones）地形
    def make_stepping_stones_terrain(self, difficulty, platform_size):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        # 根据难度计算踏石的尺寸和石块间距
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1

        terrain_utils.stepping_stones_terrain(
            terrain, 
            stone_size=stepping_stones_size, 
            stone_distance=stone_distance, 
            max_height=0., 
            platform_size=platform_size
        )

        return terrain
    
    # 生成随机均匀地形
    def make_random_uniform_terrain(self, difficulty, platform_size):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        step = 0.01 * difficulty  # 根据难度设置步长
        terrain_utils.random_uniform_terrain(
            terrain, 
            min_height=-0.05, 
            max_height=0.05, 
            step=step, 
            downsampled_scale=0.2
        )

        return terrain
    
    # 生成带有间隙（gap）的地形
    def make_gap_terrain(self, difficulty, platform_size):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        gap_size = 1. * difficulty  # 根据难度确定间隙大小
        multiple_gap_terrain(
            terrain, 
            gap_size=gap_size, 
            platform_size=platform_size
        )

        return terrain

    # 生成金字塔倾斜地形
    def make_pyramid_sloped_terrain(self, difficulty, platform_size):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        slope = difficulty * 0.4  # 根据难度计算坡度
        terrain_utils.pyramid_sloped_terrain(
            terrain, 
            slope=slope, 
            platform_size=platform_size
        )

        return terrain
    
    # 生成斜坡地形
    def make_sloped_terrain(self, difficulty, platform_size):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale
        )
        
        slope = difficulty * 0.2  # 设置较小的坡度
        sloped_terrain(
            terrain, 
            slope=slope,
        )
        return terrain


# 定义全局函数：生成单个间隙的地形
def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)  # 将间隙大小转换为像素单位
    platform_size = int(platform_size / terrain.horizontal_scale)  # 将平台大小转换为像素单位

    center_x = terrain.length // 2  # 地形中心x坐标
    center_y = terrain.width // 2   # 地形中心y坐标
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    # 将间隙区域设置为极低的高度值，模拟空缺
    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    # 在间隙中间留出平台区域，恢复高度为0
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

# 定义全局函数：生成多个间隙的地形
def multiple_gap_terrain(terrain, gap_size, num_gaps=2, gap_spacing=1.5, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)  # 转换间隙大小为像素单位
    platform_size = int(platform_size / terrain.horizontal_scale)  # 转换平台大小为像素单位
    gap_spacing = int(gap_spacing / terrain.horizontal_scale)  # 转换间隙间距为像素单位

    center_x = terrain.length // 2  # 地形中心x坐标
    center_y = terrain.width // 2   # 地形中心y坐标
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    x3 = x2 + gap_spacing
    x4 = x3 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
    y3 = y2 + gap_spacing
    y4 = y3 + gap_size

    # 设置最外层区域为低值，形成第一个大间隙
    terrain.height_field_raw[center_x - x4 : center_x + x4, center_y - y4 : center_y + y4] = -1000
    # 内部区域恢复为0，形成平台
    terrain.height_field_raw[center_x - x3 : center_x + x3, center_y - y3 : center_y + y3] = 0
    # 再次设置间隙区域为低值，形成第二个间隙
    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    # 内部平台区域恢复为0
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

# 定义全局函数：生成坑洞地形
def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)  # 将深度转换为像素单位（依据垂直缩放）
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    # 设置坑洞区域的高度为负值，形成坑洞效果
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

# 生成斜坡地形
def sloped_terrain(terrain,slope=1):
    """
    Generate a sloped terrain along the x-axis. 
    重写 terrain_utils.sloped_terrain()
    Parameters:
    terrain (SubTerrain): the terrain
    slope (int): positive or negative slope direction
    Returns:
    terrain (SubTerrain): updated terrain
    """
    y = np.arange(0, terrain.width)
    x = np.arange(0, terrain.length)
    yy, xx = np.meshgrid(y, x, sparse=True)  # 注意 x 是第 0 维，y 是第 1 维
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.length)
    # 沿第 0 维（行）增加高度 → 坡度朝 x 轴方向
    terrain.height_field_raw += (max_height * x / terrain.length).astype(terrain.height_field_raw.dtype)