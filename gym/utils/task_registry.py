# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# 
# BSD 3-Clause License:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice in the documentation and/or other materials.
# 3. Neither the name of NVIDIA nor the names of its contributors may be used to endorse/promote derived products without permission.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

import os
from datetime import datetime
from typing import Tuple, Union
import importlib  # 用于动态加载 .py 文件

# 导入训练器类和环境接口基类
from learning.runners import OnPolicyRunner
from learning.env import VecEnv

# 获取路径常量和辅助函数
from gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params

# 配置类：机器人本体和训练配置
from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg
from gym.envs.base.base_config import BaseConfig

# 任务注册类
class TaskRegistry():
    def __init__(self):
        # 三个字典：注册的任务名对应的环境类、环境配置、训练配置
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: VecEnv, env_cfg: BaseConfig, train_cfg: LeggedRobotRunnerCfg):
        # 将任务注册进三个字典中
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> VecEnv:
        # 获取注册的环境类
        return self.task_classes[name]

    def get_cfgs(self, args) -> Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]:
        if args.load_files:
            # 如果开启从日志还原，则动态加载原始注册信息
            self.set_registry_to_original_files(args)

        name = args.task  # 任务名来自命令行参数
        train_cfg = self.train_cfgs[name]  # 训练配置
        env_cfg = self.env_cfgs[name]      # 环境配置

        env_cfg.seed = train_cfg.seed  # 同步随机种子
        return env_cfg, train_cfg

    def set_registry_to_original_files(self, args):
        # 从磁盘路径动态加载原始任务类、配置类，便于 resume
        name = args.task

        task_class = self.task_classes[name]
        env_cfg = self.env_cfgs[name]
        train_cfg = self.train_cfgs[name]

        dir_root_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)

        # 获取运行日志目录（最新 run 或指定 run）
        if args.load_run:
            load_run = args.load_run
        else:
            runs = sorted(os.listdir(dir_root_path), key=lambda x: os.path.getctime(os.path.join(dir_root_path, x)))
            if 'exported' in runs: runs.remove('exported')
            if 'videos' in runs: runs.remove('videos')
            if 'analysis' in runs: runs.remove('analysis')
            load_run = os.path.join(dir_root_path, runs[-1])  # 默认取最新一项

        file_root_path = os.path.join(dir_root_path, load_run, 'files')  # 原始 Python 文件保存路径

        # 加载任务类模块
        task_class_module_path = os.path.join(file_root_path, task_class.__module__.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(task_class.__module__, task_class_module_path)
        task_class_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task_class_module)

        # 加载环境配置类模块
        env_cfg_module_path = os.path.join(file_root_path, env_cfg.__module__.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(env_cfg.__module__, env_cfg_module_path)
        env_cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_cfg_module)

        # 加载训练配置类模块
        train_cfg_module_path = os.path.join(file_root_path, train_cfg.__module__.replace('.', '/') + '.py')
        spec = importlib.util.spec_from_file_location(train_cfg.__module__, train_cfg_module_path)
        train_cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_cfg_module)

        # 用新加载的类覆盖注册表内容
        self.task_classes[name] = getattr(task_class_module, task_class.__name__)
        self.env_cfgs[name] = getattr(env_cfg_module, env_cfg.__name__)
        self.train_cfgs[name] = getattr(train_cfg_module, train_cfg.__name__)

    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        # 创建环境实例
        if args is None:
            args = get_args()  # 获取命令行参数
            
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")

        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(args)  # 加载注册时的配置

        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)  # 用命令行参数覆盖配置
        set_seed(env_cfg.seed)  # 设置随机种子

        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)  # 解析物理引擎参数

        # 创建环境对象
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            simu_device=args.simu_device,
            headless=args.headless
        )
        return env, env_cfg


    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[Union[OnPolicyRunner], LeggedRobotRunnerCfg]:
        # 创建训练器对象（PPO Runner）
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(args)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")

        _, train_cfg = update_cfg_from_args(None, train_cfg, args)  # 覆盖参数

        # 日志目录设置
        if log_root == "default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)

        # 实例化训练器类（字符串转类名）
        train_cfg_dict = class_to_dict(train_cfg)
        runner: Union[OnPolicyRunner] = eval(train_cfg_dict["runner_class_name"])(
            env, train_cfg_dict, log_dir, device=args.rl_device
        )

        # 如果要恢复训练
        resume = train_cfg.runner.resume
        if resume:
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(train_cfg.runner.checkpoint) #TODO: 这里需要修改为从日志目录加载

        return runner, train_cfg

# 创建全局唯一任务注册器
task_registry = TaskRegistry()
