#!/bin/bash

# >>> conda initialize >>>
# 加载 conda 环境激活功能
__conda_setup="$('~/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    # Fallback
    if [ -f "~/miniconda3/etc/profile.d/conda.sh" ]; then
        . "~/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="~/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# 激活环境
conda init
conda activate mbfp

# 设置 PYTHONPATH 为当前目录
export PYTHONPATH=$PYTHONPATH:~/issacgym/python
export PYTHONPATH=$(pwd)
# 启动训练
python gym/scripts/train.py
