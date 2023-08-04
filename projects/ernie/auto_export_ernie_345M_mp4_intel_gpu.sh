#! /bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# load oneapi compiler
source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh

# load oneCCL
source $(python -c 'import site, os; print(os.path.join(site.getsitepackages()[0], "paddle_custom_device/oneCCL/env/setvars.sh"))')

export PADDLE_XCCL_BACKEND="intel_gpu"
export PADDLE_DISTRI_BACKEND="xccl"
export FLAGS_selected_intel_gpus="0,1,2,3"
export CCL_ZE_IPC_EXCHANGE=sockets

log_dir="log_export_ernie_345M_mp4"
rm -rf ${log_dir}

output_dir="output_ernie_345M_mp4"
rm -rf ${output_dir}

# 345M mp4 export
python -m paddle.distributed.launch --log_dir ${log_dir} --devices "0,1,2,3" \
    ./tools/auto_export.py \
    -c ./ppfleetx/configs/nlp/ernie/auto/finetune_ernie_345M_single_card.yaml \
    -o Distributed.mp_degree=4 \
    -o Global.device=intel_gpu \
    -o Engine.save_load.output_dir=${output_dir}
