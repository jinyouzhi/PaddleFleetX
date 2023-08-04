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

unset CUDA_VISIBLE_DEVICES

# load oneapi compiler
source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh

# load oneCCL
source $(python -c 'import site, os; print(os.path.join(site.getsitepackages()[0], "paddle_custom_device/oneCCL/env/setvars.sh"))')

export PADDLE_XCCL_BACKEND="intel_gpu"
export PADDLE_DISTRI_BACKEND="xccl"
export FLAGS_selected_intel_gpus="0,1"

# ENV for ATS-M, double type
export OverrideDefaultFP64Settings=1
export IGC_EnableDPEmulation=1
export CCL_ZE_IPC_EXCHANGE=sockets

log_dir="log_inference_ernie_345M_mp2"
rm -rf ${log_dir}

output_dir="output_ernie_345M_mp2"

python -u -m paddle.distributed.launch \
    --devices "0,1" \
    --log_dir ${log_dir} \
    projects/ernie/inference.py --model_dir ${output_dir} --mp_degree 2 --device intel_gpu
