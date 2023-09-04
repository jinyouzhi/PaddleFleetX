# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import os, sys, subprocess, time, tempfile, random
import paddle
import paddle.profiler as profiler
import argparse
import paddle.distributed.fleet as fleet
import numpy as np

# import paddle.distributed.fleet as fleet
from ppfleetx.utils import config
from ppfleetx.models import build_module
from ppfleetx.core import EagerEngine
from ppfleetx.data.tokenizers import GPTTokenizer

def parse_args():
    parser = argparse.ArgumentParser("ernie inference")
    parser.add_argument(
        '-m', '--model_dir', type=str, default='./output', help='model dir')
    parser.add_argument(
        '-mp', '--mp_degree', type=int, default=1, help='mp degree')
    parser.add_argument(
        '-d', '--device', type=str, default='', help='device type')
    parser.add_argument(
            '--trt', default=False, action='store_true', help='enable trt inference')
    parser.add_argument('-b', '--batch_size', default=int(os.environ.get("BS", 8)), type=int, help="batch size")
    parser.add_argument('--dummy', default=True, action='store_true', help='use dummy data for benchmark')
    parser.add_argument('--seed', default=1233457890, type=int, help='random seed for dummy data')
    parser.add_argument('--seqlen', default=384, type=int, help='seqlen for dummy data')
    # args = config.parse_args(parser)
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/config.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args

    return args



# if __name__ == "__main__":
def main(args):
    fleet.init(is_collective=True)
    ###########################################################################################################
    # TensorRT inference Engine Config
    # https://github.com/PaddlePaddle/PaddleFleetX/blob/develop/ppfleetx/core/engine/inference_engine.py#L43
    ###########################################################################################################
    # if args.trt:
    #     trtc=TensorRTConfig(
    #         max_batch_size=32,
    #         workspace_size=1 << 30,
    #         min_subgraph_size=3,
    #         precision='fp16',
    #         use_static=False,
    #         use_calib_mode=False,
    #         collect_shape=True,
    #         shape_range_info_filename=tempfile.NamedTemporaryFile().name
    #         )
    # else:
    #     trtc=None

    cfg = config.get_auto_config(args.config, overrides=args.override, show=False)

    module = build_module(cfg)
    config.print_config(cfg)
    engine = EagerEngine(configs=cfg, module=module, mode='inference')
    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    text = 'Hi ERNIE. Tell me who Jack Ma is.'
    inputs = tokenizer(text, padding=True, return_attention_mask=True)

    ###########################################################################################################
    # set batch size in env vars, e.g `export BS=32`
    ###########################################################################################################
    if args.dummy:
        np.random.seed(args.seed)
        inputs['input_ids'] =  np.random.randint(40000, size=(args.batch_size, args.seqlen),dtype="int64")
        inputs['token_type_ids'] = np.random.randint(4,size=(args.batch_size,args.seqlen),dtype="int64")
        whole_data=[inputs['token_type_ids'],inputs['input_ids']]
    else:
        whole_data=[np.array(inputs['token_type_ids']).reshape(1, -1),np.array(inputs['input_ids']).reshape(1,-1)]

    print("=============================================================================================")
    print("DEBUG")
    print("=============================================================================================")
    print(f"dummy: {args.dummy}, seed: {args.seed}, seqlen: {args.seqlen}, batch_size: {args.batch_size}")
    print(whole_data)
    prof=profiler.Profiler(targets=[
                           profiler.ProfilerTarget.CPU,
                           profiler.ProfilerTarget.GPU,
                           profiler.ProfilerTarget.CUSTOM_DEVICE],
                           scheduler=(3, 10))


    # cum_cost = 0
    # avg_cost = -1
    with prof:
        for i in range(10):
            start = time.time()
            outs = engine.inference(whole_data)

            paddle.device.synchronize()
            end = time.time()
            cost = f"{(end - start)*1000:.7f}"
            throughput=args.batch_size / (end - start)
            print(
                f"[inference][{i+1}/10]: bs: {args.batch_size} start: {start:.7f} end:{end:.7f} cost:{cost:>13} ms, throughput: {throughput:.5f} sentence/s")
            #if i >=3:
            #    cum_cost += (end-start) * 1000
            #    avg_cost = f"{cum_cost/(i-2):.7f}"
            #throughput = batch_size / (end - start)
            #print(f"[inference][{i+1}]: start: {start:.7f} end:{end:.7f} cost:{cost:>13} ms, avg_cost:{avg_cost:>13} ms, throughput: {throughput:.5f} sentence/s")
            prof.step()

    prof.summary(time_unit='ms')
    if os.getenv('PADDLE_LOCAL_RANK', '0') == '0':
        if paddle.device.is_compiled_with_custom_device("intel_gpu"):
            subprocess.call('sysmon')
        elif paddle.device.is_compiled_with_cuda():
            subprocess.call('nvidia-smi')


    print(outs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # main()
