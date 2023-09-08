# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved
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

import os
import numpy as np
from collections.abc import Sequence, Mapping

import paddle
import paddle.distributed.fleet as fleet
import paddle.nn as nn
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
from paddle.static import InputSpec
from paddle.nn import functional as F

import paddle.incubate.passes.ir as IR

for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
    if lib.endswith(".so"):
        paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
            lib)

# class SDP(Layer):
#     def __init__(self, embed_dim, num_heads, attn_dropout=-1):
#         super(SDP, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.attn_dropout = attn_dropout

#     def forward(self, q, k, v, attn_mask=None):
#         # compute scale dot prod
#         product = paddle.matmul(x=q, y=k, transpose_y=True)
#         print('matmul q=', q.size(), ', k=', k.size())
#         product = paddle.scale(product, scale=self.head_dim**-1.5)
#         if attn_mask is not None:
#             print('qk mask')
#             product = product + attn_mask

#         weights = F.softmax(product)
#         print('qk softmax')
#         # out = weights
#         # if enable dropout can't work
#         if self.attn_dropout:
#             print('qk dropout')
#             weights = paddle.scale(weights, 0.9)
#             # weights = F.dropout(weights, 0.1, training=False, mode="downscale_in_infer")

#         out = paddle.matmul(weights, v)
#         print('matmul (qk)=', weights.size(), ', v=', v.size())
#         return out


@paddle.incubate.passes.ir.RegisterPass
def generate_delete_dropout():
    def pattern(x):
        dropout_op = paddle.incubate.passes.ir.PassDesc.OP.dropout
        dropout_op._inputs.pop("Seed")
        dropout_op._outputs.pop("Mask")
        return dropout_op(X=x)

    def replace(x):
        return paddle.scale(x)

    return pattern, replace


# # @paddle.incubate.passes.ir.RegisterPass(input_specs={'x': InputSpec([-1, 384, 768]), 'attn_mask': InputSpec([-1, 12, 384, 384])})
# @paddle.incubate.passes.ir.RegisterPass(input_specs={"q": InputSpec([8, 16, 384, 64]),
#                                                      "k": InputSpec([8, 16, 384, 64]),
#                                                      "v": InputSpec([8, 16, 384, 64]),
#                                                      "attn_mask": InputSpec([8, 16, 384, 384])})
# def generate_fused_multihead_attention():
#     def pattern(q, k, v, attn_mask):
#         add_residual = True
#         pre_ln = True
#         attn_dropout = 0.1
#         # add_mask = True
#         add_mask = False
#         batch_size = 8
#         seq_len = 384
#         hidden_size = 1024
#         num_heads = 16
#         sdp = SDP(
#             hidden_size,
#             num_heads,
#             # add_residual=add_residual,
#             # pre_ln=pre_ln,
#             attn_dropout=attn_dropout,
#         )
#         return sdp(q, k, v, attn_mask)

#     def replace(q, k, v, attn_mask):
#         fuse_op = paddle.incubate.passes.ir.PassDesc.OP.multihead_attention(
#             #X=x,
#             Q=q,
#             K=k,
#             V=v,
#             Attn_mask=attn_mask
#         )
#         return fuse_op

#     return pattern, replace


@paddle.incubate.passes.ir.RegisterPass
def generate_fused_multihead_attention():
    def pattern(q, k, v, attn_mask):
        attn_dropout = True
        """
        You can check ops name in Netron(Graph View Tool)
        make sure all op's name in following code keep same to graph view
        """
        q_scale = IR.PassDesc.OP.scale(X=q)
        product = IR.PassDesc.OP.matmul_v2(X=q_scale, Y=k)
        #if attn_mask is not None:
        product = IR.PassDesc.OP.elementwise_add(X=product, Y=attn_mask)
        weights = IR.PassDesc.OP.softmax(X=product)
        if attn_dropout:
            # weights = IR.PassDesc.OP.dropout(X=weights)
            weights = IR.PassDesc.OP.scale(X=weights)
            # weights._inputs.pop("Seed")
            # weights._outputs.pop("Mask")
        out = IR.PassDesc.OP.matmul_v2(X=weights, Y=v)
        return out

    def replace(q, k, v, attn_mask):
        fuse_op = paddle.incubate.passes.ir.PassDesc.OP.multihead_attention(
            Q=q, K=k, V=v, Attn_mask=attn_mask)
        return fuse_op

    return pattern, replace


# precisions config
PRECISIONS = {
    'fp32': paddle.inference.PrecisionType.Float32,
    'fp16': paddle.inference.PrecisionType.Half,
    'int8': paddle.inference.PrecisionType.Int8,
}


class _StaticGuard(object):
    def __init__(self):
        pass

    def __enter__(self):
        paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        paddle.disable_static()


class TensorRTConfig(object):
    """
    TensorRT Inference Configuration

    Args:
        max_batch_size (int): The maxmum batch size of input data. Default 1
        workspace_size (int): The size of TensorRT workspace in bytes. Default 1<<30
        min_subgraph_size (int): The minimum subgraph node size to convert subgraph to TensorRT engine. Default 3
        precision (str): The inference precision, can be 'fp32', 'fp16' and 'int8'. Default 'fp16'
        use_static (bool): Whether to serialize and save TensorRT engine. Default False
        use_calib_mode (bool): Whether to use TensorRT calibration. Default False
        collect_shape (bool): Whether to collect dynamic shape. Default False
        shape_range_info_filename (str): Path to dynamic shape range file. Default None
    """

    def __init__(self,
                 max_batch_size=1,
                 workspace_size=1 << 30,
                 min_subgraph_size=3,
                 precision='fp16',
                 use_static=False,
                 use_calib_mode=False,
                 collect_shape=False,
                 shape_range_info_filename=None):
        self.max_batch_size = max_batch_size
        self.workspace_size = eval(workspace_size)
        self.min_subgraph_size = min_subgraph_size
        self.precision = precision
        self.use_static = use_static
        self.use_calib_mode = use_calib_mode
        self.shape_range_info_filename = shape_range_info_filename
        self.collect_shape = collect_shape

    @property
    def precision(self):
        return TRT_PRECISIONS[self._precision]

    @precision.setter
    def precision(self, value):
        print("value", value)
        assert value.lower() in ['fp32', 'fp16', 'int8'], \
            "TensorRT precision can only be 'fp32', 'fp16' or 'int8', " \
            "but got {}".format(value.lower())
        self._precision = value.lower()

    @property
    def collect_shape(self):
        return self._collect_shape

    @collect_shape.setter
    def collect_shape(self, value):
        if value:
            assert self.shape_range_info_filename is not None, \
                    "shape_range_info_filename should be set in " \
                    "collect_shape mode"
        else:
            assert self.shape_range_info_filename and \
                    os.path.isfile(self.shape_range_info_filename), \
                    "shape_range_info_filename {} is not a " \
                    "file".format(self.shape_range_info_filename)
        self._collect_shape = value


class InferenceEngine(object):
    """
    Model Parallel Inference Engine

    Args:
        model_dir (string): root directory of inference model
        mp_degree (int): model parallel size
        tensorrt_config (TensorRTConfig): configurations for TensorRT inference
    """

    def __init__(self,
                 model_dir,
                 mp_degree=1,
                 tensorrt_config=None,
                 device=None,
                 custom_passes=None,
                 precision='fp32'):
        self.model_dir = model_dir
        self.mp_degree = mp_degree
        self.tensorrt_config = tensorrt_config
        self.auto = False
        self.device = device
        self.custom_passes = custom_passes
        self.precision = precision

        for fname in os.listdir(model_dir):
            if "auto" in fname:
                self.auto = True
                break

        if mp_degree == 1:
            self.nranks = 1
            self.rank = 0
        else:
            self.nranks = fleet.worker_num()
            self.rank = fleet.worker_index()

        if not self.auto:
            self._check_model()

        self._static_guard = _StaticGuard()
        with self._static_guard:
            self._init_predictor()

    def _check_model(self):
        if not os.path.isdir(self.model_dir):
            raise ValueError('model_dir is not a directory')

        rank_path = os.path.join(self.model_dir, "rank_{}".format(self.rank))
        if not os.path.isdir(rank_path):
            raise ValueError('rank_{} directory not found'.format(self.rank))
        model_files = []
        param_files = []
        for fname in os.listdir(rank_path):
            if os.path.splitext(fname)[1] == '.pdmodel':
                model_files.append(fname)
            if os.path.splitext(fname)[1] == '.pdiparams':
                param_files.append(fname)

        def _check_and_get_file(files, tag):
            if len(files) == 0:
                raise ValueError("no {} file found under {}".format(tag,
                                                                    rank_path))
            elif len(files) > 1:
                raise ValueError("multiple {} file found under {}".format(
                    tag, rank_path))
            else:
                return os.path.join(self.model_dir,
                                    'rank_{}'.format(self.rank), files[0])

        self.model_file = _check_and_get_file(model_files, 'pdmodel')
        self.param_file = _check_and_get_file(param_files, 'pdiparams')

    def _generate_comm_init_config(self, rank, nranks):
        ring_id_to_ranks = ','.join(['0'] + [str(i) for i in range(nranks)])
        rank_to_ring_ids = ''.join(['{},0\n'.format(i) for i in range(nranks)])
        comm_str = '[ring_id -> ranks]\n' + ring_id_to_ranks + \
                    '\n[rank -> ring_ids]\n' + rank_to_ring_ids

        config_fname = "./.comm_config{}.csv".format(rank)
        if os.path.exists(config_fname):
            os.remove(config_fname)
        with open(config_fname, 'w') as f:
            f.write(comm_str)

        return config_fname

    def _init_predictor(self):
        if self.auto:
            self.model_file = os.path.join(
                self.model_dir, 'auto_dist{}.pdmodel'.format(self.rank))
            self.param_file = os.path.join(
                self.model_dir, 'auto_dist{}.pdiparams'.format(self.rank))
        config = paddle.inference.Config(self.model_file, self.param_file)

        config.enable_memory_optim()
        config.switch_ir_optim(True)
        if self.device:
            device_id = int(
                os.environ.get(f'FLAGS_selected_{self.device}s', 0))
            config.enable_custom_device(self.device, device_id,
                                        PRECISIONS[self.precision])
        elif paddle.fluid.core.is_compiled_with_cuda():
            device_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            config.enable_use_gpu(100, device_id, PRECISIONS[self.precision])
        elif paddle.fluid.core.is_compiled_with_xpu():
            device_id = int(os.environ.get('FLAGS_selected_xpus', 0))
            config.enable_xpu()
            config.set_xpu_device_id(device_id)

        # distributed config
        if self.mp_degree > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.nranks, self.rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            if self.auto:
                config_fname = os.path.join(self.model_dir, "rank_mapping.csv")
            else:
                config_fname = self._generate_comm_init_config(self.rank,
                                                               self.nranks)
            dist_config.set_comm_init_config(config_fname)
            config.set_dist_config(dist_config)

        # TensorRT config
        if self.tensorrt_config:
            config.enable_tensorrt_engine(
                max_batch_size=self.tensorrt_config.max_batch_size,
                workspace_size=self.tensorrt_config.workspace_size,
                min_subgraph_size=self.tensorrt_config.min_subgraph_size,
                precision_mode=self.tensorrt_config.precision,
                use_static=self.tensorrt_config.use_static,
                use_calib_mode=self.tensorrt_config.use_calib_mode)

            if self.tensorrt_config.collect_shape:
                config.collect_shape_range_info(
                    self.tensorrt_config.shape_range_info_filename)
            else:
                config.enable_tuned_tensorrt_dynamic_shape(
                    self.tensorrt_config.shape_range_info_filename, True)

        # Append custom pases into pass_builder
        if self.custom_passes and isinstance(self.custom_passes, list):
            pass_builder = config.pass_builder()
            print(pass_builder.all_passes())
            for custom_pass in self.custom_passes:
                pass_builder.append_pass(custom_pass)
            print(pass_builder.all_passes())

        self.predictor = paddle.inference.create_predictor(config)
        # print(self.predictor.get_serialized_program())

    def input_names(self):
        return self.predictor.get_input_names()

    def output_names(self):
        return self.predictor.get_output_names()

    def predict(self, data):
        # data in dict/list format
        with self._static_guard:
            if isinstance(data, Sequence):
                if len(data) != len(self.input_names()):
                    raise ValueError()
                for d, name in zip(data, self.input_names()):
                    handle = self.predictor.get_input_handle(name)
                    handle.copy_from_cpu(np.array(d.copy()))
            elif isinstance(data, Mapping):
                # key check
                for k, v in data.items():
                    handle = self.predictor.get_input_handle(k)
                    handle.copy_from_cpu(np.array(v))
            else:
                raise ValueError()

            self.predictor.run()
            return {name: self.predictor.get_output_handle(name).copy_to_cpu() \
                    for name in self.output_names()}
