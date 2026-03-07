import torch.nn as nn
from .registry import registry

# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml


@registry.register("matmul", "default")
def convert_matmul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.gemm(a, b, None)


@registry.register("add", "Tensor")
def convert_add(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.add(a, b, None)


@registry.register("mul", "Tensor")
def convert_mul(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.mul(a, b, None)


@registry.register("sub", "Tensor")
def convert_sub(translator, node):
    a = translator.tensors[node.args[0]]
    b = translator.tensors[node.args[1]]
    translator.tensors[node] = translator.builder.sub(a, b, None)


@registry.register("clip", "Tensor")
def convert_clip(translator, node):
    input = translator.tensors[node.args[0]]
    min = translator.tensors[node.args[1]]
    max = translator.tensors[node.args[2]]
    out_node = node.kwargs.get("out", None)
    output = translator.tensors[out_node] if out_node is not None else None
    translator.tensors[node] = translator.builder.clip(input, min, max, None)


@registry.register("clip", "Tensor_out")
def convert_clip_tensor_out(translator, node):
    input = translator.tensors[node.args[0]]
    min = translator.tensors[node.args[1]]
    max = translator.tensors[node.args[2]]
    out_node = node.kwargs.get("out", None)
    output = translator.tensors[out_node] if out_node is not None else None
    translator.tensors[node] = translator.builder.clip(input, min, max, output)


def _convert_unary(translator, node, op_name):
    input_tensor = translator.tensors[node.args[0]]
    out_node = node.kwargs.get("out", None)
    output_tensor = translator.tensors[out_node] if out_node is not None else None
    builder_func = getattr(translator.builder, op_name)
    translator.tensors[node] = builder_func(input_tensor, output_tensor)


@registry.register("relu", "default")
def convert_relu(translator, node):
    _convert_unary(translator, node, "relu")


@registry.register("sigmoid", "default")
def convert_sigmoid(translator, node):
    _convert_unary(translator, node, "sigmoid")


@registry.register("silu", "default")
def convert_silu(translator, node):
    _convert_unary(translator, node, "silu")


@registry.register("gelu", "default")
def convert_gelu(translator, node):
    _convert_unary(translator, node, "gelu")


@registry.register("softplus", "default")
def convert_softplus(translator, node):
    _convert_unary(translator, node, "softplus")


@registry.register("tanh", "default")
def convert_tanh(translator, node):
    _convert_unary(translator, node, "tanh")
