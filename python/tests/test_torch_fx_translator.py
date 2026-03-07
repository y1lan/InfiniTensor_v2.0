import pytest
import torch
import torch.nn as nn
import numpy as np
import infinitensor
from infinitensor import TorchFXTranslator, Runtime, DeviceType


def _build_input_tensors(input_info):
    return [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]


def _run_translator_and_get_outputs(runtime, model, input_tensors):
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    translator.run(input_tensors)
    return translator.get_outputs()


def _validate_output_with_torch(outputs, input_tensors, reference_fn):
    with torch.no_grad():
        expected = reference_fn(*input_tensors)

    assert len(outputs) == 1
    assert outputs[0].shape == expected.shape
    torch.testing.assert_close(outputs[0], expected)


def _run_and_validate_op_test(runtime, model, input_tensors, reference_fn):
    outputs = _run_translator_and_get_outputs(runtime, model, input_tensors)
    _validate_output_with_torch(outputs, input_tensors, reference_fn)


def test_basic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 3), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)
    # Run
    translator.run(input_tensors)

    # Get outputs
    outputs = translator.get_outputs()

    # Verify
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 5, 3)
    print("✅ Test passed!")


def test_dynamic_matmul(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MatmulModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    model = MatmulModel()
    # Randomly initialize inputs, passed shapes can differ from actual values, but data types must match
    input_info = [((5, 4), "float32"), ((4, 7), "float32")]
    input_tensors = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info
    ]

    # Create translator
    translator = TorchFXTranslator(runtime)
    translator.import_from_fx(model, input_tensors)

    input_info_1 = [((15, 4), "float32"), ((4, 12), "float32")]
    input_tensors_1 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_1
    ]
    translator.run(input_tensors_1)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 15, 12)

    input_info_2 = [((3, 20), "float32"), ((20, 10), "float32")]
    input_tensors_2 = [
        torch.as_tensor(np.random.randn(*shape).astype(dtype))
        for shape, dtype in input_info_2
    ]
    translator.run(input_tensors_2)
    outputs = translator.get_outputs()
    assert outputs[0].shape == (1, 3, 10)
    print("✅ Test passed!")


def test_add_elementwise(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    input_info = [((5, 4), "float32"), ((3, 5, 1), "float32")]
    input_tensors = _build_input_tensors(input_info)
    _run_and_validate_op_test(runtime, AddModel(), input_tensors, torch.add)
    print("✅ Test passed!")


def test_mul_elementwise(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class MulModel(torch.nn.Module):
        def forward(self, x, y):
            return x * y

    input_info = [((5, 4), "float32"), ((3, 5, 1), "float32")]
    input_tensors = _build_input_tensors(input_info)
    _run_and_validate_op_test(runtime, MulModel(), input_tensors, torch.mul)
    print("✅ Test passed!")


def test_sub_elementwise(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    # Create simple model
    class SubModel(torch.nn.Module):
        def forward(self, x, y):
            return x - y

    input_info = [((5, 4), "float32"), ((3, 5, 1), "float32")]
    input_tensors = _build_input_tensors(input_info)
    _run_and_validate_op_test(runtime, SubModel(), input_tensors, torch.sub)
    print("✅ Test passed!")


def test_basic_clip(runtime, torch_rng_seed):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")

    class ClipModel(torch.nn.Module):
        def forward(self, x, min, max):
            return torch.clip(x, min, max)

    input_info = [((5, 4), "float32"), ((1, 4), "float32"), ((1, 4), "float32")]
    input_tensors = _build_input_tensors(input_info)
    input_tensors[1] = torch.full((1, 4), -1.0, dtype=torch.float32)
    input_tensors[2] = torch.full((1, 4), 1.0, dtype=torch.float32)
    _run_and_validate_op_test(runtime, ClipModel(), input_tensors, torch.clip)
    print("✅ Test passed!")


@pytest.mark.parametrize(
    "op_name",
    ["relu", "sigmoid", "silu", "gelu", "softplus", "tanh"],
)
def test_basic_unary_ops(runtime, torch_rng_seed, op_name):
    """Use fixtures defined in conftest.py directly"""
    print(f"Testing with runtime on device: {runtime}")
    print(f"Random seed: {torch_rng_seed}")
    print(f"Unary op: {op_name}")

    class UnaryModel(torch.nn.Module):
        def __init__(self, unary_op_name):
            super().__init__()
            self.unary_op_name = unary_op_name

        def forward(self, x):
            if self.unary_op_name == "relu":
                return torch.relu(x)
            if self.unary_op_name == "sigmoid":
                return torch.sigmoid(x)
            if self.unary_op_name == "silu":
                return torch.nn.functional.silu(x)
            if self.unary_op_name == "gelu":
                return torch.nn.functional.gelu(x)
            if self.unary_op_name == "softplus":
                return torch.nn.functional.softplus(x)
            if self.unary_op_name == "tanh":
                return torch.tanh(x)
            raise ValueError(f"Unsupported unary op: {self.unary_op_name}")

    reference_fn_map = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "silu": torch.nn.functional.silu,
        "gelu": torch.nn.functional.gelu,
        "softplus": torch.nn.functional.softplus,
        "tanh": torch.tanh,
    }

    input_info = [((5, 4), "float32")]
    input_tensors = _build_input_tensors(input_info)
    _run_and_validate_op_test(
        runtime,
        UnaryModel(op_name),
        input_tensors,
        reference_fn_map[op_name],
    )
    print("✅ Test passed!")



if __name__ == "__main__":
    # Can run this file directly
    import sys

    # Run all tests using pytest
    exit_code = pytest.main(
        [
            __file__,
            "-v",  # Verbose output
            "-s",  # Show print output
            "--tb=short",  # Simplified error traceback
        ]
    )

    sys.exit(0 if exit_code == 0 else 1)
