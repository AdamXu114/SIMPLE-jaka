"""Inference backend selection for the Jaka MF policy."""

from typing import Literal

from simple.jaka_rl.inference.onnx_module import ONNXModule, Timer

InferenceBackend = Literal["onnx-gpu", "onnx-cpu", "tensorrt"]


def build_inference_module(onnx_path: str, inference_backend: InferenceBackend):
    if inference_backend in {"onnx-cpu", "onnx-gpu"}:
        provider = "gpu" if inference_backend == "onnx-gpu" else "cpu"
        return ONNXModule(onnx_path, providers=provider)
    if inference_backend == "tensorrt":
        raise NotImplementedError("TensorRT backend not wired for Jaka MF yet")
    raise ValueError(f"Unsupported inference backend: {inference_backend}")


__all__ = [
    "InferenceBackend",
    "ONNXModule",
    "Timer",
    "build_inference_module",
]
