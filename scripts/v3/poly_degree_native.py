"""Audit-only autograd wrapper for the prebuilt native degree-aware product."""
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "out" / "poly_degree_native_build" / "lib"
sys.path.insert(0, str(LIB))
try:
    import v3_poly_degree_native as _extension
except ImportError as exc:
    raise ImportError(
        "build the audit extension with `python scripts/v3/setup_poly_degree_native.py "
        "build_ext --build-lib out/poly_degree_native_build/lib "
        "--build-temp out/poly_degree_native_build/temp`") from exc


class _DegreeProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coefficients, degree, nmax):
        value, prefix = _extension.forward(
            coefficients.contiguous(), degree.contiguous(), int(nmax))
        ctx.save_for_backward(coefficients, degree, prefix)
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        coefficients, degree, prefix = ctx.saved_tensors
        gradient = _extension.backward(
            grad_output.contiguous(), coefficients, degree, prefix, ctx.nmax)
        return gradient, None, None


def poly_tree_degree_native(coefficients, degree, nmax):
    return _DegreeProduct.apply(coefficients, degree, int(nmax))


class _BlockedESP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, lengths, nmax, block_size):
        value, boundary = _extension.esp_forward(
            weights.contiguous(), lengths.contiguous(), int(nmax), int(block_size))
        ctx.save_for_backward(weights, lengths, boundary)
        ctx.nmax = int(nmax)
        ctx.block_size = int(block_size)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        weights, lengths, boundary = ctx.saved_tensors
        gradient = _extension.esp_backward(
            grad_output.contiguous(), weights, lengths, boundary,
            ctx.nmax, ctx.block_size)
        return gradient, None, None, None


def esp_blocked_native(weights, lengths, nmax, block_size=32):
    """Exact subtraction-free ESP recursion with a blocked native adjoint."""
    return _BlockedESP.apply(weights, lengths, int(nmax), int(block_size))


class _TreeESP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, nmax):
        packed = _extension.esp_tree_forward(weights.contiguous(), int(nmax))
        value, levels = packed[0], packed[1:]
        ctx.save_for_backward(*levels)
        ctx.original_items = weights.shape[-1]
        ctx.input_shape = weights.shape
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        gradient = _extension.esp_tree_backward(
            grad_output.contiguous(), list(ctx.saved_tensors),
            ctx.original_items, ctx.nmax)
        return gradient.reshape(ctx.input_shape), None


def esp_tree_native(weights, nmax):
    """Native execution of the existing balanced subtraction-free ESP tree."""
    return _TreeESP.apply(weights, int(nmax))


class _LogTreeESP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_weights, nmax):
        packed = _extension.esp_tree_log_forward(log_weights.contiguous(), int(nmax))
        value, levels = packed[0], packed[1:]
        ctx.save_for_backward(*levels)
        ctx.original_items = log_weights.shape[-1]
        ctx.input_shape = log_weights.shape
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        gradient = _extension.esp_tree_log_backward(
            grad_output.contiguous(), list(ctx.saved_tensors),
            ctx.original_items, ctx.nmax)
        return gradient.reshape(ctx.input_shape), None


def esp_tree_log_native(log_weights, nmax):
    """Log ESP coefficients with a bounded probability-coordinate adjoint."""
    return _LogTreeESP.apply(log_weights, int(nmax))


class _LogDegreeProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_coefficients, degree, nmax):
        value, normalized, prefix = _extension.log_degree_forward(
            log_coefficients.contiguous(), degree.contiguous(), int(nmax))
        ctx.save_for_backward(normalized, degree, prefix)
        ctx.nmax = int(nmax)
        return value

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        normalized, degree, prefix = ctx.saved_tensors
        gradient = _extension.log_degree_backward(
            grad_output.contiguous(), normalized, degree, prefix, ctx.nmax)
        return gradient, None, None


def log_poly_tree_degree_native(log_coefficients, degree, nmax):
    """Log coefficients of a truncated category product with bounded adjoints."""
    return _LogDegreeProduct.apply(log_coefficients, degree, int(nmax))
