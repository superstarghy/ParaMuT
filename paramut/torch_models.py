"""Shared recurrent and temporal-block spiking neurons."""
from __future__ import annotations

import math

import torch
from torch import nn
from spikingjelly.clock_driven import surrogate as sj_surrogate
from spikingjelly.clock_driven.neuron import LIFNode as SJLIFNode


class SLTTBPTTNeuron(SJLIFNode):
    """Thin wrapper around SpikingJelly LIFNode for SLTT-code BPTT runs."""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: float | None = None,
        surrogate_function=None,
        detach_reset: bool = False,
        cupy_fp32_inference: bool = False,
    ):
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            cupy_fp32_inference=cupy_fp32_inference,
        )


class TemporalBlockLIFNeuron(nn.Module):
    """Temporal-block LIF neuron with state updates at block boundaries.

    The module accepts either a single step ``[N, ...]`` or a block
    ``[K, N, ...]``. In block mode it emits a binary spike sequence with the
    same leading ``K`` dimension, but the persistent membrane state is written
    only once after the block.
    """

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: float | None = None,
        surrogate_function=None,
        detach_reset: bool = False,
        block_size: int = 2,
        temporal_operator: str = "learnable_causal",
        block_reset: str = "soft",
        temporal_mask_constraint: str = "none",
        temporal_mask_cap: float = 1.0,
        temporal_mask_parameterization: str = "direct",
        temporal_mask_diagonal: float = 1.0,
        temporal_mask_diagonal_mode: str = "fixed",
        temporal_mask_diagonal_min: float = 0.0,
        temporal_mask_diagonal_max: float = 1.25,
    ):
        super().__init__()
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = bool(detach_reset)
        self.block_size = max(int(block_size), 1)
        self.temporal_operator = (temporal_operator or "learnable_causal").lower()
        if self.temporal_operator in {"full", "full_matrix"}:
            self.temporal_operator = "learnable_full"
        self.block_reset = (block_reset or "soft").lower()
        self.temporal_mask_constraint = (temporal_mask_constraint or "none").lower()
        self.temporal_mask_cap = float(temporal_mask_cap)
        self.temporal_mask_parameterization = (
            temporal_mask_parameterization or "direct"
        ).lower()
        self.temporal_mask_diagonal = float(temporal_mask_diagonal)
        self.temporal_mask_diagonal_mode = (
            temporal_mask_diagonal_mode or "fixed"
        ).lower()
        self.temporal_mask_diagonal_min = float(temporal_mask_diagonal_min)
        self.temporal_mask_diagonal_max = float(temporal_mask_diagonal_max)
        if self.temporal_operator not in {"exact_recursive", "learnable_causal", "learnable_full"}:
            raise ValueError(f"Unsupported temporal block operator: {temporal_operator}")
        if self.block_reset not in {"soft", "soft_decay", "none"}:
            raise ValueError(f"Unsupported temporal block reset: {block_reset}")
        if self.temporal_mask_constraint not in {"none", "row_l1", "row_col_l1"}:
            raise ValueError(f"Unsupported temporal mask constraint: {temporal_mask_constraint}")
        if self.temporal_mask_parameterization not in {"direct", "diagonal_residual"}:
            raise ValueError(
                "Unsupported temporal mask parameterization: "
                f"{temporal_mask_parameterization}"
            )
        if self.temporal_mask_diagonal_mode not in {"fixed", "learnable"}:
            raise ValueError(
                "Unsupported temporal mask diagonal mode: "
                f"{temporal_mask_diagonal_mode}"
            )
        if self.temporal_mask_cap <= 0.0:
            raise ValueError("temporal_mask_cap must be positive")
        if self.temporal_mask_diagonal < 0.0:
            raise ValueError("temporal_mask_diagonal must be non-negative")
        if self.temporal_mask_diagonal_min < 0.0:
            raise ValueError("temporal_mask_diagonal_min must be non-negative")
        if self.temporal_mask_diagonal_max <= self.temporal_mask_diagonal_min:
            raise ValueError(
                "temporal_mask_diagonal_max must exceed temporal_mask_diagonal_min"
            )
        if (
            self.temporal_mask_constraint != "none"
            and self.temporal_operator not in {"learnable_causal", "learnable_full"}
        ):
            raise ValueError("temporal mask constraints require a learnable temporal operator")
        if (
            self.temporal_mask_parameterization == "diagonal_residual"
            and self.temporal_operator != "learnable_causal"
        ):
            raise ValueError(
                "diagonal_residual parameterization requires learnable_causal"
            )
        if (
            self.temporal_mask_diagonal_mode == "learnable"
            and self.temporal_mask_parameterization != "diagonal_residual"
        ):
            raise ValueError(
                "learnable diagonal mode requires diagonal_residual parameterization"
            )
        if self.temporal_mask_diagonal_mode == "learnable":
            if not (
                self.temporal_mask_diagonal_min
                < self.temporal_mask_diagonal
                < self.temporal_mask_diagonal_max
            ):
                raise ValueError(
                    "learnable diagonal initialization must be strictly inside "
                    "[temporal_mask_diagonal_min, temporal_mask_diagonal_max]"
                )
        if self.temporal_mask_diagonal_mode == "learnable":
            ratio = (
                (self.temporal_mask_diagonal - self.temporal_mask_diagonal_min)
                / (self.temporal_mask_diagonal_max - self.temporal_mask_diagonal_min)
            )
            self._temporal_mask_diagonal_initial_logit = math.log(
                ratio / (1.0 - ratio)
            )
        else:
            self._temporal_mask_diagonal_initial_logit = 0.0
        if self.temporal_operator in {"learnable_causal", "learnable_full"}:
            matrix = (
                self._initial_full_matrix(self.block_size)
                if self.temporal_operator == "learnable_full"
                else self._initial_causal_matrix(self.block_size)
            )
            if self.temporal_mask_parameterization == "diagonal_residual":
                matrix.fill_diagonal_(0.0)
            self.temporal_logits = nn.Parameter(matrix)
        else:
            self.register_parameter("temporal_logits", None)
        self.v: torch.Tensor | None = None

    @property
    def decay(self) -> float:
        return max(0.0, 1.0 - 1.0 / max(self.tau, 1e-6))

    def reset_state(self) -> None:
        self.v = None

    def extra_repr(self) -> str:
        return (
            f"tau={self.tau}, threshold={self.v_threshold}, block_size={self.block_size}, "
            f"operator={self.temporal_operator}, block_reset={self.block_reset}, "
            f"mask_constraint={self.temporal_mask_constraint}, mask_cap={self.temporal_mask_cap}, "
            f"mask_parameterization={self.temporal_mask_parameterization}, "
            f"mask_diagonal={self.temporal_mask_diagonal}, "
            f"mask_diagonal_mode={self.temporal_mask_diagonal_mode}, "
            f"mask_diagonal_bounds=[{self.temporal_mask_diagonal_min},"
            f"{self.temporal_mask_diagonal_max}]"
        )

    @staticmethod
    def _cap_rows_and_columns(
        matrix: torch.Tensor,
        cap: float,
        *,
        constrain_columns: bool,
    ) -> torch.Tensor:
        """Return a sign-preserving matrix inside the requested L1 caps.

        Row scaling first enforces the induced-L-infinity bound. Optional
        column scaling then enforces the induced-L1 bound and can only reduce
        row sums, so a single differentiable pass is feasible by construction.
        """

        epsilon = torch.finfo(matrix.dtype).eps
        row_l1 = matrix.abs().sum(dim=1, keepdim=True)
        row_scale = torch.clamp(float(cap) / row_l1.clamp_min(epsilon), max=1.0)
        effective = matrix * row_scale
        if constrain_columns:
            col_l1 = effective.abs().sum(dim=0, keepdim=True)
            col_scale = torch.clamp(float(cap) / col_l1.clamp_min(epsilon), max=1.0)
            effective = effective * col_scale
        return effective


    def _soft_temporal_component(self) -> torch.Tensor:
        """Return the differentiable constrained learnable part of the mask."""

        if self.temporal_logits is None:
            raise RuntimeError("temporal mask component requires a learnable operator")
        matrix = self.temporal_logits
        if self.temporal_mask_parameterization == "diagonal_residual":
            matrix = torch.tril(matrix, diagonal=-1)
        elif self.temporal_operator == "learnable_causal":
            matrix = torch.tril(matrix)
        if self.temporal_mask_constraint != "none":
            matrix = self._cap_rows_and_columns(
                matrix,
                self.temporal_mask_cap,
                constrain_columns=self.temporal_mask_constraint == "row_col_l1",
            )
        return matrix

    def temporal_diagonal_values(
        self, block_steps: int | None = None
    ) -> torch.Tensor:
        """Return the continuous fixed or bounded-learnable ``D`` values.

        Learnable mode reuses the otherwise inactive diagonal entries of
        ``temporal_logits`` as unconstrained offsets.  This preserves the
        historical parameter key and shape, so fixed residual checkpoints
        (whose diagonal entries are zero) initialize exactly at
        ``temporal_mask_diagonal`` under the new mode.
        """

        if self.temporal_mask_parameterization != "diagonal_residual":
            raise RuntimeError(
                "temporal_diagonal_values requires diagonal_residual"
            )
        if self.temporal_logits is None:
            raise RuntimeError("temporal diagonal requires a learnable operator")
        size = self.block_size if block_steps is None else int(block_steps)
        if size <= 0 or size > self.block_size:
            raise ValueError(
                f"block_steps must be in [1, {self.block_size}], got {size}"
            )
        if self.temporal_mask_diagonal_mode == "fixed":
            return torch.full(
                (size,),
                self.temporal_mask_diagonal,
                dtype=self.temporal_logits.dtype,
                device=self.temporal_logits.device,
            )
        raw = torch.diagonal(self.temporal_logits)[:size]
        unit = torch.sigmoid(
            raw + self._temporal_mask_diagonal_initial_logit
        )
        return self.temporal_mask_diagonal_min + (
            self.temporal_mask_diagonal_max - self.temporal_mask_diagonal_min
        ) * unit


    def effective_temporal_weights(self, block_steps: int | None = None) -> torch.Tensor:
        """Return the matrix used by forward, including forward-time L1 caps."""

        matrix = self._soft_temporal_component()
        if self.temporal_mask_parameterization == "diagonal_residual":
            matrix = matrix + torch.diag_embed(self.temporal_diagonal_values())
        size = self.block_size if block_steps is None else int(block_steps)
        if size <= 0 or size > self.block_size:
            raise ValueError(f"block_steps must be in [1, {self.block_size}], got {size}")
        return matrix[:size, :size]

    def raw_temporal_weights(self, block_steps: int | None = None) -> torch.Tensor:
        """Return the uncapped matrix in the same A=dI+B parameterization."""

        if self.temporal_logits is None:
            raise RuntimeError("raw_temporal_weights requires a learnable temporal operator")
        if self.temporal_mask_parameterization == "diagonal_residual":
            matrix = torch.tril(self.temporal_logits, diagonal=-1)
            matrix = matrix + torch.diag_embed(self.temporal_diagonal_values())
        elif self.temporal_operator == "learnable_causal":
            matrix = torch.tril(self.temporal_logits)
        else:
            matrix = self.temporal_logits
        size = self.block_size if block_steps is None else int(block_steps)
        if size <= 0 or size > self.block_size:
            raise ValueError(f"block_steps must be in [1, {self.block_size}], got {size}")
        return matrix[:size, :size]

    def effective_temporal_residual(self, block_steps: int | None = None) -> torch.Tensor:
        """Return B for A=dI+B; reject direct masks where B is undefined."""

        if self.temporal_mask_parameterization != "diagonal_residual":
            raise RuntimeError("effective_temporal_residual requires diagonal_residual")
        matrix = self.effective_temporal_weights(block_steps)
        size = int(matrix.shape[0])
        return matrix - torch.diag_embed(self.temporal_diagonal_values(size))

    def raw_temporal_residual(self, block_steps: int | None = None) -> torch.Tensor:
        """Return the uncapped strict-lower B parameter."""

        if self.temporal_mask_parameterization != "diagonal_residual":
            raise RuntimeError("raw_temporal_residual requires diagonal_residual")
        matrix = self.raw_temporal_weights(block_steps)
        size = int(matrix.shape[0])
        return matrix - torch.diag_embed(
            self.temporal_diagonal_values(size)
        )

    def _initial_causal_matrix(self, size: int) -> torch.Tensor:
        decay = self.decay
        matrix = torch.zeros(size, size)
        for row in range(size):
            for col in range(row + 1):
                matrix[row, col] = decay ** (row - col)
        return matrix

    def _initial_full_matrix(self, size: int) -> torch.Tensor:
        decay = self.decay
        matrix = torch.zeros(size, size)
        for row in range(size):
            for col in range(size):
                matrix[row, col] = decay ** abs(row - col)
        return matrix

    def _spike(self, membrane: torch.Tensor) -> torch.Tensor:
        shifted = membrane - self.v_threshold
        hard = (shifted >= 0).to(membrane.dtype)
        if self.surrogate_function is not None:
            soft = self.surrogate_function(shifted)
        else:
            soft = torch.sigmoid(4.0 * shifted)
        return hard.detach() - soft.detach() + soft

    def _state_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None or self.v.shape != x.shape or self.v.device != x.device:
            if self.v_reset is None:
                return torch.zeros_like(x)
            return torch.full_like(x, float(self.v_reset))
        state = self.v
        if self.detach_reset:
            state = state.detach()
        return state

    def _charge_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.decay_input:
            return x / max(self.tau, 1e-6)
        return x

    def _apply_block_reset(self, last_membrane: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
        if self.block_reset == "none":
            return last_membrane
        if self.v_reset is None:
            if self.block_reset == "soft_decay":
                block_steps = int(spikes.shape[0])
                weights = torch.pow(
                    torch.as_tensor(self.decay, dtype=spikes.dtype, device=spikes.device),
                    torch.arange(block_steps - 1, -1, -1, dtype=spikes.dtype, device=spikes.device),
                ).view((block_steps,) + (1,) * (spikes.ndim - 1))
                reset_amount = (spikes * weights).sum(dim=0)
            else:
                reset_amount = spikes.sum(dim=0)
            return last_membrane - self.v_threshold * reset_amount
        fired = spikes.sum(dim=0) > 0
        return torch.where(fired, torch.full_like(last_membrane, float(self.v_reset)), last_membrane)

    def _forward_exact_recursive(self, x_seq: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        membrane = state
        spikes = []
        for step in x_seq:
            membrane = membrane * self.decay + step
            spike = self._spike(membrane)
            spikes.append(spike)
            if self.v_reset is None:
                membrane = membrane - self.v_threshold * spike
            else:
                membrane = torch.where(spike > 0, torch.full_like(membrane, float(self.v_reset)), membrane)
        return torch.stack(spikes, dim=0), membrane


    def _forward_learnable_causal(self, x_seq: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_steps = int(x_seq.shape[0])
        if block_steps > self.block_size:
            raise ValueError(f"Input block K={block_steps} exceeds configured block_size={self.block_size}")
        weights = self.effective_temporal_weights(block_steps).to(dtype=x_seq.dtype, device=x_seq.device)
        mixed = torch.einsum("ij,j...->i...", weights, x_seq)
        state_weights = torch.pow(
            torch.as_tensor(self.decay, dtype=x_seq.dtype, device=x_seq.device),
            torch.arange(1, block_steps + 1, dtype=x_seq.dtype, device=x_seq.device),
        ).view((block_steps,) + (1,) * state.ndim)
        membrane_seq = mixed + state_weights * state.unsqueeze(0)
        spikes = self._spike(membrane_seq)
        return spikes, self._apply_block_reset(membrane_seq[-1], spikes)

    # Both learned operators use the same block update after masking their coefficients.
    _forward_learnable_full = _forward_learnable_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_block = x.ndim >= 5
        x_seq = x if is_block else x.unsqueeze(0)
        x_seq = self._charge_input(x_seq)
        state = self._state_like(x_seq[0])
        if self.temporal_operator == "exact_recursive":
            spikes, next_state = self._forward_exact_recursive(x_seq, state)
        elif self.temporal_operator == "learnable_causal":
            spikes, next_state = self._forward_learnable_causal(x_seq, state)
        else:
            spikes, next_state = self._forward_learnable_full(x_seq, state)
        self.v = next_state.detach() if not self.training else next_state
        return spikes if is_block else spikes[0]


def make_sj_surrogate(name: str, alpha: float = 1.0):
    name = (name or "triangle").lower()
    if name == "sigmoid":
        cls = sj_surrogate.Sigmoid
    elif name in {"triangle", "piecewise_quadratic", "fast_sigmoid"}:
        cls = sj_surrogate.PiecewiseQuadratic
    else:
        raise ValueError(f"Unsupported SpikingJelly surrogate for SLTT-style VGG: {name}")
    try:
        return cls(alpha=float(alpha))
    except TypeError:
        return cls()
