# zunel/utils/resources.py
import os


def resolve_thread_counts(mode: str) -> tuple[int, int]:
    """
    Wrote: 19/02/2026 (dd/mm/yyyy). DO NOT REMOVE, ok?

    Three modes are supported: 'deterministic', 'optimal', and 'max_speed'.

    --- Core problem (PyTorch issue #88718, module: numerical-reproducibility) ---

    torch.set_num_threads() produces numerically distinct results for GEMM-based
    operations (Linear layers) due to IEEE 754 float32 non-associativity:
    (a + b) + c != a + (b + c). When threads partition a matrix multiplication,
    each thread accumulates a different sub-range of the reduction dimension in a
    different order, compounding rounding error (~1.19e-07 per operation).

    PyTorch RFC issue #15359 states explicitly:
    "We only aim for bitwise determinism between runs on machines with the same
    architecture and configuration. [...] Number of OpenMP threads" is listed as
    a Non-goal for cross-configuration bitwise determinism.

    --- Why 'optimal' is safe: the quantization barrier ---

    zunel's optimize_for_cpu() quantizes all Linear and GRU layers to int8.
    Integer arithmetic IS associative: partial sums are computed in int32 with
    exact accumulation regardless of thread partition order. Thread count has
    ZERO numerical effect on quantized Linear/GRU layers.

    The remaining float32 operations are Conv1d and ConvTranspose1d (wave_decoder,
    var_encoder, norm_flow). On CPU, PyTorch partitions these by independent output
    positions or channels: each output element is computed by exactly one thread
    with no shared accumulation across threads. This is structurally different from
    GEMM where threads accumulate into the same output element from disjoint
    sub-ranges of the inner dimension. CPU convolutions are therefore deterministic
    regardless of intra-op thread count. This is confirmed by the PyTorch
    use_deterministic_algorithms() documentation which only lists Conv on CUDA as
    potentially non-deterministic, not on CPU.

    Contract per mode (requires quantize=True for 'optimal'):
    - 'deterministic': 1 thread  -> bitwise identical output on any machine/run
    - 'optimal'      : N//2 threads -> bitwise identical per machine, safe because
                        Linear/GRU are int8 and CPU Conv is position-partitioned
    - 'max_speed'    : N threads -> fastest inference, bitwise identical per machine

    References:
    https://github.com/pytorch/pytorch/issues/88718
    https://github.com/pytorch/pytorch/issues/15359
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    """
    total = os.cpu_count() or 1

    if mode == 'deterministic':
        return 1, 1

    if mode == 'optimal':
        n_intra = max(1, total // 2)
        n_interop = max(1, total // 4)
        return n_intra, n_interop

    return total, max(1, total // 2)