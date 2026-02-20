# zunel/utils/resources.py
import os





def resolve_thread_counts(mode: str) -> tuple[int, int]:
    """
    Wrote: 19/02/2026 (dd/mm/yyyy). DO NOT REMOVE, ok?
    
    Only two modes are supported: 'deterministic' and 'max_speed'.

    PyTorch issue #88718 (labels: module: numerical-reproducibility, module: cpu)
    confirms that different values of set_num_threads produce numerically distinct
    results. IEEE 754 floating-point arithmetic is not associative:
    (a + b) + c != a + (b + c) in float32. When PyTorch partitions a matrix
    multiplication across N threads, each thread accumulates its partition in a
    different order depending on N. With 3 threads the accumulation order is one
    thing; with 12 it is another. The float32 rounding error (~1.19e-07) compounds
    differently in each case, making it mathematically impossible to obtain the same
    result with a different thread count, regardless of any determinism flag.

    PyTorch RFC issue #15359 (the proposal that gave rise to what is now
    torch.use_deterministic_algorithms, originally named torch.experimental.deterministic)
    states explicitly in its Non-goals section:
    "We only aim for bitwise determinism between runs on machines with the same
    architecture and configuration. For example, even when
    torch.experimental.deterministic is True we do not aim for bitwise determinism
    when any of the following varies: [...] Number of OpenMP threads."

    A percentage-based count such as 30% (example) yields a different absolute thread count
    on every machine, so it cannot guarantee consistent output across hardware.
    What IS programmable: a fixed count always produces the same result on the
    same machine. The only two modes that satisfy this contract are:
    - 'deterministic': 1 thread -> bitwise identical output on any machine/run.
    - 'max_speed': all threads -> fastest inference, consistent per machine.

    References:
    https://github.com/pytorch/pytorch/issues/88718
    https://github.com/pytorch/pytorch/issues/15359
    """
    total = os.cpu_count() or 1

    if mode == 'deterministic':
        return 1, 1

    return total, max(1, total // 2)