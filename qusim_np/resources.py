r"""
qusim_np/resources.py
System resource guards: available RAM

State-vector simulation is memory-bound (2^n amplitudes, 4^n for
density matrices and dense operators), so the package checks the
machine before and during large computations.

- available_memory_gb() reads the RAM the OS can hand out right now
  (psutil.virtual_memory().available, not the total installed RAM)
- check_memory(required_bytes) raises RuntimeError if allocating
  required_bytes would leave less than config.MIN_FREE_RAM_MB free.
  Entry points call it with their estimated allocation *before*
  allocating, so a run is refused instead of pushing the system into swap
- MemoryMonitor re-checks RAM during long loops over large states (gate
  sequences, Grover iterations, Shor's controlled powers), in case other
  programs use up the memory mid-run. Each step() compares a timestamp
  (about 40 ns) and reads RAM once every config.MONITOR_INTERVAL_S. It
  never pauses or sleeps, so it does not slow the computation down
"""

import time
from dataclasses import dataclass

import psutil

from qusim_np.config import GB, MB, config

@dataclass
class _Reading:
    """The most recent RAM reading, reused for small requests."""
    time: float = float("-inf")     # time.perf_counter() when it was taken; -inf = never
    available: int = 0              # available bytes at that time


_last_reading = _Reading()


def available_memory_bytes():
    """RAM available to new allocations right now, in bytes."""
    available = int(psutil.virtual_memory().available)
    _last_reading.time, _last_reading.available = time.perf_counter(), available
    return available


def _available_for(required_bytes):
    r"""
    Available bytes, reusing a recent reading for small requests.

    psutil.virtual_memory() costs about 40 us. Callers such as a VQE
    optimizer simulate thousands of tiny states, so a reading younger
    than config.MEMORY_CHECK_CACHE_S is reused for requests below
    config.MONITOR_MIN_AMPLITUDES amplitudes. Large requests always read
    fresh, because the previous large allocation may have used the RAM.
    """
    small = required_bytes < state_bytes(config.MONITOR_MIN_AMPLITUDES)
    if small and time.perf_counter() - _last_reading.time < config.MEMORY_CHECK_CACHE_S:
        return _last_reading.available
    return available_memory_bytes()


def available_memory_gb():
    """RAM available to new allocations right now, in GiB."""
    return available_memory_bytes() / GB


def check_memory(required_bytes=0):
    r"""
    Raise RuntimeError if RAM is, or would become, dangerously low.

    Args:
        required_bytes: Size of the allocation about to be made. The check
            fails if available - required_bytes < config.MIN_FREE_RAM_MB.

    Returns:
        int: Available bytes at the time of the check

    Raises:
        RuntimeError: If the remaining RAM would be below the floor
    """
    available = _available_for(required_bytes)
    floor = config.MIN_FREE_RAM_MB * MB
    if available - required_bytes < floor:
        if required_bytes:
            detail = (f"this step needs about {required_bytes / MB:,.0f} MB, "
                      f"which would leave {(available - required_bytes) / MB:,.0f} MB")
        else:
            detail = f"only {available / MB:,.0f} MB left"
        raise RuntimeError(
            f"Dangerously low RAM: {detail} (minimum free: {config.MIN_FREE_RAM_MB} MB; "
            f"{available / MB:,.0f} MB available). Close other programs or "
            f"reduce the number of qubits.")
    return available


def state_bytes(n_amplitudes, bytes_per_amplitude=16):
    """Bytes for an array of n_amplitudes complex128 entries."""
    return int(n_amplitudes) * bytes_per_amplitude


class MemoryMonitor:
    r"""
    Periodic RAM re-check for long loops over large states.

    Usage:
        monitor = MemoryMonitor(state.size)
        for gate in gates:
            ...
            monitor.step()

    It never sleeps: between checks, step() only compares a timestamp.

    Attributes:
        active: False for states smaller than config.MONITOR_MIN_AMPLITUDES;
            step() then returns immediately
        checks: Number of RAM checks made so far
    """

    def __init__(self, n_amplitudes):
        self.active = int(n_amplitudes) >= config.MONITOR_MIN_AMPLITUDES
        self.checks = 0
        self._next = time.perf_counter() + config.MONITOR_INTERVAL_S

    def step(self):
        """Call once per loop iteration; reads RAM at most once per interval."""
        if not self.active:
            return
        now = time.perf_counter()
        if now < self._next:
            return
        self.checks += 1
        if available_memory_bytes() < config.MIN_FREE_RAM_MB * MB:
            check_memory()      # fresh reading above; raises with the message
        self._next = now + config.MONITOR_INTERVAL_S
