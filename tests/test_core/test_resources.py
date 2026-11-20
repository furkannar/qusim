"""Tests for the RAM guards (qusim_np/resources.py)."""

from collections import namedtuple

import numpy as np
import pytest

import qusim_np
from qusim_np import QuantumCircuit, NoiseModel, depolarizing_kraus
from qusim_np import resources
from qusim_np.config import GB, KB, MB, config
from qusim_np.resources import (
    MemoryMonitor, available_memory_bytes, available_memory_gb, check_memory, state_bytes,
)

_VMem = namedtuple("svmem", "total available")


@pytest.fixture
def fake_ram(monkeypatch):
    """Pretend the machine has `available` bytes free; returns a setter."""
    def set_available(available):
        monkeypatch.setattr(resources.psutil, "virtual_memory",
                            lambda: _VMem(total=32 * GB, available=available))
        resources._last_reading.time = float("-inf")     # drop the cached reading
    yield set_available
    resources._last_reading.time = float("-inf")


class TestMemory:
    def test_available_memory_is_real(self):
        assert available_memory_bytes() > 0
        assert np.isclose(available_memory_gb() * GB, available_memory_bytes(), rtol=0.05)

    def test_state_bytes(self):
        assert state_bytes(2 ** 10) == 16 * 2 ** 10
        assert state_bytes(4, 32) == 128

    def test_plenty_of_ram_passes(self, fake_ram):
        fake_ram(8 * GB)
        assert check_memory(1 * GB) == 8 * GB

    def test_low_ram_raises(self, fake_ram):
        fake_ram(100 * MB)
        with pytest.raises(RuntimeError, match="Dangerously low RAM: only 100 MB left"):
            check_memory()

    def test_allocation_that_would_go_below_floor_raises(self, fake_ram):
        fake_ram(500 * MB)
        check_memory(372 * MB)                       # leaves exactly 128 MB
        with pytest.raises(RuntimeError, match="would leave 127 MB"):
            check_memory(373 * MB)

    def test_floor_comes_from_config(self, fake_ram, monkeypatch):
        fake_ram(200 * MB)
        check_memory()
        monkeypatch.setattr(config, "MIN_FREE_RAM_MB", 256)
        with pytest.raises(RuntimeError, match="minimum free: 256 MB"):
            check_memory()

    def test_small_requests_reuse_recent_reading(self, fake_ram, monkeypatch):
        fake_ram(8 * GB)
        check_memory(1 * GB)                    # large: fresh reading, cached
        calls = []
        monkeypatch.setattr(resources.psutil, "virtual_memory",
                            lambda: calls.append(1) or _VMem(0, 8 * GB))
        check_memory(KB)                           # small and recent: cached
        assert calls == []
        check_memory(1 * GB)                    # large: always fresh
        assert calls == [1]

    @pytest.mark.parametrize("run", [
        lambda: qusim_np.grover_search(4, [3]),
        lambda: qusim_np.deutsch_jozsa(4, lambda x: 0),
        lambda: qusim_np.qft_matrix(3),
        lambda: qusim_np.qft_state(np.eye(8)[1]),
        lambda: qusim_np.hamiltonian([1.0], ["ZZ"]),
        lambda: qusim_np.hea_state(2, np.zeros(4)),
        lambda: qusim_np.to_density_matrix(np.eye(4)[0]),
        lambda: QuantumCircuit(2).h(0).statevector(),
        lambda: QuantumCircuit(2, NoiseModel().add_gate_error(depolarizing_kraus(0.1), gates="h"))
                .h(0).density_matrix(),
        lambda: qusim_np.shor(15, a=7),
        lambda: qusim_np.estimate_capacity(),
    ], ids=["grover", "deutsch_jozsa", "qft_matrix", "qft_state", "hamiltonian",
            "hea_state", "density_matrix", "circuit", "noisy_circuit", "shor", "capacity"])
    def test_entry_points_refuse_on_low_ram(self, fake_ram, run):
        fake_ram(64 * MB)
        with pytest.raises(RuntimeError, match="Dangerously low RAM"):
            run()


class TestMemoryMonitor:
    @pytest.fixture
    def clock(self, monkeypatch):
        """Fake clock: time only moves when the test advances it."""
        now = [0.0]
        monkeypatch.setattr(resources.time, "perf_counter", lambda: now[0])
        return now

    @pytest.fixture
    def no_pausing(self, monkeypatch):
        """Fail the test if anything sleeps or reads the CPU load."""
        def forbidden(*args, **kwargs):
            raise AssertionError("the memory monitor must not pause or read CPU load")
        monkeypatch.setattr(resources.time, "sleep", forbidden)
        monkeypatch.setattr(resources.psutil, "cpu_percent", forbidden)

    def test_small_states_are_not_monitored(self, clock):
        t = MemoryMonitor(config.MONITOR_MIN_AMPLITUDES - 1)
        for _ in range(10):
            clock[0] += 10
            t.step()
        assert not t.active and t.checks == 0

    def test_checks_once_per_interval(self, clock, no_pausing):
        t = MemoryMonitor(config.MONITOR_MIN_AMPLITUDES)
        for _ in range(100):                          # 100 steps of 0.1 s = 10 s
            clock[0] += 0.1
            t.step()
        # about one check per 0.5 s interval (float steps make some intervals
        # 6 steps long)
        assert 16 <= t.checks <= 20

    def test_steps_between_checks_skip_psutil(self, clock, monkeypatch):
        calls = []
        monkeypatch.setattr(resources.psutil, "virtual_memory",
                            lambda: calls.append(1) or _VMem(0, 8 * GB))
        t = MemoryMonitor(config.MONITOR_MIN_AMPLITUDES)
        for _ in range(1000):
            t.step()                                  # clock does not move
        assert calls == []

    def test_ram_rechecked_during_long_loop(self, clock, fake_ram):
        fake_ram(8 * GB)
        t = MemoryMonitor(config.MONITOR_MIN_AMPLITUDES)
        clock[0] += 1
        t.step()
        fake_ram(50 * MB)                             # RAM fills up mid-run
        clock[0] += 1
        with pytest.raises(RuntimeError, match="Dangerously low RAM"):
            t.step()

    def test_large_run_never_pauses(self, monkeypatch, no_pausing):
        """A Grover run above the size threshold is monitored but never sleeps."""
        monkeypatch.setattr(config, "MONITOR_MIN_AMPLITUDES", 2 ** 10)
        monkeypatch.setattr(config, "MONITOR_INTERVAL_S", 0.0)     # check every step
        _, k, p = qusim_np.grover_search(10, [123])
        assert k == 25 and p > 0.99

    def test_large_circuit_never_pauses(self, monkeypatch, no_pausing):
        monkeypatch.setattr(config, "MONITOR_MIN_AMPLITUDES", 2 ** 4)
        monkeypatch.setattr(config, "MONITOR_INTERVAL_S", 0.0)
        qc = QuantumCircuit(6).h(0)
        for k in range(5):
            qc.cx(k, k + 1)
        assert qc.sample_counts(100, rng=np.random.default_rng(0)).keys() <= {"000000", "111111"}
