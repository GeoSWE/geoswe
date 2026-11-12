"""``SWE_HALO_CUDA_AWARE=1`` against an MPI library that cannot take device buffers.

The switch hands device pointers to MPI. A library that cannot take them does not raise:
it reads the pointer as host memory and the rank dies with a segmentation fault, or it
transmits something else. Two layers stand in front of that, and these tests pin both
down without an MPI job:

* ``probe_cuda_aware`` asks the library and falls back to the host-staged halo, with a
  warning, when the library says it has no such support. It used to ask through
  ``MPI.Query_cuda_support``, which mpi4py does not have, so the answer was always
  "trust the variable": two ranks on an Open MPI built without CUDA ended with exit
  code 139 and no message.
* ``check_gpu_aware_exchange`` tests the path once before the run, because a library
  can report the support and still fail between two ranks. It stops the job with the
  remedy in the log; it does not fall back, since a segmentation fault cannot be caught.

The two-rank runs on real MPI traffic are ``benchmark/frontier_amd/partition_check.py``
with ``SWE_HALO_CUDA_AWARE=1``.
"""
import ctypes
import warnings
from types import SimpleNamespace

import pytest

from geoswe import mpi_halo as mh


class _MPIError(Exception):
    """Stands in for ``mpi4py.MPI.Exception``."""


def _fake_mpi(version="Open MPI v5.0.5", waitall=None):
    return SimpleNamespace(Get_library_version=lambda: version, __file__="/nowhere/MPI.so",
                           Exception=_MPIError, MAX="max",
                           Request=SimpleNamespace(Waitall=waitall))


def _library(**answers):
    """A stand-in for the loaded MPI library: one query per keyword, each returning its value."""
    return SimpleNamespace(**{name: (lambda v=value: v) for name, value in answers.items()})


# ---------------------------------------------------------------- what the library says

def test_the_query_is_the_one_of_the_gpu_stack():
    assert mh._gpu_query_names("cuda") == ("MPIX_Query_cuda_support",)
    assert mh._gpu_query_names("hip") == ("MPIX_Query_rocm_support", "MPIX_Query_hip_support")


@pytest.mark.parametrize("stack, library, expected", [
    ("cuda", _library(MPIX_Query_cuda_support=1), True),
    ("cuda", _library(MPIX_Query_cuda_support=0), False),
    ("cuda", _library(), None),                                   # a library without the query
    # On AMD the CUDA answer is not the question: this library would switch a working
    # path off if it were.
    ("hip", _library(MPIX_Query_cuda_support=0, MPIX_Query_rocm_support=1), True),
    ("hip", _library(MPIX_Query_cuda_support=1, MPIX_Query_rocm_support=0), False),
    ("hip", _library(MPIX_Query_hip_support=1), True),            # MPICH's name for it
    ("hip", _library(MPIX_Query_cuda_support=1), None),
])
def test_the_library_is_asked_about_its_own_stack(monkeypatch, stack, library, expected):
    monkeypatch.setattr(mh, "MPI", _fake_mpi())
    monkeypatch.setattr(mh, "_gpu_stack", lambda: stack)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: library)
    assert mh._mpi_reports_gpu_support() is expected


def test_cray_mpich_is_not_asked(monkeypatch):
    """Its GPU support is a per-job switch, read from MPICH_GPU_SUPPORT_ENABLED instead."""
    monkeypatch.setattr(mh, "MPI", _fake_mpi("MPI VERSION    : CRAY MPICH version 8.1.31"))
    monkeypatch.setattr(mh, "_gpu_stack", lambda: "hip")
    monkeypatch.setattr(ctypes, "CDLL", lambda path: pytest.fail("the library was loaded"))
    assert mh._mpi_reports_gpu_support() is None


def test_no_answer_without_cupy_or_without_the_library(monkeypatch):
    monkeypatch.setattr(mh, "MPI", _fake_mpi())
    monkeypatch.setattr(mh, "_gpu_stack", lambda: None)
    assert mh._mpi_reports_gpu_support() is None

    def cannot_load(path):
        raise OSError("cannot open shared object file")
    monkeypatch.setattr(mh, "_gpu_stack", lambda: "cuda")
    monkeypatch.setattr(ctypes, "CDLL", cannot_load)
    assert mh._mpi_reports_gpu_support() is None


# ------------------------------------------------------------------------- the probe

@pytest.fixture
def probe(monkeypatch):
    """``probe_cuda_aware`` with the library's answer set by the test."""
    monkeypatch.setattr(mh, "MPI", _fake_mpi())
    monkeypatch.setattr(mh, "_cray_mpich_gpu_support_off", lambda: False)

    def run(env, says, rank=0):
        asked = []
        monkeypatch.setattr(mh, "_mpi_reports_gpu_support", lambda: asked.append(1) or says)
        if env is None:
            monkeypatch.delenv("SWE_HALO_CUDA_AWARE", raising=False)
        else:
            monkeypatch.setenv("SWE_HALO_CUDA_AWARE", env)
        return mh.probe_cuda_aware(SimpleNamespace(rank=rank)), bool(asked)
    return run


def test_the_path_is_off_unless_it_is_asked_for(probe):
    assert probe(None, True) == (False, False)      # and the library is left alone
    assert probe("0", True) == (False, False)


def test_a_library_without_the_support_gets_the_host_staged_halo(probe):
    with pytest.warns(RuntimeWarning, match="reports no support for device buffers"):
        on, _ = probe("1", False)
    assert on is False


def test_only_rank_0_warns(probe):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on, _ = probe("1", False, rank=1)
    assert on is False


@pytest.mark.parametrize("says", [True, None])
def test_a_library_that_says_yes_or_cannot_be_asked_keeps_the_path(probe, says):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert probe("1", says) == (True, True)


# ---------------------------------------------------------------- the start-up exchange

class _Comm:
    """Rank 0 of a two-rank job whose neighbour, rank 1, behaves as ``deliver`` says."""

    def __init__(self, deliver="intact", flags=None, others_bad=0, size=2):
        self.rank, self.size = 0, size
        self.deliver, self.flags, self.others_bad = deliver, flags, others_bad
        self.calls, self.recvs, self.sends = [], [], []

    def allgather(self, flag):
        return list(self.flags) if self.flags is not None else [flag] * self.size

    def Barrier(self):
        self.calls.append("barrier")

    def Irecv(self, buf, source):
        self.calls.append("irecv")
        self.recvs.append((buf, source))

    def Isend(self, buf, dest):
        self.calls.append("isend")
        self.sends.append((buf, dest))

    def waitall(self, requests):
        """``MPI.Request.Waitall``: the neighbour's side of every posted face."""
        self.calls.append("waitall")
        assert len(requests) == len(self.recvs) + len(self.sends)
        if self.deliver == "error":
            raise _MPIError("MPI_ERR_IN_STATUS: error code in status")
        for (rbuf, source), (sbuf, dest) in zip(self.recvs, self.sends):
            assert source == dest
            if self.deliver == "intact":
                rbuf[...] = mh._gpu_aware_pattern(source, rbuf.size,
                                                  rbuf.dtype).reshape(rbuf.shape)
            elif self.deliver == "own":       # this rank's send buffer came back
                rbuf[...] = sbuf
            # "nothing": the receive buffer is left as it was
        self.recvs, self.sends = [], []

    def allreduce(self, value, op):
        return max(value, self.others_bad)

    def Abort(self, code):
        raise SystemExit(code)                # the real one does not return either


@pytest.fixture
def job(monkeypatch):
    """Builds rank 0's view of a job, with ``mpi_halo`` wired to it."""
    monkeypatch.setattr(mh, "_gpu_aware_announced", False)

    def make(*args, **kwargs):
        comm = _Comm(*args, **kwargs)
        monkeypatch.setattr(mh, "MPI", _fake_mpi(waitall=comm.waitall))
        return comm
    return make


def test_ranks_that_disagree_on_the_switch_are_told_so(job):
    """A variable that reached one node and not the other: without this the ranks that
    have it wait in the test for the ranks that do not, until the wall clock."""
    comm = job(flags=[True, False])
    with pytest.raises(RuntimeError, match="differs across ranks"):
        mh.check_gpu_aware_exchange(comm, True, [])
    assert comm.calls == []


def test_the_host_staged_path_exchanges_nothing(job, capsys):
    comm = job()
    mh.check_gpu_aware_exchange(comm, False, [(1, None, None)])
    assert comm.calls == [] and capsys.readouterr().out == ""


def _faces(cp, dtype="float32"):
    return [(1, cp.empty((3, 4, 50), dtype), cp.empty((3, 4, 50), dtype))]


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_an_intact_exchange_passes_and_leaves_the_buffers_zero(job, capsys, dtype):
    cp = pytest.importorskip("cupy")
    comm, faces = job(), _faces(cp, dtype)
    mh.check_gpu_aware_exchange(comm, True, faces)
    # the remedy is in the log before any device buffer reaches MPI, and every receive
    # is posted before any send
    assert comm.calls == ["barrier", "irecv", "isend", "waitall"]
    assert "SWE_HALO_CUDA_AWARE=0" in capsys.readouterr().out
    assert not bool(faces[0][1].any()) and not bool(faces[0][2].any())
    # once per process: a run builds a dense and a flat halo, and tests both
    mh.check_gpu_aware_exchange(comm, True, faces)
    assert capsys.readouterr().out == ""


@pytest.mark.gpu
@pytest.mark.parametrize("deliver", ["own", "nothing"])
def test_values_that_did_not_cross_stop_the_run(job, deliver):
    cp = pytest.importorskip("cupy")
    with pytest.raises(RuntimeError, match=r"do not cross MPI intact.*rank\(s\) \[1\].*"
                                           r"SWE_HALO_CUDA_AWARE=0"):
        mh.check_gpu_aware_exchange(job(deliver), True, _faces(cp))


@pytest.mark.gpu
def test_every_rank_raises_when_another_rank_saw_the_fault(job):
    cp = pytest.importorskip("cupy")
    with pytest.raises(RuntimeError, match="another rank's did not"):
        mh.check_gpu_aware_exchange(job(others_bad=1), True, _faces(cp))


@pytest.mark.gpu
def test_an_mpi_error_aborts_the_job_with_the_remedy(job, capsys):
    cp = pytest.importorskip("cupy")
    with pytest.raises(SystemExit):
        mh.check_gpu_aware_exchange(job("error"), True, _faces(cp))
    err = capsys.readouterr().err
    assert "MPI_ERR_IN_STATUS" in err and "SWE_HALO_CUDA_AWARE=0" in err


@pytest.mark.gpu
def test_the_pattern_tells_ranks_and_elements_apart():
    cp = pytest.importorskip("cupy")
    a, b = (mh._gpu_aware_pattern(r, 200000, "float32") for r in (0, 1))
    assert not bool((a == b).any())                      # a neighbour's values, never one's own
    assert int(cp.unique(a[:65521]).size) == 65521       # every element differs from the next
    assert float(a.max()) < 2**24                        # exact in float32
