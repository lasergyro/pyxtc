import os
import pickle
import tempfile
from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pytest
from tqdm import tqdm

import pyxtc._core as core
from pyxtc import XTCReader
from tests.test_core import (
    Opts,
    XTCframe,
    XTCtraj,
    assert_traj_equivalent,
    write_traj,
)


def test_incremental():
    traj = XTCtraj(frames=list(make_traj_frames_iter(nframes=40)))
    a = len(traj.frames) // 2

    with tempfile.NamedTemporaryFile() as f:
        write_traj(XTCtraj(frames=traj.frames[:a]), f.name)

        with XTCReader.open(f.name) as r:
            assert len(r) == a
            write_traj(XTCtraj(frames=traj.frames[a:]), f.name, "ab")
            r.update_index()
            assert len(r) == len(traj.frames), f"{len(r):=},{len(traj.frames):=}"


def XTCreader_to_traj(rd: XTCReader) -> XTCtraj:
    rng = np.random.default_rng(0)
    nframes = len(rd.index)
    frames = [None] * nframes
    frame_order = rng.choice(nframes, nframes, replace=False)
    for frame in tqdm(frame_order):
        rd.seek(frame=frame, data=True)
        assert rd.X is not None
        header = rd.index[frame]
        frames[frame] = XTCframe(
            step=int(header["step"]),
            time=float(header["time"]),
            box=header["box"].reshape(3, 3).copy(),
            natoms=int(header["natoms"]),
            coords=rd.X.copy(),
            precision=float(header["precision"]),
            offset=int(header["offset"]),
        )
    return XTCtraj(frames)  # type:ignore


def read_traj_reader(p: Path) -> XTCtraj:
    with XTCReader.open(str(p)) as rd:
        return XTCreader_to_traj(rd)


def read_index_reader(p: Path):
    with XTCReader.open(str(p)) as rd:
        return rd.index


def test_read():
    with tempfile.NamedTemporaryFile() as f:
        traj = XTCtraj(frames=list(make_traj_frames_iter(nframes=40)))
        write_traj(traj, f.name)

        traj2 = read_traj_reader(Path(f.name))

        assert_traj_equivalent(
            traj, traj2, Opts(coords=True, coords_equal=False, offset=False)
        )

        write_traj(traj2, f.name)
        traj3 = read_traj_reader(Path(f.name))
        assert_traj_equivalent(
            traj2, traj3, Opts(coords=True, coords_equal=True, offset=True)
        )

        ind = traj3.to_index()
        ind2 = read_index_reader(Path(f.name))

        assert (ind == ind2).all()

        with tempfile.NamedTemporaryFile() as f2:
            b = Path(f.name).read_bytes()
            b2 = b[: (ind["offset"][-1] + len(b)) // 2]

            with Path(f2.name).open("wb") as f3:
                f3.write(b2)

            assert os.path.getsize(f2.name) < os.path.getsize(f.name)

            ind3 = read_index_reader(Path(f2.name))
            assert len(ind3) == (len(ind) - 1)
            assert (ind3 == ind[:-1]).all()


def test_serialize():
    with tempfile.NamedTemporaryFile() as f:
        traj = XTCtraj(frames=list(make_traj_frames_iter(nframes=40)))
        write_traj(traj, f.name)

        with XTCReader.open(f.name) as rd:
            traj2 = XTCreader_to_traj(rd)
            index2 = rd.index
            rd_s = pickle.dumps(rd)
        with pickle.loads(rd_s) as rd:
            assert (rd.index == index2).all()
            traj3 = XTCreader_to_traj(rd)
        assert_traj_equivalent(
            traj2, traj3, Opts(coords=True, coords_equal=True, offset=True)
        )


from .test_core import get_traj, make_traj_frames_iter


@pytest.fixture(scope="module")
def large_traj():
    return get_traj(natoms=100000, nframes=100, precision=1000.0)


@pytest.fixture(scope="module")
def huge_traj():
    return get_traj(natoms=17424, nframes=4000, precision=1000.0)


@pytest.mark.parametrize("data", [False, True])
def test_perf(benchmark, large_traj, data):
    def xtc_info(traj_path: Path, data):
        with XTCReader.open(traj_path) as reader:
            if data:
                for i in range(
                    len(reader.index)
                ):  #:tqdm(range(len(reader.index)), total=len(reader.index)):
                    reader.seek(i, data=True)

    benchmark(partial(xtc_info, large_traj), data)


def test_offset(huge_traj):
    with XTCReader.open(huge_traj) as reader:
        assert len(reader.index) == 4000


def test_memory_leak(large_traj):
    for i in range(10):
        with XTCReader.open(large_traj) as rd:
            XTCreader_to_traj(rd)
