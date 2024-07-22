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
    make_traj,
    write_traj,
)


def test_incremental():
    traj = make_traj()
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
    for frame in frame_order:
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
        traj = make_traj()
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
        traj = make_traj()
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


def make_traj_frames_iter(
    *, nframes: int, natoms: Optional[int] = None, precision: Optional[float] = None
):
    gen = np.random.default_rng(seed=0)
    if precision is None:
        precision = 10.0
    for frame in tqdm(range(1, nframes + 1), total=nframes):
        step = gen.integers(low=-1000, high=100)
        time = (gen.random(dtype=np.float32, size=(1,)) * 1000.0)[
            0
        ]  # float32 scalar array gets converted to float if multiplied
        natoms = frame + (frame > 10) * 100 if natoms is None else natoms
        box = gen.random((3, 3), dtype=np.float32) + 1
        data = (gen.random((natoms, 3), dtype=np.float32) - 0.5) * 100.0
        yield (
            XTCframe(
                step=step,
                time=time,
                box=box,
                precision=10.0 if natoms > 9 else 0.0,
                natoms=natoms,
                coords=data,
                offset=-1,
            )
        )


def write_traj_iter(frames: Iterable[XTCframe], filename: str, mode: str = "wb+"):
    x = core.xdropen(filename, mode)
    try:
        for frame in frames:
            assert frame.coords is not None
            n = frame.coords.shape[0]
            core.header(x, n, frame.step, frame.time, frame.box)
            core.data(x, frame.coords, n, frame.precision)
    finally:
        core.xdrclose(x)


def make_large_traj(
    traj_path: Path,
    natoms: int = 10000,
    nframes: int = 1000,
    precision: Optional[float] = None,
):
    write_traj_iter(
        make_traj_frames_iter(natoms=natoms, nframes=nframes, precision=precision),
        str(traj_path),
    )


@pytest.fixture(scope="module")
def large_traj():
    d = Path(__file__).parent / ".cache"
    d.mkdir(exist_ok=True)
    path = d / "large_traj.xtc"
    if not path.exists():
        make_large_traj(path, natoms=100000, nframes=1000)
    return path


@pytest.fixture(scope="module")
def huge_traj():
    d = Path(__file__).parent / ".cache"
    d.mkdir(exist_ok=True)
    path = d / "huge_traj.xtc"
    if not path.exists():
        make_large_traj(path, natoms=174246, nframes=4000, precision=1000.0)
    return path


@pytest.mark.parametrize("data", [False, True])
def test_perf(benchmark, large_traj, data):
    def xtc_info(traj_path: Path, data):
        with XTCReader.open(traj_path) as reader:
            if data:
                for i in range(len(reader.index)):
                    reader.seek(i, data=True)

    benchmark(partial(xtc_info, large_traj), data)


def test_offset(huge_traj):
    with XTCReader.open(huge_traj) as reader:
        assert len(reader.index) == 4000


def test_memory_leak(large_traj):
    # with tempfile.NamedTemporaryFile() as f:
    # traj = make_traj()
    # write_traj(traj, f.name)

    for i in range(100):
        with XTCReader.open(large_traj) as rd:
            XTCreader_to_traj(rd)
