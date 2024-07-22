from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, Optional

import numpy as np
import yaml

import pyxtc as m
import pyxtc._core as core
from tqdm.auto import tqdm


@dataclass(frozen=True)
class XTCframe:
    step: int
    time: float
    box: np.ndarray
    precision: float
    natoms: int
    coords: Optional[np.ndarray]
    offset: int

    def __post_init__(self):
        if self.coords is not None:
            assert self.coords.shape[1] == 3
            assert self.coords.shape[0] == self.natoms
            if self.coords.shape[0] <= 9:
                assert self.precision == 0.0


@dataclass(frozen=True)
class XTCtraj:
    frames: list[XTCframe]

    def to_index(self):
        return np.array(
            [
                (f.step, f.time, f.natoms, f.box.reshape(-1), f.precision, f.offset)
                for f in self.frames
            ],
            dtype=m._index_dtype,
        )


def make_traj_lmp(*, path: Path, nframes: int = 10, natoms: int, precision=1000.0):
    l = (natoms ** (1 / 3)) * 2
    lmp_str = textwrap.dedent(
        f"""\
        units           lj
        atom_style      atomic
        boundary        p p p
        region          box block 0 {l} 0 {l} 0 {l}
        create_box      1 box
        create_atoms    1 random {natoms} 1 NULL overlap 0.2
        mass            1 1.0

        velocity        all create 1.0 1
        

        pair_style      lj/cut 2.5
        pair_coeff      1 1 1.0 1.0 2.5

        neighbor        0.3 bin
        neigh_modify    delay 0 every 1 check yes

        
        minimize 1.0e-4 1.0e-6 100 1000
        timestep 0.001
        
        
        fix             1 all npt temp 1.0 1.0 1. iso 0.001 0.001 1000.
        thermo 1
        thermo_style custom step dt lx ly lz
        thermo_modify line yaml

        dump dump_yaml all yaml 1 traj.yaml x y z ix iy iz
        dump_modify dump_yaml sort id format float "%.20f"
        dump dump_xtc all xtc 1 traj.xtc
        dump_modify dump_xtc precision {precision} sfactor 1. tfactor 1. unwrap yes

        run             {nframes}
        """
    )

    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    (path / "in.lmp").write_text(lmp_str)

    subprocess.run("lmp_mpi -in in.lmp -screen none", shell=True, cwd=path, check=True)

    subprocess.run(
        r"egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > log.yaml",
        shell=True,
        cwd=path,
        check=True,
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
        natoms = (frame + (frame > 10) * 100) if natoms is None else natoms
        box = gen.random((3, 3), dtype=np.float32) + 1
        data = (gen.random((natoms, 3), dtype=np.float32) - 0.5) * 100.0
        yield (
            XTCframe(
                step=step,
                time=time,
                box=box,
                precision=precision if natoms > 9 else 0.0,
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


def read_lmp_yaml(p, precision: float, time_scale_factor: float):
    with (p / "log.yaml").open("r") as thermo_f:
        thermo = yaml.safe_load(thermo_f)
    with (p / "traj.yaml").open("r") as traj_f:
        traj = list(yaml.safe_load_all(traj_f))

    assert thermo["keywords"] == ["Step", "Dt", "Lx", "Ly", "Lz"]

    natoms_set = set([len(frame["data"]) for frame in traj])
    assert len(natoms_set) == 1
    natoms = list(natoms_set)[0]

    frames = []
    for frame, thermo_frame in zip(traj, thermo["data"], strict=True):
        assert thermo_frame[0] == frame["timestep"]

        lab = np.array(frame["box"])  # 3 by 2
        box_origin = lab[:, 0]
        box_l = lab[:, 1] - lab[:, 0]

        assert frame["keywords"] == ["x", "y", "z", "ix", "iy", "iz"]
        data = np.array(frame["data"])  # (natoms,6)
        p, q = data[:, :3], data[:, 3:]
        coords = p + q * box_l

        box = np.array(
            [[box_l[0], 0.0, 0.0], [0.0, box_l[1], 0.0], [0.0, 0.0, box_l[2]]]
        )
        frames.append(
            XTCframe(
                step=frame["timestep"],
                time=frame["timestep"] * time_scale_factor * thermo_frame[1],
                box=box,
                precision=precision if natoms > 9 else 0.0,
                natoms=natoms,
                coords=coords,
                offset=-1,
            )
        )
    return XTCtraj(frames)


def write_traj(traj: XTCtraj, filename: str, mode: str = "wb+"):
    x = core.xdropen(filename, mode)
    try:
        for frame in tqdm(traj.frames):
            assert frame.coords is not None
            n = frame.coords.shape[0]
            core.header(x, n, frame.step, frame.time, frame.box)
            core.data(x, frame.coords, n, frame.precision)
    finally:
        core.xdrclose(x)


def read_traj(filename: str, skip_data: bool = False):
    x = core.xdropen(filename, "r")
    try:
        frames = []
        coords = None
        offset = 0
        while True:
            box = np.ones((3, 3), dtype=np.float32)
            offset = core.xdrgetpos(x)
            ok, n, step, time = core.header(x, 0, 0, 0.0, box)
            if not ok:
                break
            if not skip_data:
                coords = np.zeros((n, 3), dtype=np.float32)

            ok, n2, precision = core.data(x, coords, n, 0.0)

            if not ok:
                break
            assert n2 == n
            frames.append(
                XTCframe(
                    step=step,
                    time=time,
                    box=box,
                    precision=precision,
                    natoms=n,
                    coords=coords,
                    offset=offset,
                )
            )
        return XTCtraj(frames=frames)
    finally:
        core.xdrclose(x)


def read_index(filename: str):
    x = core.xdropen(filename, "r")
    try:
        return core.index(x, 0, False)
    finally:
        core.xdrclose(x)


@dataclass(frozen=True)
class Opts:
    coords: bool
    coords_equal: bool
    offset: bool


def assert_frame_equivalent(
    frame: XTCframe,
    frame2: XTCframe,
    opts: Opts,
):
    f32 = lambda x: np.array(x, dtype=np.float32)
    assert frame.step == frame2.step
    assert (a := f32(frame.time)) == (b := f32(frame2.time)), f"{a}!={b}"
    assert ((a := f32(frame.box)) == (b := f32(frame2.box))).all(), f"{a}!={b}"

    assert frame.precision == frame2.precision, f"{frame.precision}!={frame2.precision}"

    assert frame.natoms == frame2.natoms

    if opts.coords:
        if (frame.coords is None) and (frame2.coords is None):
            pass
        else:
            assert frame.coords is not None and frame2.coords is not None
            assert frame.coords.shape[0] == frame2.coords.shape[0]
            n = frame.coords.shape[0]
            if opts.coords_equal:
                assert (frame.coords == frame2.coords).all()
            else:
                if n > 9:
                    trunc = (
                        lambda x, p: (
                            f32(x) * f32(p) + np.where(x >= 0, 1, -1) * 0.5
                        ).astype(int)
                        / p
                    )
                    assert (
                        (a := trunc(frame.coords, frame.precision))
                        == (b := trunc(frame2.coords, frame2.precision))
                    ).all(), f"{(a[(j:=np.where((a!=b).all(axis=1)))],b[j],frame.coords[j],frame2.coords[j],frame.precision,frame2.precision,n)}"
                else:
                    assert (
                        (a := f32(frame.coords)) == (b := f32(frame2.coords))
                    ).all(), f"{a}!={b}"

    if opts.offset:
        assert frame.offset == frame2.offset


def assert_traj_equivalent(traj: XTCtraj, traj2: XTCtraj, opts: Opts):
    assert len(traj.frames) == len(traj2.frames)

    for frame_i, (frame, frame2) in enumerate(zip(traj.frames, traj2.frames)):
        assert_frame_equivalent(frame, frame2, opts)


def test_read_write():
    with NamedTemporaryFile() as f:
        traj = XTCtraj(frames=list(make_traj_frames_iter(nframes=40)))
        write_traj(traj, f.name)

        traj2 = read_traj(f.name)
        assert_traj_equivalent(
            traj, traj2, Opts(coords=True, coords_equal=False, offset=False)
        )

        write_traj(traj2, f.name)
        traj3 = read_traj(f.name)
        assert_traj_equivalent(
            traj2, traj3, Opts(coords=True, coords_equal=True, offset=True)
        )
        traj3 = read_traj(f.name, skip_data=True)
        ind = traj3.to_index()

        ind2 = read_index(f.name)
        assert (ind == ind2).all()

        with NamedTemporaryFile() as f2:
            b = Path(f.name).read_bytes()
            b2 = b[: (ind["offset"][-1] + len(b)) // 2]

            with Path(f2.name).open("wb") as f3:
                f3.write(b2)
            assert os.path.getsize(f2.name) < os.path.getsize(f.name)

            ind3 = read_index(f2.name)
            assert len(ind3) == (len(ind) - 1)
            assert (ind3 == ind[:-1]).all()


def get_traj(*, natoms: int, nframes: int, precision: float):
    d = Path(__file__).parent / ".cache"
    d.mkdir(exist_ok=True)
    p = d / f"traj_{nframes=}_{natoms=}_{precision=}.xtc"
    if not p.exists():
        write_traj_iter(
            make_traj_frames_iter(natoms=natoms, nframes=nframes, precision=precision),
            str(p),
        )
    return p


def get_lmp_run(*, natoms: int, nframes: int, precision: float):
    d = Path(__file__).parent / ".cache"
    d.mkdir(exist_ok=True)
    p = d / f"lmp_{nframes=}_{natoms=}_{precision=}"
    if not p.exists():
        make_traj_lmp(path=p, natoms=natoms, precision=precision)
    return p


def test_lmp():
    for natoms, precision in product([2, 9, 10, 100], [10.0, 100.0, 1000.0]):
        p = get_lmp_run(natoms=natoms, precision=precision, nframes=10)
        traj = read_lmp_yaml(p, precision=precision, time_scale_factor=1.0)
        traj2 = read_traj(str(p / "traj.xtc"))
        assert_traj_equivalent(
            traj, traj2, Opts(coords=True, coords_equal=False, offset=False)
        )
