from __future__ import annotations

from . import _core as core
from ._core import __doc__, __version__

__all__ = [
    "__doc__",
    "__version__",
    "XTCReader",
    # "XDR",
    # "xdropen",
    # "xdrclose",
    # "xdrfreebuf",
    # "xdr_header",
    # "xdr3dfcoord",
]

import weakref
from pathlib import Path
from typing import Optional, Union

import numpy as np

_index_dtype = np.dtype(
    [
        ("step", "<i4"),
        ("time", "<f4"),
        ("natoms", "<i4"),
        ("box", "<f4", (9,)),
        ("precision", "<f4"),
        ("offset", "<u4"),
    ]
)


class XTCReader:
    """
    Main class to access MD trajectory data stored in xtc format.

    Supports context manager and generator protocols.
    Refer to the readme and tests for usage examples.

    Scope
    ----------

    This file reader deals with:
        - 0 length files
        - files with partially written frames, truncating that and following data.
    On creation, reader caches header data, indexed by frame.
    This index can be updated manually, cached after construction, serialized to disk, and updated.

    Attributes
    ----------

    frame: Optional[int]
        the current frame, None if none has been read yet.
    X : numpy.ndarray
        the atom coordinates of the current frame.
    index:
        structured numpy.ndarray with _index_dtype
            ("step", "<i4"),
            ("time", "<f4"),
            ("natoms", "<i4"),
            ("box", "<f4", (9,)),
            ("precision", "<f4"),
            ("offset", "<u4"),
    """

    X: Optional[np.ndarray]
    frame: Optional[int]

    @classmethod
    def open(cls, fname: Union[Path, str], mode: str = "r"):
        assert mode == "r"
        obj = cls(fname)
        obj.update_index()
        return obj

    @classmethod
    def _finalize(cls, obj_dict):
        v = obj_dict.get("_fxtc", None)
        if v is not None:
            core.xdrclose(v)

    def __init__(self, file: Union[Path, str]):
        """
        Parameters
        ----------
        fname : str
            Input file name

        interface
            __iter__
            read(n:Optional[int]=None)
            __len__
            seek(n:int)
        """
        weakref.finalize(self, self.__class__._finalize, self.__dict__)
        self.index: np.ndarray = np.empty((0,), dtype=_index_dtype)
        self.X = None
        self._box = np.empty((3, 3), dtype=np.float32)
        assert isinstance(file, (str, Path))

        self._fxtc = core.xdropen(str(file), "rb")
        self._fstat = Path(file).stat()
        self._file = file
        self.frame = None

    @staticmethod
    def __unreduce__(
        _index,
        _fstat,
        _fname,
    ):
        obj = XTCReader(
            _fname,
        )
        obj.index = _index
        obj._fstat = _fstat
        return obj

    def __reduce__(self):
        if self._file is None:
            raise ValueError
        return self.__class__.__unreduce__, (
            self.index,
            self._fstat,
            self._file,
        )

    def seek(self, frame: int, data: bool = False):
        """
        validates header against cache,throwing in case of mismatch
        loads header and optionally data, updating .box and .X
        """
        assert self.index is not None

        h = self.index[frame]
        ok = core.xdrsetpos(self._fxtc, h["offset"])
        if not ok:
            raise RuntimeError
        ok, natoms, step, time = core.header(self._fxtc, 0, 0, 0.0, self._box)
        if self.X is None or self.X.shape[0] != natoms:
            self.X = np.empty((natoms, 3), dtype=np.float32)
        assert h["step"] == step and h["time"] == time and natoms == h["natoms"]
        if not ok:
            raise RuntimeError
        ok, natoms2, precision = core.data(
            self._fxtc, self.X if data else None, natoms, 0.0
        )
        assert natoms2 == natoms
        assert h["precision"] == precision

        self.frame = frame

    def update_index(self):
        def index_from_zero():
            self.index = core.index(self._fxtc, 0, False)

        if len(self.index) == 0:
            index_from_zero()
        else:
            if self._file:
                fstat_old = self._fstat
                assert fstat_old is not None
                self._fstat = Path(self._file).stat()
                if self._fstat.st_mtime == fstat_old.st_mtime:
                    return False
                elif self._fstat.st_size < fstat_old.st_size:
                    index_from_zero()
                    return True
            h = self.index[-1]
            nhs = core.index(self._fxtc, h["offset"], True)
            ok = len(nhs) == 1
            if ok and (nh := nhs[0])["step"] == h["step"] and nh["time"] == h["time"]:
                # index_from_current
                index2 = core.index(self._fxtc, core.xdrgetpos(self._fxtc), False)

                self.index = np.append(self.index, index2)
            else:
                index_from_zero()
        return True

    def close(self):
        if self._fxtc is not None:
            core.xdrclose(self._fxtc)
            self._fxtc = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __len__(self):
        return len(self.index)

    def __iter__(self):
        for frame in range(len(self)):
            self.seek(frame, data=True)
            assert self.X is not None
            yield self.index[frame]["time"], self.index[frame]["step"], self.index[
                frame
            ]["box"].reshape(3, 3).copy(), self.X.copy()
