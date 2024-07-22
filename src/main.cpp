#include "dump_xtc.h"
#include "xdr_compat.h"
#include <iostream>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define xdr_getint32(xdrs, int32p) xdrstdio_getint32(xdrs, int32p)
#define xdr_putint32(xdrs, int32p) xdrstdio_putint32(xdrs, int32p)
#define xdr_getuint32(xdrs, uint32p) xdrstdio_getuint32(xdrs, uint32p)
#define xdr_putuint32(xdrs, uint32p) xdrstdio_putuint32(xdrs, uint32p)
#define xdr_getbytes(xdrs, addr, len) xdrstdio_getbytes(xdrs, addr, len)
#define xdr_putbytes(xdrs, addr, len) xdrstdio_putbytes(xdrs, addr, len)
#define xdr_getpos(xdrs) xdrstdio_getpos(xdrs)
#define xdr_setpos(xdrs, pos) xdrstdio_setpos(xdrs, pos)
#define xdr_inline(xdrs, len) xdrstdio_inline(xdrs, len)
#define xdr_destroy(xdrs) xdrstdio_destroy(xdrs)

namespace py = pybind11;

auto pyxdr_header(XDR *xdr, int n, int ntimestep, float time_value,
                  py::array_t<float, py::array::c_style> box) {
  py::buffer_info buf = box.request();
  if (!(buf.ndim == 2 && buf.shape[0] == 3 && buf.shape[1] == 3))
    throw std::runtime_error("box must have shape (3,3)");
  float *box_ptr = static_cast<float *>(buf.ptr);

  int n2 = n;
  int ntimestep2 = ntimestep;
  float time_value2 = time_value;
  bool_t ok = xdr_header(xdr, &n2, &ntimestep2, &time_value2, box_ptr);
  return py::make_tuple(ok, n2, ntimestep2, time_value2);
}

auto pyxdr_data(XDR *xdr,
                std::optional<py::array_t<float, py::array::c_style>> &data,
                int size, float precision) {
  float *data_ptr;
  if (data.has_value()) {
    py::buffer_info buf = data->request();
    if (!(buf.ndim == 2 && buf.shape[0] == size && buf.shape[1] == 3))
      throw std::runtime_error("box must have shape (size,3)");
    data_ptr = static_cast<float *>(buf.ptr);
  } else {
    data_ptr = nullptr;
  }

  int size2 = size;
  float precision2 = precision;
  bool_t ok = xdr3dfcoord(xdr, data_ptr, &size2, &precision2);
  return py::make_tuple(ok, size2, precision2);
}
struct XTCheader {
  int step;
  float time;
  int natoms;
  float box[9];
  float precision;
  off_t offset;
};

template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &seq) {

  auto size = seq.size();
  auto data = seq.data();
  std::unique_ptr<Sequence> seq_ptr =
      std::make_unique<Sequence>(std::move(seq));
  auto capsule = py::capsule(seq_ptr.get(), [](void *p) {
    std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p));
  });
  seq_ptr.release();
  return py::array(size, data, capsule);
}

off_t filesize(FILE *fp) {
  // error checking omitted for clarity
  int fd = fileno(fp);
  struct stat sb;

  fstat(fd, &sb);
  return sb.st_size;
}

py::array_t<XTCheader> pyxdr_index(XDR *xdr, off_t offset, bool only_first) {

  if (xdr->x_op != XDR_DECODE) {
    throw std::runtime_error("must be in read mode");
  }

  auto size = filesize((FILE *)xdr->x_private);

  std::vector<XTCheader> headers;
  headers.reserve(128);
  bool_t ok;
  ok = xdr_setpos(xdr, offset);
  if (ok) {
    int natoms;
    while (true) {
      XTCheader header;
      if (!xdr_header(xdr, &natoms, &header.step, &header.time, &header.box[0]))
        break;

      header.natoms = 0;
      header.precision = 0.;
      if (!xdr3dfcoord(xdr, nullptr, &header.natoms, &header.precision))
        break;
      if (natoms != header.natoms) {
        break;
      }
      header.offset = offset;
      offset = xdrgetpos(xdr);
      if (offset > size)
        break;
      headers.push_back(header);
      if (only_first)
        break;
    }
  }
  return as_pyarray(headers);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        pyxtc - xtc file writer and reader
        -----------------------

        .. currentmodule:: pyxtc

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  py::class_<XDR>(m, "XDR").def(py::init<>());
  m.def("xdropen", &xdropen);
  m.def("xdrclose", &xdrclose);
  m.def("xdrfreebuf", &xdrfreebuf);
  m.def("xdrgetpos", &xdrgetpos);
  m.def("xdrsetpos", &xdrsetpos);

  m.def("header", &pyxdr_header);
  m.def("data", &pyxdr_data);

  PYBIND11_NUMPY_DTYPE(XTCheader, step, time, natoms, box, precision, offset);
  m.def("index", &pyxdr_index, py::return_value_policy::take_ownership);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
