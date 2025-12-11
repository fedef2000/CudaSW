#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Auto-converts C++ vector to Python list
#include "cusw/aligner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cusw, m) {
    m.doc() = "CUDA Smith-Waterman Library";

    py::class_<cusw::Aligner>(m, "Aligner")
        .def(py::init<int>(), py::arg("gpu_id") = 0)
        .def("score_batch", &cusw::Aligner::score_batch);
}
