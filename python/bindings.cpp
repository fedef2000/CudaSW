#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cusw/aligner.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cusw, m) {
    m.doc() = "CUDA Smith-Waterman Library";

    py::class_<cusw::Aligner>(m, "Aligner")
        .def(py::init<int>(), py::arg("gpu_id") = 0)
        .def("score_batch", &cusw::Aligner::score_batch,
             "Calculate Smith-Waterman scores for a batch of pairs",
             py::arg("queries"),
             py::arg("targets"),
             py::arg("match_score") = 2,
             py::arg("mismatch_score") = -1,
             py::arg("gap_open") = -1,
             py::arg("gap_extend") = -1
        );
}
