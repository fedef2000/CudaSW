#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sw_cuda.h"

namespace py = pybind11;

PYBIND11_MODULE(sw_cuda_py, m) {
    m.doc() = "HPC CUDA Smith-Waterman Library";

    // Bind the Config struct
    py::class_<SWConfig>(m, "SWConfig")
        .def(py::init<int, int, int>(), 
             py::arg("match_score"), py::arg("mismatch_score"), py::arg("gap_score"))
        .def_readwrite("match_score", &SWConfig::match_score)
        .def_readwrite("mismatch_score", &SWConfig::mismatch_score)
        .def_readwrite("gap_score", &SWConfig::gap_score);

    // Bind functions
    m.def("sw_cuda_tiled", &sw_cuda_tiled, "Tiled CUDA implementation");
    m.def("sw_cuda_diagonal", &sw_cuda_diagonal, "Diagonal Wavefront CUDA implementation");
    m.def("sw_cpu", &sw_cpu, "CPU Reference implementation");
    m.def("sw_cuda_o2m", &sw_cuda_o2m, "One-to-Many Batch CUDA implementation");
}