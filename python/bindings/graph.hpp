#pragma once
#include <pybind11/pytypes.h>
#ifndef PYTHON_GRAPH_HPP
#define PYTHON_GRAPH_HPP
#include "core/graph_builder.h"
#include "core/runtime.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infini {
void bind_graph_builder(py::module &m) {
    py::class_<GraphObj, std::shared_ptr<GraphObj>>(m, "Graph");
    // GraphBuilder
    py::class_<GraphBuilderObj>(m, "GraphBuilder")
        .def(py::init<Runtime>())
        .def("tensor", &GraphBuilderObj::tensor, py::arg("dims"),
             py::arg("dtype"), py::arg("stride") = py::none())
        .def("gemm", &GraphBuilderObj::gemm, py::arg("A"), py::arg("B"),
             py::arg("C"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0,
             py::arg("transA") = false, py::arg("transB") = false,
             py::arg("Y") = py::none())
        .def("add", &GraphBuilderObj::add, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("sub", &GraphBuilderObj::sub, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("mul", &GraphBuilderObj::mul, py::arg("A"), py::arg("B"),
             py::arg("Y") = py::none())
        .def("clip", &GraphBuilderObj::clip, py::arg("Input"),
             py::arg("MinVal") = py::none(), py::arg("MaxVal") = py::none(),
             py::arg("Output") = py::none())
        .def("relu", &GraphBuilderObj::relu, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("sigmoid", &GraphBuilderObj::sigmoid, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("gelu", &GraphBuilderObj::gelu, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("silu", &GraphBuilderObj::silu, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("softplus", &GraphBuilderObj::softplus, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("tanh", &GraphBuilderObj::tanh, py::arg("Input"),
             py::arg("Output") = py::none())
        .def("to_string", &GraphBuilderObj::printGraph)
        .def_property_readonly("graph", &GraphBuilderObj::getGraph);
}

} // namespace infini
#endif // PYTHON_GRAPH_HPP
