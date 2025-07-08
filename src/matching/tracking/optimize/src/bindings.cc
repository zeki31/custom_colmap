// ParticleSfM
// Copyright (C) 2022  ByteDance Inc.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "trajectory_base.h"

PYBIND11_MODULE(particlesfm, m){
    using namespace particlesfm;
    m.doc() = "pybind11 for point trajectories";

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<int, V2D, VXD>(), py::arg("time"), py::arg("point"), py::arg("desc"))
        .def(py::init<double, V2D, VXD>(), py::arg("time"), py::arg("point"), py::arg("desc"))
        .def(py::init<const std::vector<int>&, const std::vector<V2D>&, const std::vector<VXD>&>(), py::arg("times"), py::arg("xys"), py::arg("descs"))
        .def(py::init<py::dict>())
        .def("as_dict", &Trajectory::as_dict)
        .def(py::pickle(
            [](const Trajectory& input) { // dump
                return input.as_dict();
            },
            [](const py::dict& dict) { // load
                return Trajectory(dict);
            }
        ))
        .def_readonly("times", &Trajectory::times)
        .def_readonly("xys", &Trajectory::xys)
        .def_readonly("descs", &Trajectory::descs)
        .def("extend", &Trajectory::extend)
        .def("length", &Trajectory::length)
        .def("get_tail_location", &Trajectory::get_tail_location);

    py::class_<TrajectorySet>(m, "TrajectorySet")
        .def(py::init<>())
        .def(py::init<const std::map<int, Trajectory>&>())
        .def(py::init<std::map<int, py::dict>>())
        .def("as_dict", &TrajectorySet::as_dict)
        .def(py::pickle(
            [](const TrajectorySet& input) { // dump
                return input.as_dict();
            },
            [](const std::map<int, py::dict>& dict) { // load
                return TrajectorySet(dict);
            }
        ))
        .def_readonly("trajs", &TrajectorySet::trajs)
        .def("build_invert_indexes", &TrajectorySet::build_invert_indexes)
        .def_readonly("invert_maps", &TrajectorySet::invert_maps);

}
