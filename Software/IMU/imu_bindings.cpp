
// imu_bindings.cpp
#include <pybind11/pybind11.h>
#include "RPI_BNO055.h"
#include <stdexcept>

namespace py = pybind11;
// A simple wrapper class for the sensor class PyBNO055
py::class_<PyBNO055>(m, "BNO055")
	.def(py::init<>())
	.def("get_orientation_x", &PyBNO055::getOrientation);
	// Extract other methods similarly
	
	
// simple cpp function to be called in python
double sum(double a, double b) {
	return a + b;
}

PYBIND11_MODULE(SumFunction, var) {
	var.doc() = "pybind11 example module";
	var.def("sum", &sum, "This function adds two input numbers");
}
