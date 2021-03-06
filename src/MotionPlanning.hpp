#ifndef _MOTIONPLANNING_LIBRARIES_HPP_
#define _MOTIONPLANNING_LIBRARIES_HPP_

#include <base/samples/RigidBodyState.hpp>
#include <base/samples/DistanceImage.hpp>
#include <base/samples/Frame.hpp>
#include <base/Waypoint.hpp>

#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <cmath>

namespace MotionPlanning_lib
{
    class MotionPlanning
    {
        public:
			PyObject* initPython(char* pyName);
			void runPyFunction(char pyFunctionName[], PyObject *pModule, double xm, double ym, double xr, double yr, double initHeading, char mapDirectory[], double resolution, double size);
			void sizePyArray( int &size, char pyVariableName[],  PyObject *pModule);
			void returnPyArrayDouble(int nDim, char pyVariableName[], double* &dVariable, PyObject *pModule);
			void returnPyArrayInt(int nDim, char pyVariableName[], int* &iVariable, PyObject *pModule);
			int shutDownPython(PyObject *pModule);
    };

} // end namespace coupled_motion_planner

#endif
