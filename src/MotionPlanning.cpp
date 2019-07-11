#include "MotionPlanning.hpp"

using namespace MotionPlanning_lib;

PyObject* MotionPlanning::initPython(char* pyName)
{

	std::cout<<"Loading python file named '"<<pyName<<"'...";
    PyObject *pName, *pModule;


    Py_Initialize();
	import_array(); 

    pName = PyUnicode_DecodeFSDefault(pyName);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
		std::cout<<" done"<<std::endl;
	}	
	else {
        PyErr_Print();
        fprintf(stderr, "failed to load \n");
    }
	return pModule;
}

void MotionPlanning::runPyFunction(char pyFunctionName[], PyObject *pModule, double xm, double ym, double xr, double yr, double initHeading, char mapDirectory[], double resolution, double size)
{

	PyObject *func = PyObject_GetAttrString(pModule, pyFunctionName);
	PyObject *args;
	
	std::cout<<"Running function '"<<pyFunctionName<<"'..."<<std::endl;
	args = PyTuple_New(8);
	PyTuple_SetItem(args,0,PyFloat_FromDouble(xm));
	PyTuple_SetItem(args,1,PyFloat_FromDouble(ym));
	PyTuple_SetItem(args,2,PyFloat_FromDouble(xr));
	PyTuple_SetItem(args,3,PyFloat_FromDouble(yr));
	PyTuple_SetItem(args,4,PyFloat_FromDouble(initHeading));
	PyTuple_SetItem(args,5,PyUnicode_FromString(mapDirectory));
	PyTuple_SetItem(args,6,PyFloat_FromDouble(resolution));
	PyTuple_SetItem(args,7,PyFloat_FromDouble(size));
	
	if (!PyObject_CallObject(func,args))
		std::cout<<"... ERROR when calling function "<<pyFunctionName<<std::endl;
	else
		std::cout<<"... done"<<std::endl;

	return;
}

void MotionPlanning::sizePyArray( int &size, char pyVariableName[],  PyObject *pModule)
{

	PyObject *obj = PyObject_GetAttrString(pModule, pyVariableName);

	// Array Dimensions

	size = (int)PyArray_DIM(obj, 0);
	return;
}
void MotionPlanning::returnPyArrayDouble(int nDim, char pyVariableName[], double* &dVariable, PyObject *pModule)
{

	std::cout<<"Loading python variable named '"<<pyVariableName<<"'... ";	

	PyObject *obj = PyObject_GetAttrString(pModule, pyVariableName);

	// The pointer to the array data is accessed using PyArray_DATA()
	dVariable = (double*)PyArray_DATA(obj);
	
	std::cout<<"done"<<std::endl;


}

	
void MotionPlanning::returnPyArrayInt(int nDim, char pyVariableName[], int* &iVariable, PyObject *pModule)
{
	std::cout<<"Loading python variable named '"<<pyVariableName<<"'... ";	
	PyObject *obj = PyObject_GetAttrString(pModule, pyVariableName);


	// The pointer to the array data is accessed using PyArray_DATA()
	iVariable = (int*)PyArray_DATA(obj);
	std::cout<<"done"<<std::endl;
}

int MotionPlanning::shutDownPython(PyObject *pModule)
{	
    std::cout<<"Finalizing python interpreter...";    
    Py_DECREF(pModule);
    Py_Finalize();
	std::cout<<" done"<<std::endl;
	return 0;
}


