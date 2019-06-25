#include "MotionPlanning.hpp"

using namespace MotionPlanning_lib;

PyObject* MotionPlanning::initPython(char* pyName)
{

	std::cout<<"Loading python file named '"<<pyName<<"'..."<<std::endl;
    PyObject *pName, *pModule;


    Py_Initialize();
	import_array(); 

    pName = PyUnicode_DecodeFSDefault(pyName);
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
		std::cout<<"...correctly loaded"<<std::endl;
	}	
	else {
        PyErr_Print();
        fprintf(stderr, "...failed to load \n");
    }
	return pModule;
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


