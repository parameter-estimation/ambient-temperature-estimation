
#include <Python.h>
#include <cstdio>
#include <optional>
#include "optimizer_api.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"


static optimizer_api optimizer;

//==================================================================================
//
//  copy_numpy_array_to_double_array
//
//==================================================================================
static int copy_numpy_array_to_double_array(PyArrayObject *inputArg, double * values)
{
    NpyIter *in_iter;
    NpyIter_IterNextFunc *in_iternext;
    in_iter = NpyIter_New(inputArg, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (in_iter == NULL) {
        return -1;
    }
    in_iternext = NpyIter_GetIterNext(in_iter, NULL);
    if (in_iternext == NULL) {
        NpyIter_Deallocate(in_iter);
        return -1;
    }
    auto **in_dataptr = (double **) NpyIter_GetDataPtrArray(in_iter);
    int length = PyArray_Size((PyObject *) inputArg);
    for (int ii = 0; ii < length; ii++) {
        values[ii] = **in_dataptr;
        in_iternext(in_iter);

        printf("Copied value: %f %f\n", values[ii], **in_dataptr);
    }
    NpyIter_Deallocate(in_iter);
    return 0;
}

//==================================================================================
//
//  copy_1d_vector_to_numpy_array
//
//==================================================================================
static int copy_1d_vector_to_numpy_array(PyArrayObject *outputArg, std::vector<double> values, int length)
{
    NpyIter *out_iter;
    NpyIter_IterNextFunc *out_iternext;
    out_iter = NpyIter_New(outputArg, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        return -1;
    }
    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (out_iternext == NULL) {
        NpyIter_Deallocate(out_iter);
        return -1;
    }
    double **out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);
    for (int ii = 0; ii < length; ii++) {
        **out_dataptr = values[ii];
        out_iternext(out_iter);
    }
    NpyIter_Deallocate(out_iter);
    return 0;
}

//==================================================================================
//
//  copy_2d_vector_to_numpy_array
//
//==================================================================================
static int copy_2d_vector_to_numpy_array(std::vector<std::vector<double>> inputVector, PyArrayObject *outputArray)
{
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    npy_intp multi_index[2];

    iter = NpyIter_New(outputArray, NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (iter == NULL) {
        return -1;
    }
    if (NpyIter_GetNDim(iter) != 2) {
        NpyIter_Deallocate(iter);
        PyErr_SetString(PyExc_ValueError, "Array must be 2-D");
        return -1;
    }


    if (NpyIter_GetIterSize(iter) != 0) {

        iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        NpyIter_GetMultiIndexFunc *get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);
        if (get_multi_index == NULL) {
            NpyIter_Deallocate(iter);
            return -1;
        }

        do {
            get_multi_index(iter, multi_index);
//            printf("multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "]\n", multi_index[0], multi_index[1]);

            auto **out_dataptr = (double **) NpyIter_GetDataPtrArray(iter);
            **out_dataptr = inputVector[multi_index[0]][multi_index[1]];
        } while (iternext(iter));
    }

    if (!NpyIter_Deallocate(iter)) {
        return -1;
    }
    return 0;
}


std::optional<std::string> parse_dict_string(PyObject *dict, const std::string& key)
{
    PyObject *valueObject = PyDict_GetItemString(dict, (const char *) key.c_str());
    if (valueObject == nullptr) {
        return std::nullopt;
    } else {
//        Py_DECREF(valueObject);
        return PyUnicode_AsUTF8(valueObject);
    }
}

bool parse_dict_bool(PyObject *dict, const std::string& key)
{
    PyObject *valueObject = PyDict_GetItemString(dict, (const char *) key.c_str());
    if (valueObject == nullptr) {
        return false;
    } else {
        Py_DECREF(valueObject);
        return PyObject_IsTrue(valueObject);
    }
}

std::vector<double> parse_dict_double_vect(PyObject *dict, const std::string& key)
{
    PyObject *valueObject = PyDict_GetItemString(dict, (const char *) key.c_str());
    if (!valueObject) {
        return {};
    } else {
        int n = PyObject_Length(valueObject);
        std::vector<double> x(n);
        for (int ii=0; ii<n; ii++) {
            PyObject *item;
            item = PyList_GetItem(valueObject, ii);
            x[ii] = PyFloat_AsDouble(item);
        }
        Py_DECREF(valueObject);
        return x;
    }
}

//==================================================================================
//
//  init
//
//==================================================================================
static char init_docstring[] = "init optimizer";
PyObject *init_wrapper(PyObject *self, PyObject *args)
{
    PyObject* dict;
    PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict);

    std::optional<std::string> model = parse_dict_string(dict, "model");
//
    if (!model) {
        printf("Settings must contain model\n");
        return nullptr;
    }

    bool verbose = parse_dict_bool(dict, "verbose");
    std::vector<double> initial_guesses = parse_dict_double_vect(dict, "initial_guesses");
    std::vector<double> fixed_parameters = parse_dict_double_vect(dict, "fixed_parameters");

    optimizer_settings_t settings = {
            .model = *model,
            .verbose = verbose,
            .initial_guesses = initial_guesses,
            .fixed_parameters = fixed_parameters
    };

//    optimizer_settings_t settings = {
//            .model = "train",
//            .verbose = true,
//            .initial_guesses = {},
//            .fixed_parameters = {}
//    };

    optimizer.init(settings);
    Py_RETURN_NONE;
}



//==================================================================================
//
//  feed
//
//==================================================================================
static char feed_docstring[] = "Feed data to optimizer";
static PyObject *feed_wrapper(PyObject *self, PyObject *args)
{
    double t;
    PyObject *float_list;
    if (!PyArg_ParseTuple(args, "dO", &t, &float_list)) {
        return nullptr;
    }

    int n = PyObject_Length(float_list);

    std::vector<double> x(n);
    for (int ii=0; ii<n; ii++) {
        PyObject *item;
        item = PyList_GetItem(float_list, ii);
        x[ii] = PyFloat_AsDouble(item);
    }

    optimizer.feed(t,&x);
    Py_RETURN_NONE;
}


//==================================================================================
//
//  fit
//
//==================================================================================
static char fit_docstring[] = "Fit the data given";
static PyObject *fit_wrapper(PyObject *self, PyObject *args)
{

    optimizer_result_t  result = optimizer.fit();

    PyObject *fitted_params_object = Py_BuildValue("{s:d,s:d,s:d,s:d}",
                                                   "h", result.fitted_params.h,
                                                   "q", result.fitted_params.q,
                                                   "T_dev_0", result.fitted_params.T_dev_0,
                                                   "T_amb_0", result.fitted_params.T_amb_0);

    return Py_BuildValue("{s:d,s:d,s:i,s:i,s:O}",
                         "is_valid", (double)result.is_valid,
                         "rmse", result.rmse,
                         "icount", result.icount,
                         "ifault", result.ifault,
                         "fitted_params", fitted_params_object);
}


//==================================================================================
//
//  generate
//
//==================================================================================
static char generate_docstring[] = "Generate data from given model params";
static PyObject *generate_wrapper(PyObject *self, PyObject *args)
{
    double h;
    double q;
    double T_dev_0;
    if (!PyArg_ParseTuple(args, "ddd", &h, &q, &T_dev_0)) {
        return nullptr;
    }

    PyArrayObject *timeOutput;
    PyArrayObject *xOutput;

    optimizer_model_params_t model_params = {
            .h = h,
            .q = q,
            .T_dev_0 = T_dev_0
    };

    modeled_state_timeseries_t data = optimizer.solve(&model_params);

    if (data.t.empty()) {
        timeOutput = (PyArrayObject *)PyArray_SimpleNew(0, {}, NPY_DOUBLE);
        xOutput = (PyArrayObject *)PyArray_SimpleNew(0, {}, NPY_DOUBLE);
    } else {
        int n = (int) data.t.size();
        npy_intp dims_1d[] = {n};
        npy_intp dims_2d[] = {n, (int) data.x[0].size()};
        timeOutput = (PyArrayObject *) PyArray_SimpleNew(1, dims_1d, NPY_DOUBLE);
        xOutput = (PyArrayObject *) PyArray_SimpleNew(2, dims_2d, NPY_DOUBLE);
        copy_1d_vector_to_numpy_array(timeOutput, data.t, n);
        copy_2d_vector_to_numpy_array(data.x, xOutput);
    }

    return Py_BuildValue("{s:i,s:O,s:O}",
                         "length", (int)data.t.size(),
                         "t", timeOutput,
                         "x", xOutput);


}




static char version_docstring[] = "Get module version";
static PyObject *version_wrapper(PyObject *self, PyObject *args)
{
    return Py_BuildValue("s", "1.0.0");
}

static PyMethodDef module_methods[] = {
        {"init", init_wrapper, METH_VARARGS, init_docstring},
        {"feed", feed_wrapper, METH_VARARGS, feed_docstring},
        {"fit", fit_wrapper, METH_VARARGS, fit_docstring},
        {"generate", generate_wrapper, METH_VARARGS, generate_docstring},
        {"version", version_wrapper, METH_VARARGS, version_docstring},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myPyModuleDef =
        {
                PyModuleDef_HEAD_INIT,
                "ambient_optimizer_python_api",
                "This module provides an interface for using the Ambient Temperature Optimizer library from Python.",
                -1,             /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                module_methods
        };

PyMODINIT_FUNC PyInit_ambient_optimizer_python_api(void) {
    import_array();
    return PyModule_Create(&myPyModuleDef);
}

