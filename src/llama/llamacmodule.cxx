/*
 * Copyright (c) 2021 The authors of Llama All rights reserved.
 * 
 * Modified from CModule of CoverTree
 * https://github.com/manzilzaheer/CoverTree
 * Copyright (c) 2017 Manzil Zaheer All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "numpy/arrayobject.h"
#include "llama.h"

#include <future>
#include <thread>

#include <iostream>
#include <iomanip>

static PyObject *LLAMAcError;

static PyObject *new_llamac(PyObject *self, PyObject *args)
{
  PyArrayObject *rows_in;
  PyArrayObject *cols_in;
  PyArrayObject *sims_in;
  long linkage;
  long num_rounds;
  PyArrayObject *thresholds_in;
  long cores;
  long max_num_parents;
  long max_num_neighbors;
  double lowest_value;

  if (!PyArg_ParseTuple(args, "O!O!O!llO!llld:new_llamac",
                        &PyArray_Type, &rows_in,
                        &PyArray_Type, &cols_in,
                        &PyArray_Type, &sims_in,
                        &linkage,
                        &num_rounds,
                        &PyArray_Type, &thresholds_in,
                        &cores,
                        &max_num_parents,
                        &max_num_neighbors,
                        &lowest_value))
    return NULL;

  long rowsInDim = PyArray_DIM(rows_in, 0);
  long numDimRow = PyArray_DIM(rows_in, 1);
  long colsInDim = PyArray_DIM(cols_in, 0);
  long numDimCol = PyArray_DIM(cols_in, 1);
  long simsInDim = PyArray_DIM(sims_in, 0);
  long numDimSims = PyArray_DIM(sims_in, 1);
  long threshInDim = PyArray_DIM(thresholds_in, 0);
  long numDimThresh = PyArray_DIM(thresholds_in, 1);
  long idx[2] = {0, 0};
  node_id_t *row = reinterpret_cast<node_id_t *>(PyArray_GetPtr(rows_in, idx));
  node_id_t *col = reinterpret_cast<node_id_t *>(PyArray_GetPtr(cols_in, idx));
  scalar *sims = reinterpret_cast<scalar *>(PyArray_GetPtr(sims_in, idx));
  scalar *thresholds = reinterpret_cast<scalar *>(PyArray_GetPtr(thresholds_in, idx));

  std::vector<node_id_t> row_v(row, row + rowsInDim);
  std::vector<node_id_t> col_v(col, col + colsInDim);
  std::vector<scalar> sims_v(sims, sims + simsInDim);
  LLAMA *d = LLAMA::from_graph(row_v, col_v, sims_v, linkage, num_rounds, thresholds, cores, max_num_parents, max_num_neighbors, (scalar)lowest_value);
  size_t int_ptr = reinterpret_cast<size_t>(d);
  return Py_BuildValue("k", int_ptr);
}

static PyObject *llamac_cluster(PyObject *self, PyObject *args)
{

  LLAMA *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:llamac_cluster", &int_ptr))
    return NULL;

  obj = reinterpret_cast<LLAMA *>(int_ptr);
  obj->cluster();

  Py_RETURN_NONE;
}

static PyObject *llamac_all_nodes_coo(PyObject *self, PyObject *args)
{

  LLAMA *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:llamac_all_nodes_coo", &int_ptr))
    return NULL;
  // std::cout << "converting to coo matrix...." << std::endl;
  auto startt = std::chrono::system_clock::now();

  obj = reinterpret_cast<LLAMA *>(int_ptr);
  obj->set_descendants();
  // std::cout << "number of edges... " << obj->descendants_r.size() << std::endl;

  node_id_t *results = new node_id_t[obj->descendants_r.size() * 2];
  size_t offset = 0;
  for (size_t i = 0; i < obj->descendants_r.size(); i++)
  {
    results[offset++] = obj->descendants_r[i];
    results[offset++] = obj->descendants_c[i];
  }
  auto endt = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = endt - startt;
  // std::cout << "converting to coo matrix.... done in  " << elapsed_seconds.count() << std::endl;

  long dims[2] = {obj->descendants_r.size(), 2};
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_UINT32, results);
  // std::cout << "out array created  " << std::endl;

  Py_INCREF(out_array);
  // std::cout << "Py_INCREF(out_array) done  " << std::endl;

  return out_array;
}

static PyObject *llamac_child_parent_coo(PyObject *self, PyObject *args)
{

  LLAMA *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:llamac_child_parent_coo", &int_ptr))
    return NULL;
  // std::cout << "converting to coo matrix...." << std::endl;
  auto startt = std::chrono::system_clock::now();

  obj = reinterpret_cast<LLAMA *>(int_ptr);
  obj->get_child_parent_edges();
  // std::cout << "number of edges... " << obj->descendants_r.size() << std::endl;

  node_id_t *results = new node_id_t[obj->children.size() * 2];
  size_t offset = 0;
  for (size_t i = 0; i < obj->children.size(); i++)
  {
    results[offset++] = obj->children[i];
    results[offset++] = obj->parents[i];
  }
  auto endt = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = endt - startt;
  // std::cout << "converting to coo matrix.... done in  " << elapsed_seconds.count() << std::endl;

  long dims[2] = {obj->children.size(), 2};
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_UINT32, results);
  // std::cout << "out array created  " << std::endl;

  Py_INCREF(out_array);
  // std::cout << "Py_INCREF(out_array) done  " << std::endl;

  return out_array;
}

static PyObject *llamac_get_round_coo(PyObject *self, PyObject *args)
{

  LLAMA *obj;
  size_t int_ptr;
  PyArrayObject *in_array;
  long r;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "kl:llamac_get_round_coo", &int_ptr, &r))
    return NULL;
  // std::cout << "getting descendants for round ... " << r << std::endl;
  auto startt = std::chrono::system_clock::now();

  obj = reinterpret_cast<LLAMA *>(int_ptr);
  obj->get_child_parent_edges();
  // std::cout << "number of edges... " << obj->descendants_r.size() << std::endl;
  std::unordered_set<node_id_t> *this_round_desc = obj->all_node2descendants[r];
  size_t round_size = obj->number_of_active_ids[r];
  size_t result_size = 0;
  for (size_t m = 0; m < round_size; m++)
  {
    result_size += this_round_desc[m].size();
  }
  node_id_t *results = new node_id_t[result_size * 2];
  size_t offset = 0;
  for (size_t m = 0; m < round_size; m++)
  {
    for (const auto c : this_round_desc[m])
    {
      results[offset++] = m;
      results[offset++] = c;
    }
  }

  auto endt = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = endt - startt;
  // std::cout << "converting to coo matrix.... done in  " << elapsed_seconds.count() << std::endl;

  long dims[2] = {result_size, 2};
  PyObject *out_array = PyArray_SimpleNewFromData(2, dims, NPY_UINT32, results);
  // std::cout << "out array created  " << std::endl;

  Py_INCREF(out_array);
  // std::cout << "Py_INCREF(out_array) done  " << std::endl;

  return out_array;
}

static PyObject *delete_llamac(PyObject *self, PyObject *args)
{
  LLAMA *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args, "k:delete_llamac", &int_ptr))
    return NULL;

  obj = reinterpret_cast<LLAMA *>(int_ptr);
  delete obj;

  return Py_BuildValue("k", int_ptr);
}

PyMODINIT_FUNC PyInit_llamac(void)
{
  PyObject *m;
  static PyMethodDef LLAMAcMethods[] = {
      {"new", new_llamac, METH_VARARGS, "Initialize."},
      {"delete", delete_llamac, METH_VARARGS, "Delete."},
      {"cluster", llamac_cluster, METH_VARARGS, "Run alg."},
      {"get_descendants", llamac_all_nodes_coo, METH_VARARGS, "get descendants coo."},
      {"get_child_parent_edges", llamac_child_parent_coo, METH_VARARGS, "get coo."},
      {"get_round", llamac_get_round_coo, METH_VARARGS, "get round descendants coo."},
      {NULL, NULL, 0, NULL}};
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                    "llamac",
                                    "Example module that creates an extension type.",
                                    -1,
                                    LLAMAcMethods};
  m = PyModule_Create(&mdef);
  if (m == NULL)
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  LLAMAcError = PyErr_NewException("llamac.error", NULL, NULL);
  Py_INCREF(LLAMAcError);
  PyModule_AddObject(m, "error", LLAMAcError);

  return m;
}

int main(int argc, char *argv[])
{
  /* Convert to wchar */
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL)
  {
    std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
    return 1;
  }

  /* Add a built-in module, before Py_Initialize */
  //PyImport_AppendInittab("llamac", PyInit_llamac);

  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_llamac();
}
