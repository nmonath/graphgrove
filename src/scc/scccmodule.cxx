/*
 * Based on covertreecmodule.cxx from CoverTree Copyright (c) 2017 Manzil Zaheer All rights reserved.
 * Copyright (c) 2021 The authors of SG Tree All rights reserved.
 * Copyright (c) 2021 The authors of SCC All rights reserved.
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
#include "scc.h"

#include <future>
#include <thread>

#include <iostream>
#include <iomanip>

static PyObject *SCCcError;

static PyObject *init_sccc(PyObject *self, PyObject *args)
{
  PyArrayObject *thresholds;
  long cores; 
  long cc_alg;
  long par_min;
  long verbo_level;

  if (!PyArg_ParseTuple(args,"O!llll:init_sccc", &PyArray_Type, &thresholds, &cores, &cc_alg, &par_min, &verbo_level))
    return NULL;

  long threshInDim = PyArray_DIM(thresholds, 0);
  long idx[2] = {0, 0};
  scalar * thresh = reinterpret_cast< scalar * >( PyArray_GetPtr(thresholds, idx) );

  std::vector<scalar> threshs(thresh, thresh + threshInDim);
  SCC* d = SCC::init(threshs, (unsigned) cores, (unsigned) cc_alg, (size_t) par_min, (unsigned) verbo_level);
  size_t int_ptr = reinterpret_cast< size_t >(d);
  return Py_BuildValue("k", int_ptr);
}

static PyObject *sccc_fit(PyObject *self, PyObject *args) {

    SCC *obj;
    size_t int_ptr;

    /*  parse the input, from python int to c++ int */
    if (!PyArg_ParseTuple(args, "k:sccc_fit", &int_ptr))
        return NULL;

    obj = reinterpret_cast< SCC * >(int_ptr);
    obj->fit();

    Py_RETURN_NONE;
}

static PyObject *sccc_update(PyObject *self, PyObject *args) {

    SCC *obj;
    size_t int_ptr;

    /*  parse the input, from python int to c++ int */
    if (!PyArg_ParseTuple(args, "k:sccc_update", &int_ptr))
        return NULL;

    obj = reinterpret_cast< SCC * >(int_ptr);
    obj->fit_on_graph();

    Py_RETURN_NONE;
}


static PyObject *delete_sccc(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"k:delete_sccc", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  delete obj;

  return Py_BuildValue("k", int_ptr);
}

static PyObject *sccc_set_marking_strategy(PyObject *self, PyObject *args) {

    SCC *obj;
    size_t int_ptr;
    long strat;


    /*  parse the input, from python int to c++ int */
    if (!PyArg_ParseTuple(args, "kl:sccc_set_marking_strategy", &int_ptr, &strat))
        return NULL;

    obj = reinterpret_cast< SCC * >(int_ptr);
    obj->set_marking_strategy(strat);

    Py_RETURN_NONE;
}

static PyObject *sccc_insert_graph_mb(PyObject *self, PyObject *args) {

  SCC *obj;
  size_t int_ptr;
  PyArrayObject *rows_in;
  PyArrayObject *cols_in;
  PyArrayObject *sims_in;
  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!O!:sccc_insert_graph_mb", &int_ptr, &PyArray_Type, &rows_in, &PyArray_Type, &cols_in, &PyArray_Type, &sims_in))
    return NULL;

  long rowsInDim = PyArray_DIM(rows_in, 0);
  // long numDimRow = PyArray_DIM(rows_in, 1);
  // std::cout<< "rows "<<rowsInDim<<", "<<numDimRow<<std::endl;
  long colsInDim = PyArray_DIM(cols_in, 0);
  // long numDimCol = PyArray_DIM(cols_in, 1);
  // std::cout<< "cols "<<colsInDim<<", "<<numDimCol<<std::endl;
  long simsInDim = PyArray_DIM(sims_in, 0);
  // long numDimSims = PyArray_DIM(sims_in, 1);
  // std::cout<< "sims "<<simsInDim<<", "<<numDimSims<<std::endl;

  long idx[2] = {0, 0};
  node_id_t * row = reinterpret_cast< node_id_t * >( PyArray_GetPtr(rows_in, idx) );
  node_id_t * col = reinterpret_cast< node_id_t * >( PyArray_GetPtr(cols_in, idx) );
  scalar * sims = reinterpret_cast< scalar * >( PyArray_GetPtr(sims_in, idx) );
  // std::cout<< "finished reinterpret cast " <<std::endl;

  // for (size_t i=0; i < rowsInDim; i++) {
  //   std::cout << "r " << row[i] << " c " << col[i] << " s " << sims[i] << std::endl;
  // }

  std::vector<node_id_t> row_v(row, row + rowsInDim);
  // std::cout<< "row_v done " <<std::endl;
  std::vector<node_id_t> col_v(col, col + colsInDim);
  // std::cout<< "col_v done " <<std::endl;
  std::vector<scalar> sims_v(sims, sims + simsInDim);
  // std::cout<< "col_v done " <<std::endl;

  // std::cout << "CALLING INSERT GRAPH MB! " << std::endl;
 
  obj = reinterpret_cast< SCC * >(int_ptr);

  obj->insert_graph_mb(row_v, col_v, sims_v);
  
  // std::cout << "returning!" << std::endl;
  return Py_BuildValue("k", int_ptr);
}

static PyObject *sccc_add_graph_edges_mb(PyObject *self, PyObject *args) {

  SCC *obj;
  size_t int_ptr;
  PyArrayObject *rows_in;
  PyArrayObject *cols_in;
  PyArrayObject *sims_in;
  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!O!:sccc_add_graph_edges_mb", &int_ptr, &PyArray_Type, &rows_in, &PyArray_Type, &cols_in, &PyArray_Type, &sims_in))
    return NULL;

  long rowsInDim = PyArray_DIM(rows_in, 0);
  // long numDimRow = PyArray_DIM(rows_in, 1);
  // std::cout<< "rows "<<rowsInDim<<", "<<numDimRow<<std::endl;
  long colsInDim = PyArray_DIM(cols_in, 0);
  // long numDimCol = PyArray_DIM(cols_in, 1);
  // std::cout<< "cols "<<colsInDim<<", "<<numDimCol<<std::endl;
  long simsInDim = PyArray_DIM(sims_in, 0);
  // long numDimSims = PyArray_DIM(sims_in, 1);
  // std::cout<< "sims "<<simsInDim<<", "<<numDimSims<<std::endl;

  long idx[2] = {0, 0};
  node_id_t * row = reinterpret_cast< node_id_t * >( PyArray_GetPtr(rows_in, idx) );
  node_id_t * col = reinterpret_cast< node_id_t * >( PyArray_GetPtr(cols_in, idx) );
  scalar * sims = reinterpret_cast< scalar * >( PyArray_GetPtr(sims_in, idx) );
  // std::cout<< "finished reinterpret cast " <<std::endl;

  // for (size_t i=0; i < rowsInDim; i++) {
  //   std::cout << "r " << row[i] << " c " << col[i] << " s " << sims[i] << std::endl;
  // }

  std::vector<node_id_t> row_v(row, row + rowsInDim);
  // std::cout<< "row_v done " <<std::endl;
  std::vector<node_id_t> col_v(col, col + colsInDim);
  // std::cout<< "col_v done " <<std::endl;
  std::vector<scalar> sims_v(sims, sims + simsInDim);
  // std::cout<< "col_v done " <<std::endl;

  // std::cout << "CALLING INSERT GRAPH MB! " << std::endl;
 
  obj = reinterpret_cast< SCC * >(int_ptr);

  obj->add_graph_edges_mb(row_v, col_v, sims_v);
  
  // std::cout << "returning!" << std::endl;
  return Py_BuildValue("k", int_ptr);
}

static PyObject *sccc_insert_initial_batch(PyObject *self, PyObject *args) {

  // long k=2L;
  SCC *obj;
  size_t int_ptr;
  long num_points;
  PyArrayObject *rows_in;
  PyArrayObject *cols_in;
  PyArrayObject *sims_in;
  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nlO!O!O!:sccc_insert_initial_batch", &int_ptr, &num_points, &PyArray_Type, &rows_in, &PyArray_Type, &cols_in, &PyArray_Type, &sims_in))
    return NULL;

  long rowsInDim = PyArray_DIM(rows_in, 0);
  // long numDimRow = PyArray_DIM(rows_in, 1);
  // std::cout<< "rows "<<rowsInDim<<", "<<numDimRow<<std::endl;
  long colsInDim = PyArray_DIM(cols_in, 0);
  // long numDimCol = PyArray_DIM(cols_in, 1);
  // std::cout<< "cols "<<colsInDim<<", "<<numDimCol<<std::endl;
  long simsInDim = PyArray_DIM(sims_in, 0);
  // long numDimSims = PyArray_DIM(sims_in, 1);
  // std::cout<< "sims "<<simsInDim<<", "<<numDimSims<<std::endl;

  long idx[2] = {0, 0};
  node_id_t * row = reinterpret_cast< node_id_t * >( PyArray_GetPtr(rows_in, idx) );
  node_id_t * col = reinterpret_cast< node_id_t * >( PyArray_GetPtr(cols_in, idx) );
  scalar * sims = reinterpret_cast< scalar * >( PyArray_GetPtr(sims_in, idx) );
  // std::cout<< "finished reinterpret cast " <<std::endl;

  // for (size_t i=0; i < rowsInDim; i++) {
  //   std::cout << "r " << row[i] << " c " << col[i] << " s " << sims[i] << std::endl;
  // }

  std::vector<node_id_t> row_v(row, row + rowsInDim);
  // std::cout<< "row_v done " <<std::endl;
  std::vector<node_id_t> col_v(col, col + colsInDim);
  // std::cout<< "col_v done " <<std::endl;
  std::vector<scalar> sims_v(sims, sims + simsInDim);
  // std::cout<< "col_v done " <<std::endl;

  // std::cout << "CALLING INSERT GRAPH MB! " << std::endl;
 
  obj = reinterpret_cast< SCC * >(int_ptr);

  obj->insert_first_batch((size_t) num_points, row_v, col_v, sims_v);
  
  // std::cout << "returning!" << std::endl;
  return Py_BuildValue("k", int_ptr);
}


static PyObject *sccc_roots(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_roots", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);

  PyObject *o;
  PyObject *results = PyList_New(0);
  for (const auto& root : obj->levels[obj->levels.size()-1]->nodes) {
    if (!root->deleted) {
      o = PyLong_FromSize_t(reinterpret_cast<size_t>(root));
      PyList_Append(results, o);
      Py_DECREF(o);
    }
  }

  return Py_BuildValue("N", results);
}

static PyObject *sccc_knn_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_knn_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->knn_time);
}


static PyObject *sccc_graph_update_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_graph_update_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->get_graph_update_time());
}


static PyObject *sccc_overall_update_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_overall_update_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->get_overall_update_time());
}

static PyObject *sccc_best_neighbor_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_best_neighbor_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->get_best_neighbor_time());
}

static PyObject *sccc_cc_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_cc_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->get_cc_time());
}

static PyObject *sccc_update_time(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_update_time", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("f", obj->update_time);
}

static PyObject *sccc_total_number_marked(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_total_number_marked", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_total_number_marked());
}

static PyObject *sccc_max_number_marked(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_max_number_marked", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_max_number_marked());
}

static PyObject *sccc_sum_number_cc_iteration(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_sum_number_cc_iteration", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_sum_cc_iterations());
}

static PyObject *sccc_max_number_cc_iteration(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_max_number_cc_iteration", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_max_cc_iterations());
}

static PyObject *sccc_sum_number_cc_edges(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_sum_number_cc_edges", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_sum_cc_edges());
}

static PyObject *sccc_sum_number_cc_nodes(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_sum_number_cc_nodes", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_sum_cc_nodes());
}


static PyObject *sccc_total_number_nodes(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_total_number_nodes", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);
  return Py_BuildValue("l", obj->get_total_number_of_nodes());
}

static PyObject *sccc_node_children(PyObject *self, PyObject *args)
{
  SCC::TreeLevel::TreeNode *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_node_children", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel::TreeNode * >(int_ptr);

  PyObject *o;
  PyObject *results = PyList_New(0);
  for (const auto& child : obj->children) {
    // if (!child.second->deleted) {
      o = PyLong_FromSize_t(reinterpret_cast<size_t>(child.second));
      PyList_Append(results, o);
      Py_DECREF(o);
    // }
  }
  return Py_BuildValue("N", results);
}
static PyObject *sccc_node_pruned_children(PyObject *self, PyObject *args)
{
  SCC::TreeLevel::TreeNode *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_node_pruned_children", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel::TreeNode * >(int_ptr);

  PyObject *o;
  PyObject *results = PyList_New(0);
  for (const auto& child : obj->children) {
    // if (!child.second->deleted) {
      SCC::TreeLevel::TreeNode * kid = child.second->fastforward_levels();
      if (kid != NULL) {
        o = PyLong_FromSize_t(reinterpret_cast<size_t>(kid));
        PyList_Append(results, o);
        Py_DECREF(o);
      }
    // }
  }
  return Py_BuildValue("N", results);
}

static PyObject *sccc_node_descendants(PyObject *self, PyObject *args)
{
  SCC::TreeLevel::TreeNode *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_node_descendants", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel::TreeNode * >(int_ptr);
  std::set<node_id_t> desc = obj->get_descendants();
  long *indices = new long[desc.size()];
  size_t i = 0;
  for (node_id_t d: desc) {
    indices[i] = d;
    i++;
  }
  npy_intp dims[2] = {desc.size(), 1};
  PyObject *out_indices = PyArray_SimpleNewFromData(2, dims, NPY_LONG, indices);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_indices, NPY_ARRAY_OWNDATA);
  return Py_BuildValue("N", out_indices);
}


  static PyObject *sccc_level_nodes(PyObject *self, PyObject *args)
{
  SCC::TreeLevel *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_level_nodes", &int_ptr))
    return NULL;

  // std::cout << "level nodes.... before cast" << std::endl;
  obj = reinterpret_cast< SCC::TreeLevel * >(int_ptr);
  //  std::cout << "level nodes.... before cast" << std::endl;

  PyObject *o;
  PyObject *results = PyList_New(0);
  for (SCC::TreeLevel::TreeNode * n : obj->nodes) {
    if (!n->deleted) {
      o = PyLong_FromSize_t(reinterpret_cast<size_t>(n));
      PyList_Append(results, o);
      Py_DECREF(o);
    }
  }
  return Py_BuildValue("N", results);

}

  static PyObject *sccc_levels(PyObject *self, PyObject *args)
{
  SCC *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_levels", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC * >(int_ptr);

  PyObject *results = PyList_New(0);
  for (SCC::TreeLevel * l : obj->levels) {
    PyList_Append(results, PyLong_FromSize_t(reinterpret_cast<size_t>(l)));
  }

  return Py_BuildValue("N", results);
}

static PyObject *sccc_node_property(PyObject *self, PyObject *args)
{
  SCC::TreeLevel::TreeNode *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_node_property", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel::TreeNode * >(int_ptr);

  PyObject *o; 
  PyObject *results = PyDict_New();

  o = PyLong_FromSize_t(obj->this_id);
  PyDict_SetItemString(results, "uid", o);
  Py_DECREF(o);

  o = PyLong_FromLong(obj->level->height);
  PyDict_SetItemString(results, "height", o);
  Py_DECREF(o);

  o = PyBytes_FromStringAndSize(obj->ext_prop.c_str(), obj->ext_prop.length());
  PyDict_SetItemString(results, "others", o);
  Py_DECREF(o);

  return Py_BuildValue("N", results);
}

static PyObject *sccc_level_property(PyObject *self, PyObject *args)
{
  SCC::TreeLevel *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:sccc_level_property", &int_ptr))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel * >(int_ptr);

  PyObject *o; 
  PyObject *results = PyDict_New();

  o = PyLong_FromLong(obj->height);
  PyDict_SetItemString(results, "height", o);
  Py_DECREF(o);

  return Py_BuildValue("N", results);
}

static PyObject *sccc_node_save(PyObject *self, PyObject *args)
{
  SCC::TreeLevel::TreeNode *obj;
  size_t int_ptr;
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args, "ny#:sccc_node_save", &int_ptr, &buff, &len))
    return NULL;

  obj = reinterpret_cast< SCC::TreeLevel::TreeNode * >(int_ptr);
  obj->ext_prop.replace(0, -1, buff, len);

  Py_RETURN_NONE;
}


PyMODINIT_FUNC PyInit_sccc(void)
{
  PyObject *m;
  static PyMethodDef SCCcMethods[] = {
    {"init", init_sccc, METH_VARARGS, "Initialize a new SCC."},
    {"delete", delete_sccc, METH_VARARGS, "Delete SCC object."},
    {"insert_graph_mb", sccc_insert_graph_mb, METH_VARARGS, "Add edges to SCC."}, 
    {"fit_on_large_batch", sccc_insert_initial_batch, METH_VARARGS, "Initialize on a large batch of data at once."}, 
    {"add_graph_edges_mb", sccc_add_graph_edges_mb, METH_VARARGS, "Add edges, but dont update SCC yet. "}, 
    {"fit", sccc_fit, METH_VARARGS, "Run SCC."},
    {"update", sccc_update, METH_VARARGS, "Update SCC."},
    {"roots", sccc_roots, METH_VARARGS, "Tallest level of the tree."},
    {"node_children", sccc_node_children, METH_VARARGS, "Get children nodes of a node in lower level."},
    {"pruned_children", sccc_node_pruned_children, METH_VARARGS, "Get children nodes."},
    {"node_property", sccc_node_property, METH_VARARGS, "Get node property."},
    {"node_save", sccc_node_save, METH_VARARGS, "Save extra node property."},
    {"knn_time", sccc_knn_time, METH_VARARGS, "Get node property."},
    {"update_time", sccc_update_time, METH_VARARGS, "Time spent updating."},
    {"graph_update_time", sccc_graph_update_time, METH_VARARGS, "Time spent in graph contraction update. "},
    {"overall_update_time", sccc_overall_update_time, METH_VARARGS, "Overall time spent updating"},
    {"best_neighbor_time", sccc_best_neighbor_time, METH_VARARGS, "Time spent finding 1-nn graph."},
    {"cc_time", sccc_cc_time, METH_VARARGS, "Time spend finding connected components."},
    {"number_marked", sccc_total_number_marked, METH_VARARGS, "Number of nodes in total marked for update."},
    {"max_number_marked", sccc_max_number_marked, METH_VARARGS, "Max over levels of number of nodes marked for update."},
    {"max_number_cc_iterations", sccc_max_number_cc_iteration, METH_VARARGS, "Max number of cc iterations used in any level. "},
    {"sum_number_cc_iterations", sccc_sum_number_cc_iteration, METH_VARARGS, "Total number of cc iterations used in any level."},
    {"sum_number_cc_edges", sccc_sum_number_cc_edges, METH_VARARGS, "Total number of cc edges used"},
    {"sum_number_cc_nodes", sccc_sum_number_cc_nodes, METH_VARARGS, "Total number of cc nodes used."},
    {"total_number_nodes", sccc_total_number_nodes, METH_VARARGS, "Total number of nodes in structure. "},
    {"levels", sccc_levels, METH_VARARGS, "Get level objects."},
    {"level_nodes", sccc_level_nodes, METH_VARARGS, "Get (not deleted) nodes in level."},
    {"level_property", sccc_level_property, METH_VARARGS, "Get level property."},
    {"descendants", sccc_node_descendants, METH_VARARGS, "Get node descendants."},
    {"set_marking_strategy", sccc_set_marking_strategy, METH_VARARGS, "Set the way we will mark nodes."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "sccc",
                                 "SCC module.",
                                 -1,
                                 SCCcMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  SCCcError = PyErr_NewException("sccc.error", NULL, NULL);
  Py_INCREF(SCCcError);
  PyModule_AddObject(m, "error", SCCcError);

  return m;
}

int main(int argc, char *argv[])
{
  /* Convert to wchar */
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
     std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
     return 1;
  }
  
  /* Add a built-in module, before Py_Initialize */
  //PyImport_AppendInittab("sccc", PyInit_sccc);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_sccc();
}

