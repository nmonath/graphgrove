/*
 * Copyright (c) 2021 The authors of SG Tree All rights reserved.
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
#include "cover_tree.h"

#include <future>
#include <thread>

#include <iostream>
#include <iomanip>

static PyObject *CovertreecError;

static PyObject *new_covertreec(PyObject *self, PyObject *args)
{
  int trunc;
  long use_multi_core;
  PyArrayObject *in_array;

  if (!PyArg_ParseTuple(args,"O!il:new_covertreec", &PyArray_Type, &in_array, &trunc, &use_multi_core))
    return NULL;

  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp idx[2] = {0, 0};
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<matrixType> pointMatrix(fnp, numDims, numPoints);

  CoverTree* cTree = CoverTree::from_matrix(pointMatrix, trunc, use_multi_core);
  size_t int_ptr = reinterpret_cast< size_t >(cTree);
  size_t node_ptr = reinterpret_cast< size_t >(cTree->get_root());

  return Py_BuildValue("nn", int_ptr, node_ptr);
}

static PyObject *delete_covertreec(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"n:delete_covertreec", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  delete obj;

  return Py_BuildValue("n", int_ptr);
}


static PyObject *covertreec_insert(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  int uid = -1L;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!p:covertreec_insert", &int_ptr, &PyArray_Type, &in_array, &uid))
    return NULL;

  // int d = PyArray_NDIM(in_array);
  npy_intp idx[1] = {0};
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  if (uid >= 0)
    obj->insert(value, uid);
  else
    obj->insert(value, obj->get_tree_size());

  Py_RETURN_NONE;
}

static PyObject *covertreec_batchinsert(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  long use_multi_core;
  PyArrayObject *in_array;
  PyArrayObject *uid_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!O!l:covertreec_batchinsert", &int_ptr, &PyArray_Type, &in_array, &PyArray_Type, &uid_array, &use_multi_core))
    return NULL;

  // int d = PyArray_NDIM(in_array);
  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  // std::cout<<numPoints<<", "<<numDims<<std::endl;
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<matrixType> insPts(fnp, numDims, numPoints);
  
  npy_intp idx2[1] = {0};
  npy_intp numPoints2 = PyArray_DIM(uid_array, 0);
  if (numPoints != numPoints2)
	  std::cerr << "Points and UID size do not match!!!" << std::endl;
  long * unp = reinterpret_cast< long * >( PyArray_GetPtr(uid_array, idx2) );
  // Eigen::Map<pointType> insUID(unp, numPoints2);

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  // std::cout << "covertreec_batchinsert use_multi_core " << use_multi_core << std::endl;
  if(use_multi_core!=0)
  {
      utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
          if(!obj->insert(insPts.col(i), unp[i]))
                    std::cout << "Insert failed!!! " << unp[i] << std::endl;
      }, use_multi_core);
  }
  else
  {
      for(npy_intp i = 0; i < numPoints; ++i) {
		  if(!obj->insert(insPts.col(i), unp[i]))
                    std::cout << "Insert failed!!! " << unp[i] << std::endl;
	  }
  }

  Py_RETURN_NONE;
}

static PyObject *covertreec_remove(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!:covertreec_remove", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  npy_intp idx[1] = {0};
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  bool val = obj->remove(value);

  if (val)
  	Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *covertreec_nn(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  long cores;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value; 

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!lp:covertreec_nn", &int_ptr, &PyArray_Type, &in_array, &cores, &return_points))
    return NULL;

  unsigned use_multi_core = (unsigned) cores;
  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<matrixType> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  #ifdef PRINTVER
  CoverTree::Node::dist_count.clear();
  #endif

  npy_intp dims[1] = {numPoints};
  PyObject *out_indices = PyArray_SimpleNew(1, dims, NPY_LONG);
  PyObject *out_dist = PyArray_SimpleNew(1, dims, MY_NPY_FLOAT);

  long *indices = reinterpret_cast<long *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_indices), idx) );
  scalar *dist = reinterpret_cast<scalar *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_dist), idx) );

  // scalar *dist = new scalar[numPoints];
  // long *indices = new long[numPoints];
  scalar *results = nullptr;
  if(return_points!=0)
  {
    results = new scalar[numDims*numPoints];

    npy_intp three_idx[3] = {0, 0, 0};
    npy_intp odims[2] = {numPoints, numDims};
    PyObject *out_array = PyArray_SimpleNew(2, odims, MY_NPY_FLOAT);
    scalar *results = reinterpret_cast<scalar *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_array), three_idx) );

    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::pair<CoverTree::Node*, scalar> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
            scalar *data = ct_nn.first->_p.data();
            offset = i*numDims;
            for(npy_intp j=0; j<numDims; ++j)
                results[offset++] = data[j];
        }, use_multi_core);
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::pair<CoverTree::Node*, scalar> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
            scalar *data = ct_nn.first->_p.data();
            offset = i*numDims;
            for(npy_intp j=0; j<numDims; ++j)
                results[offset++] = data[j];
        }
    }    
    return_value = Py_BuildValue("NNN", out_indices, out_dist, out_array);
  }
  else
  {
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::pair<CoverTree::Node*, scalar> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
        }, use_multi_core);
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::pair<CoverTree::Node*, scalar> ct_nn = obj->NearestNeighbour(queryPts.col(i));
            npy_intp offset = i;
            dist[offset] = ct_nn.second;
            indices[offset] = ct_nn.first->UID;
        }
    }
    return_value = Py_BuildValue("NN", out_indices, out_dist);
  }

  #ifdef PRINTVER
  unsigned long tot_comp = 0;
  for(auto const& qc : CoverTree::Node::dist_count)
  {
    std::cout << "Average number of distance computations at level: " << qc.first << " = " << 1.0 * (qc.second.load())/numPoints << std::endl;
    tot_comp += qc.second.load();
  }
  std::cout << "Average number of distance computations: " << 1.0*tot_comp/numPoints << std::endl;
  #endif

  return return_value;
}

static PyObject *covertreec_knn(PyObject *self, PyObject *args) {

  long k=2L;
  CoverTree *obj;
  size_t int_ptr;
  long cores;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!llp:covertreec_knn", &int_ptr, &PyArray_Type, &in_array, &k, &cores, &return_points))
    return NULL;

  unsigned use_multi_core = (unsigned) cores;
  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<matrixType> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  #ifdef PRINTVER
  CoverTree::Node::dist_count.clear();
  #endif

  // std::cout << "use_multi_core (covertreec_knn) = " << use_multi_core << std::endl;

  // long *indices = new long[k*numPoints];
  // scalar *dist = new scalar[k*numPoints];
  npy_intp dims[2] = {numPoints, k};
  PyObject *out_indices = PyArray_SimpleNew(2, dims, NPY_LONG);
  PyObject *out_dist = PyArray_SimpleNew(2, dims, MY_NPY_FLOAT);

  long *indices = reinterpret_cast<long *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_indices), idx) );
  scalar *dist = reinterpret_cast<scalar *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_dist), idx) );

  scalar *results = nullptr;
  if(return_points!=0)
  {
    // results = new scalar[k*numDims*numPoints];
    npy_intp three_idx[3] = {0, 0, 0};
    npy_intp odims[3] = {numPoints, k, numDims};
    PyObject *out_array = PyArray_SimpleNew(3, odims, MY_NPY_FLOAT);
    scalar *results = reinterpret_cast<scalar *>(
      PyArray_GetPtr(reinterpret_cast<PyArrayObject *>(out_array), three_idx) );

    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset] = ct_nn[t].second;
                scalar *data = ct_nn[t].first->_p.data();
                npy_intp inner_offset = (offset++)*numDims;
                for(long j=0; j<numDims; ++j)
                    results[inner_offset++] = data[j];
            }
        }, use_multi_core);
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            utils::progressbar(i, numPoints);
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset] = ct_nn[t].second;
                scalar *data = ct_nn[t].first->_p.data();
                npy_intp inner_offset = (offset++)*numDims;
                for(long j=0; j<numDims; ++j)
                    results[inner_offset++] = data[j];
            }
        }
    }
    return_value = Py_BuildValue("NNN", out_indices, out_dist, out_array);
  }
  else
  {
    if(use_multi_core!=0)
    {
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset++] = ct_nn[t].second;
            }
        }, use_multi_core);
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            utils::progressbar(i, numPoints);
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
            npy_intp offset = k*i;
            for(long t=0; t<k; ++t)
            {
                indices[offset] = ct_nn[t].first->UID;
                dist[offset++] = ct_nn[t].second;
            }
        }
    }
    return_value = Py_BuildValue("NN", out_indices, out_dist);
  }

  #ifdef PRINTVER
  unsigned long tot_comp = 0;
  for(auto const& qc : CoverTree::Node::dist_count)
  {
    std::cout << "Average number of distance computations at level: " << qc.first << " = " << 1.0 * (qc.second.load())/numPoints << std::endl;
    tot_comp += qc.second.load();
  }
  std::cout << "Average number of distance computations: " << 1.0*tot_comp/numPoints << std::endl;
  #endif

  return return_value;
}

static PyObject *covertreec_range(PyObject *self, PyObject *args) {

  scalar r=0.0;
  CoverTree *obj;
  size_t int_ptr;
  unsigned cores;
  int return_points;
  PyArrayObject *in_array;
  PyObject *return_value;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nO!"PYTHON_FLOAT_CHAR"lp:covertreec_range", &int_ptr, &PyArray_Type, &in_array, &r, &cores, &return_points))
    return NULL;

  unsigned use_multi_core = (unsigned) cores;
  npy_intp idx[2] = {0,0};
  npy_intp numPoints = PyArray_DIM(in_array, 0);
  npy_intp numDims = PyArray_DIM(in_array, 1);
  // std::cout<<numPoints<<", "<<numDims<<" :: "<<r<<std::endl;
  scalar * fnp = reinterpret_cast< scalar * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<matrixType> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  #ifdef PRINTVER
  CoverTree::Node::dist_count.clear();
  #endif

  PyObject *indices = PyList_New(numPoints);
  PyObject *dist = PyList_New(numPoints);
  if(return_points!=0)
  {
    PyObject *results = PyList_New(numPoints);
    if(use_multi_core!=0)
    {
        PyObject **array_indices = new PyObject*[numPoints];
        PyObject **array_dist = new PyObject*[numPoints];
        PyObject **array_point = new PyObject*[numPoints];
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            scalar *point_dist = new scalar[num_neighbours];
            scalar *point_point = new scalar[num_neighbours*numDims];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
                scalar *data = ct_nn[t].first->_p.data();
                npy_intp offset = t*numDims;
                for(long j=0; j<numDims; ++j)
                    point_point[offset++] = data[j];
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, MY_NPY_FLOAT, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            npy_intp odims[2] = {(npy_intp)num_neighbours, numDims};
            PyObject *neighbour_point = PyArray_SimpleNewFromData(2, odims, MY_NPY_FLOAT, point_point);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_point, NPY_ARRAY_OWNDATA);
            array_indices[i] = neighbour_indices;
            array_dist[i] = neighbour_dist;
            array_point[i] = neighbour_point;
        }, use_multi_core);
        for(npy_intp i = 0; i < numPoints; ++i) {
            PyList_SET_ITEM(indices, i, array_indices[i]);
            PyList_SET_ITEM(dist, i, array_dist[i]);
            PyList_SET_ITEM(results, i, array_point[i]);
        }
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            scalar *point_dist = new scalar[num_neighbours];
            scalar *point_point = new scalar[num_neighbours*numDims];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
                scalar *data = ct_nn[t].first->_p.data();
                npy_intp offset = t*numDims;
                for(long j=0; j<numDims; ++j)
                    point_point[offset++] = data[j];
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, MY_NPY_FLOAT, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            npy_intp odims[2] = {(npy_intp)num_neighbours, numDims};
            PyObject *neighbour_point = PyArray_SimpleNewFromData(2, odims, MY_NPY_FLOAT, point_point);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_point, NPY_ARRAY_OWNDATA);
            PyList_SET_ITEM(indices, i, neighbour_indices);
            PyList_SET_ITEM(dist, i, neighbour_dist);
            PyList_SET_ITEM(results, i, neighbour_point);
        }
    }
    return_value = Py_BuildValue("NNN", indices, dist, results);
  }
  else
  {
    if(use_multi_core!=0)
    {
        PyObject **array_indices = new PyObject*[numPoints];
        PyObject **array_dist = new PyObject*[numPoints];
        utils::parallel_for_progressbar(0, numPoints, [&](npy_intp i)->void{
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            scalar *point_dist = new scalar[num_neighbours];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, MY_NPY_FLOAT, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            array_indices[i] = neighbour_indices;
            array_dist[i] = neighbour_dist;
        }, use_multi_core);
        for(npy_intp i = 0; i < numPoints; ++i) {
            PyList_SET_ITEM(indices, i, array_indices[i]);
            PyList_SET_ITEM(dist, i, array_dist[i]);
        }
    }
    else
    {
        for(npy_intp i = 0; i < numPoints; ++i) {
            std::vector<std::pair<CoverTree::Node*, scalar>> ct_nn = obj->rangeNeighbours(queryPts.col(i), r);
            size_t num_neighbours = ct_nn.size();
            long *point_indices = new long[num_neighbours];
            scalar *point_dist = new scalar[num_neighbours];
            for(size_t t=0; t<num_neighbours; ++t)
            {
                point_indices[t] = ct_nn[t].first->UID;
                point_dist[t] = ct_nn[t].second;
            }
            npy_intp dims[1] = {(npy_intp)num_neighbours};
            PyObject *neighbour_indices = PyArray_SimpleNewFromData(1, dims, NPY_LONG, point_indices);
            PyObject *neighbour_dist = PyArray_SimpleNewFromData(1, dims, MY_NPY_FLOAT, point_dist);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_indices, NPY_ARRAY_OWNDATA);
            PyArray_ENABLEFLAGS((PyArrayObject *)neighbour_dist, NPY_ARRAY_OWNDATA);
            PyList_SET_ITEM(indices, i, neighbour_indices);
            PyList_SET_ITEM(dist, i, neighbour_dist);
        }
    }
    return_value = Py_BuildValue("NN", indices, dist);
  }

  #ifdef PRINTVER
  unsigned long tot_comp = 0;
  for(auto const& qc : CoverTree::Node::dist_count)
  {
    std::cout << "Average number of distance computations at level: " << qc.first << " = " << 1.0 * (qc.second.load())/numPoints << std::endl;
    tot_comp += qc.second.load();
  }
  std::cout << "Average number of distance computations: " << 1.0*tot_comp/numPoints << std::endl;
  #endif

  return return_value;
}

static PyObject *covertreec_serialize(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args, "n:covertreec_serialize", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  char* buff = obj->serialize();
  size_t len = obj->msg_size();

  return Py_BuildValue("y#", buff, len);
}

static PyObject *covertreec_deserialize(PyObject *self, PyObject *args)
{
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args, "y#:covertreec_deserialize", &buff, &len))
    return NULL;

  CoverTree* cTree = new CoverTree();
  cTree->deserialize(buff);
  size_t int_ptr = reinterpret_cast< size_t >(cTree);
  size_t node_ptr = reinterpret_cast< size_t >(cTree->get_root());

  return Py_BuildValue("nn", int_ptr, node_ptr);
}


static PyObject *covertreec_display(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_display", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  std::cout << *obj;

  Py_RETURN_NONE;
}


static PyObject *covertreec_stats(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_stats", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  obj->print_stats();

  Py_RETURN_NONE;
}

static PyObject *covertreec_size(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_size", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  size_t size = obj->get_tree_size();

  return Py_BuildValue("n", size);
}

static PyObject *covertreec_get_root(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_get_root", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  size_t node_ptr = reinterpret_cast< size_t >(obj->get_root());

  return Py_BuildValue("n", node_ptr);
}

static PyObject *covertreec_spreadout(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;
  unsigned K;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "nI:covertreec_spreadout", &int_ptr, &K))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  std::vector<unsigned> results = obj->getBestInitialPoints(K);
  unsigned* new_ptr = new unsigned[results.size()];
  for (size_t i = 0; i < results.size(); ++i)
    new_ptr[i] = results[i];

  npy_intp numPoints = results.size();
  // std::cout<<numPoints<<", "<<numDims<<std::endl;
  npy_intp dims[1] = {numPoints};
  PyObject *out_array = PyArray_SimpleNewFromData(1, dims, NPY_UINT, new_ptr);
  PyArray_ENABLEFLAGS((PyArrayObject *)out_array, NPY_ARRAY_OWNDATA);

  return Py_BuildValue("N", out_array);
}

static PyObject *covertreec_test_covering(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_test_covering", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  if(obj->check_covering())
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *covertreec_test_nesting(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_test_nesting", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  if(obj->check_nesting())
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *covertreec_node_children(PyObject *self, PyObject *args)
{
  CoverTree::Node *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_node_children", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree::Node * >(int_ptr);

  PyObject *o;
  PyObject *results = PyList_New(0);
  for (const auto& child : *obj)
  {
    o = PyLong_FromSize_t(reinterpret_cast<size_t>(child));
    PyList_Append(results, o);
    Py_DECREF(o);
  }

  return Py_BuildValue("N", results);
}

static PyObject *covertreec_node_property(PyObject *self, PyObject *args)
{
  CoverTree::Node *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "n:covertreec_node_property", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree::Node * >(int_ptr);

  npy_intp dims[1] = {obj->_p.rows()};
  PyObject *point = PyArray_SimpleNewFromData(1, dims, MY_NPY_FLOAT, obj->_p.data());
  PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(point), NPY_ARRAY_OWNDATA);
  PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(point), NPY_ARRAY_WRITEABLE);

  PyObject *o;  // generic object
  PyObject *results = PyDict_New();

  o = PyLong_FromSize_t(obj->UID);
  PyDict_SetItemString(results, "uid", o);
  Py_DECREF(o);

  o = PyLong_FromLong(obj->level);
  PyDict_SetItemString(results, "level", o);
  Py_DECREF(o);

  PyDict_SetItemString(results, "point", point);
  Py_DECREF(point);

  o = PyFloat_FromDouble(obj->maxdistUB);
  PyDict_SetItemString(results, "maxdistUB", o);
  Py_DECREF(o);

  o = PyBytes_FromStringAndSize(obj->ext_prop.c_str(), obj->ext_prop.length());
  PyDict_SetItemString(results, "others", o);
  Py_DECREF(o);

  return Py_BuildValue("N", results);
}

static PyObject *covertreec_node_save(PyObject *self, PyObject *args)
{
  CoverTree::Node *obj;
  size_t int_ptr;
  char* buff;
  size_t len;
  if (!PyArg_ParseTuple(args, "ny#:covertreec_node_save", &int_ptr, &buff, &len))
    return NULL;

  obj = reinterpret_cast< CoverTree::Node * >(int_ptr);
  obj->ext_prop.replace(0, -1, buff, len);

  Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit_covertreec(void)
{
  PyObject *m;
  static PyMethodDef CovertreecMethods[] = {
    {"new", new_covertreec, METH_VARARGS, "Initialize a new Cover Tree."},
    {"delete", delete_covertreec, METH_VARARGS, "Delete the Cover Tree."},
    {"insert", covertreec_insert, METH_VARARGS, "Insert a point to the Cover Tree."},
  	{"batchinsert", covertreec_batchinsert, METH_VARARGS, "Insert a batch of point to the Cover Tree."},
    {"remove", covertreec_remove, METH_VARARGS, "Remove a point from the Cover Tree."},
    {"NearestNeighbour", covertreec_nn, METH_VARARGS, "Find the nearest neighbour."},
    {"kNearestNeighbours", covertreec_knn, METH_VARARGS, "Find the k nearest neighbours."},
    {"RangeSearch", covertreec_range, METH_VARARGS, "Find all the neighbours in range."},
    {"serialize", covertreec_serialize, METH_VARARGS, "Serialize the current Cover Tree."},
    {"deserialize", covertreec_deserialize, METH_VARARGS, "Construct a Cover Tree from deserializing."},
    {"display", covertreec_display, METH_VARARGS, "Display the Cover Tree."},
    {"stats", covertreec_stats, METH_VARARGS, "Print statistics of the Cover Tree."},
    {"size", covertreec_size, METH_VARARGS, "Return number of points in the Cover Tree."},
    {"spreadout", covertreec_spreadout, METH_VARARGS, "Find well spreadout k points."},
    {"test_covering", covertreec_test_covering, METH_VARARGS, "Check if covering property is satisfied."},
    {"test_nesting", covertreec_test_nesting, METH_VARARGS, "Check if nesting property is satisfied."},
    {"node_children", covertreec_node_children, METH_VARARGS, "Get children nodes."},
    {"node_property", covertreec_node_property, METH_VARARGS, "Get node property."},
    {"get_root", covertreec_get_root, METH_VARARGS, "Get root node."},
    {"node_save", covertreec_node_save, METH_VARARGS, "Save extra node property."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "covertreec",
                                 "Example module that creates an extension type.",
                                 -1,
                                 CovertreecMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  CovertreecError = PyErr_NewException("covertreec.error", NULL, NULL);
  Py_INCREF(CovertreecError);
  PyModule_AddObject(m, "error", CovertreecError);

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
  //PyImport_AppendInittab("covertreec", PyInit_covertreec);

  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_covertreec();
}
