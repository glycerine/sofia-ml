
//================================================================================//
// Copyright 2009 Google Inc.                                                     //
//                                                                                //
// Licensed under the Apache License, Version 2.0 (the "License");                //
// you may not use this file except in compliance with the License.               //
// You may obtain a copy of the License at                                        //
//                                                                                //
//      http://www.apache.org/licenses/LICENSE-2.0                                //
//                                                                                //
// Unless required by applicable law or agreed to in writing, software            //
// distributed under the License is distributed on an "AS IS" BASIS,              //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       //
// See the License for the specific language governing permissions and            //
// limitations under the License.                                                 //
//================================================================================//
//
// sf-multi-label-vector.h
//
// Author: Mathieu Blondel
// mathieu@mblondel.org
//
// A subclass of SfWeightVector that knows how to handle multiple labels.

#ifndef SF_MULTI_LABEL_WEIGHT_VECTOR_H__
#define SF_MULTI_LABEL_WEIGHT_VECTOR_H__

#include <vector>

#include "sf-weight-vector.h"

using std::string;

class SfMultiLabelWeightVector : public SfWeightVector {
 public:
  SfMultiLabelWeightVector(int dimensionality, int num_labels);

  // Constructs a weight vector from a string, which is identical in format
  // to that produced by the AsString() member method.
  SfMultiLabelWeightVector(const string& weight_vector_string);

  // Frees the array of weights.
  virtual ~SfMultiLabelWeightVector();

  // Label selector
  void SelectLabel(int label_id) { selected_vector_ = label_id; }

  // Re-scales weight vector to scale of 1, and then outputs each weight in
  // order, space separated.
  virtual string AsString();

  // Computes inner product of <x_scale * x, w>
  virtual float InnerProduct(const SfSparseVector& x,
			     float x_scale = 1.0) const;

  // Computes inner product with a specified label vector.
  virtual float InnerProductLabel(const SfSparseVector& x,
           int label_id,
			     float x_scale = 1.0) const;

  // Computes the inner products with all (internal) weight vectors
  vector<float> InnerProductAll(const SfSparseVector& x,
                                float x_scale=1.0) const;

  // Computes inner product of <x_scale * (a - b), w>
  virtual float InnerProductOnDifference(const SfSparseVector& a,
				 const SfSparseVector& b,
				 float x_scale = 1.0) const;

  // w += x_scale * x
  virtual void AddVector(const SfSparseVector& x, float x_scale);

  // w *= scaling_factor
  virtual void ScaleBy(double scaling_factor);

  // Returns value of element w_index, taking internal scaling into account.
  virtual float ValueOf(int index) const;

  // Project this vector into the L1 ball of radius lambda.
  virtual void ProjectToL1Ball(float lambda);

  // Project this vector into the L1 ball of radius at most lambda, plus or
  // minus epsilon / 2.
  virtual void ProjectToL1Ball(float lambda, float epsilon);

  // Getters.
  virtual double GetSquaredNorm() const;
  virtual int GetDimensions() const;
  int NumLabels() const { return vectors_.size();}

 private:
  int num_labels_;
  int selected_vector_;
  vector<SfWeightVector>vectors_;
};

#endif  // SF_MULTI_LABEL_WEIGHT_VECTOR_H__
