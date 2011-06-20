//==========================================================================//
// Copyright 2009 Google Inc.                                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//      http://www.apache.org/licenses/LICENSE-2.0                          //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
//==========================================================================//
//
// sf-multi-labelweight-vector.cc
//
// Author: Mathieu Blondel
// mathieu@mblondel.org
//
// Implementation of sf-multi-labelweight-vector.h

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "sf-multi-label-weight-vector.h"

//-------------------------------------------------------------------------//
//---------------- SfMultiLabelWeightVector Public Methods ----------------//
//-------------------------------------------------------------------------//

SfMultiLabelWeightVector::SfMultiLabelWeightVector(int dimensionality,
                                                   int num_labels) :
  SfWeightVector(1), selected_vector_(0) {

  for (int i=0; i < num_labels; i++)
    vectors_.push_back(SfWeightVector(dimensionality));

}

SfMultiLabelWeightVector::SfMultiLabelWeightVector(const string& weight_vector_string) :
  SfWeightVector(1), selected_vector_(0) {
  std::stringstream multiple_weight_stream(weight_vector_string);

  string line_string;
  while (getline(multiple_weight_stream, line_string)) {
    vectors_.push_back(SfWeightVector(line_string));
  }
}

SfMultiLabelWeightVector::~SfMultiLabelWeightVector() {
  delete[] weights_;
}

string SfMultiLabelWeightVector::AsString() {
  std::stringstream out_string_stream;

  for (unsigned int i=0; i < vectors_.size(); ++i) {
    out_string_stream << vectors_[i].AsString();
    if (i != vectors_.size() - 1)
      out_string_stream << "\n";
  }

  return out_string_stream.str();
}

float SfMultiLabelWeightVector::InnerProduct(const SfSparseVector& x,
				    float x_scale) const {
  return vectors_[selected_vector_].InnerProduct(x, x_scale);
}

float SfMultiLabelWeightVector::InnerProductLabel(const SfSparseVector& x,
				    int label_id,
				    float x_scale) const {
  return vectors_[label_id].InnerProduct(x, x_scale);
}

void SfMultiLabelWeightVector::InnerProductAll(const SfSparseVector& x,
				    vector<float>* out,
				    float x_scale) const {
  for (unsigned int i=0; i < out->size(); i++)
    (*out)[i] = vectors_[i].InnerProduct(x, x_scale);
}

float SfMultiLabelWeightVector::InnerProductOnDifference(const SfSparseVector& a,
					       const SfSparseVector& b,
					       float x_scale) const {
  return vectors_[selected_vector_].InnerProductOnDifference(a, b, x_scale);
}

void SfMultiLabelWeightVector::AddVector(const SfSparseVector& x, float x_scale) {
  vectors_[selected_vector_].AddVector(x, x_scale);
}

void SfMultiLabelWeightVector::ScaleBy(double scaling_factor) {
  vectors_[selected_vector_].ScaleBy(scaling_factor);
}

float SfMultiLabelWeightVector::ValueOf(int index) const {
  return vectors_[selected_vector_].ValueOf(index);
}

void SfMultiLabelWeightVector::ProjectToL1Ball(float lambda, float epsilon) {
  vectors_[selected_vector_].ProjectToL1Ball(lambda, epsilon);
}


void SfMultiLabelWeightVector::ProjectToL1Ball(float lambda) {
  vectors_[selected_vector_].ProjectToL1Ball(lambda);
}

int SfMultiLabelWeightVector::GetDimensions() const {
  return vectors_[selected_vector_].GetDimensions();
}

double SfMultiLabelWeightVector::GetSquaredNorm() const {
  return vectors_[selected_vector_].GetSquaredNorm();
}

