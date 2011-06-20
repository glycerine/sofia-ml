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
//
// Implementation of sf-sparse-vector.h

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "sf-sparse-vector.h"

//----------------------------------------------------------------//
//---------------- SfSparseVector Public Methods ----------------//
//----------------------------------------------------------------//
SfSparseVector::SfSparseVector(const char* in_string)
  : a_(0.0),
    squared_norm_(0.0),
    group_id_("") {
  NoBias();
  Init(in_string);
}

SfSparseVector::SfSparseVector(const char* in_string,
			       bool use_bias_term)
  : a_(0.0),
    squared_norm_(0.0),
    group_id_("") {
  if (use_bias_term) {
    SetBias();
  } else {
    NoBias();
  }
  Init(in_string);
}

SfSparseVector::SfSparseVector(const SfSparseVector& a,
				 const SfSparseVector& b,
				 float y)
  : a_(0.0),
    squared_norm_(0.0) {
  y_.push_back(y);
  group_id_ = a.GetGroupId();
  int a_i = 0;
  int b_i = 0;
  while (a_i < a.NumFeatures() || b_i < b.NumFeatures()) {
    // a has no features remaining.
    if (!(a_i < a.NumFeatures())) {
      PushPair(b.FeatureAt(b_i), 0.0 - b.ValueAt(b_i));
      ++b_i;
      continue;
    }
    // b has no features remaining.
    if (!(b_i < b.NumFeatures())) {
      PushPair(a.FeatureAt(a_i), a.ValueAt(a_i));
      ++a_i;
      continue;
    }
    // a's current feature is less than b's current feature.
    if (b.FeatureAt(b_i) < a.FeatureAt(a_i)) {
      PushPair(b.FeatureAt(b_i), 0.0 - b.ValueAt(b_i));
      ++b_i;
      continue;
    }
    // b's current feature is less than a's current feature.
    if (a.FeatureAt(a_i) < b.FeatureAt(b_i)) {
      PushPair(a.FeatureAt(a_i), a.ValueAt(a_i));
      ++a_i;
      continue;
    }
    // a_i and b_i are pointing to the same feature.
    PushPair(a.FeatureAt(a_i), a.ValueAt(a_i) - b.ValueAt(b_i));
    ++a_i;
    ++b_i;
  }
}

string SfSparseVector::AsString() const {
  std::stringstream out_stream;

  for (unsigned int i = 0; i < y_.size(); i++) {
    out_stream << y_[i];
    if (i != y_.size() - 1)
      out_stream << ",";
  }

  out_stream << " ";

  for (int i = 0; i < NumFeatures(); ++i) {
    out_stream << FeatureAt(i) << ":" << ValueAt(i) << " ";
  }
  if (!comment_.empty()) {
    out_stream << "#" << comment_;
  }
  return out_stream.str();
}

void SfSparseVector::PushPair(int id, float value) {
  if (id > 0 && NumFeatures() > 0 && id <= FeatureAt(NumFeatures() - 1) ) {
    std::cerr << id << " vs. " << FeatureAt(NumFeatures() - 1) << std::endl;
    DieFormat("Features not in ascending sorted order.");
  }

  FeatureValuePair feature_value_pair;
  feature_value_pair.id_ = id;
  feature_value_pair.value_ = value;
  features_.push_back(feature_value_pair);
  squared_norm_ += value * value;
}

//----------------------------------------------------------------//
//--------------- SfSparseVector Private Methods ----------------//
//----------------------------------------------------------------//

void SfSparseVector::DieFormat(const string& reason) {
  std::cerr << "Wrong format for input data:\n  " << reason << std::endl;
  exit(1);
}

void SfSparseVector::Init(const char* in_string) {
  int length = strlen(in_string);
  if (length == 0) DieFormat("Empty example string.");

  float y = 0;

  const char* position = in_string;
  const char* position_space = strchr(position, ' ');

  // Get class labels (comma-separated list).
  while (true) {
    if (!sscanf(position, "%f", &y))
      DieFormat("Class label must be real number.");

    y_.push_back(y);

    const char* position_comma = strchr(position, ',');

    if (position_comma == NULL or position_comma > position_space)
      // no further labels
      break;

    position = position_comma + 1;
  }

  // Parse the group id, if any.
  position = position_space + 1;

  if ((position[0] >= 'a' && position[0] <= 'z') ||
      (position[0] >= 'A' && position[0] <= 'Z')) {
    position = strchr(position, ':') + 1;
    const char* end = strchr(position, ' ');
    group_id_ = string(position, 0, end - position);
    position = end + 1;
  }

  // Get feature:value pairs.
  for ( ;
       (position < in_string + length
	&& position - 1 != NULL
	&& position[0] != '#');
       position = strchr(position, ' ') + 1) {

    // Consume multiple spaces, if needed.
    if (position[0] == ' ' || position[0] == '\n' ||
	position[0] == '\v' || position[0] == '\r') {
      continue;
    };

    // Parse the feature-value pair.
    int id = atoi(position);
    position = strchr(position, ':') + 1;
    float value = atof(position);
    PushPair(id, value);
  }

  // Parse comment, if any.
  position = strchr(in_string, '#');
  if (position != NULL) {
    comment_ = string(position + 1);
  }
}

void SfSparseVector::SetY(float new_y, unsigned int label_id) {
  if (y_.size() >= 1) {
    if (label_id >= y_.size()) {
      std::cerr << "label_id is larger than label set size in SetY()!" << std::endl;
      exit(1);
    }
    y_[label_id] = new_y;
  }
  else
    y_.push_back(new_y);
}

float SfSparseVector::GetY(unsigned int label_id) const {
    if (label_id >= y_.size()) {
      std::cerr << "label_id is larger than label set size in GetY()!" << std::endl;
      exit(1);
    }
  return y_[label_id];
}
