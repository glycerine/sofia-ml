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
#include <assert.h>
#include <iostream>
#include "sf-multi-label-weight-vector.h"

int main (int argc, char** argv) {
  SfMultiLabelWeightVector w_5(5, 2);
  assert(w_5.GetDimensions() == 5);
  assert(w_5.GetSquaredNorm() == 0);
  assert(w_5.ValueOf(4) == 0.0);

  char x_string [100] = "1.0 0:1 1:1.0 2:2.0 4:3.0";
  SfSparseVector x(x_string);

  assert(w_5.InnerProduct(x) == 0.0);

  w_5.AddVector(x, 2.0);
  assert(w_5.ValueOf(4) == 6.0);
  assert(w_5.ValueOf(6) == 0.0);
  assert(w_5.GetSquaredNorm() == 60.0);

  assert(w_5.InnerProduct(x) == 30.0);

  w_5.ScaleBy(0.5);
  assert(w_5.GetSquaredNorm() == 15.0);
  assert(w_5.ValueOf(4) == 3.0);

  w_5.SelectLabel(1);
  assert(w_5.GetDimensions() == 5);
  assert(w_5.GetSquaredNorm() == 0);
  assert(w_5.ValueOf(4) == 0.0);

  char x_string2 [100] = "-1.0 2:3.0 4:1.0";
  SfSparseVector x2(x_string2);

  w_5.AddVector(x2, 2.0);
  assert(w_5.ValueOf(4) == 2.0);
  assert(w_5.ValueOf(6) == 0.0);
  assert(w_5.GetSquaredNorm() == 40.0);

  assert(w_5.AsString() == "1 1 2 0 3\n0 0 6 0 2");

  SfMultiLabelWeightVector w_copy(w_5.AsString());
  assert(w_copy.GetDimensions() == 5);
  assert(w_copy.NumLabels() == 2);

  assert(w_copy.GetSquaredNorm() == 15.0);
  assert(w_copy.ValueOf(4) == 3.0);

  w_copy.SelectLabel(1);
  assert(w_copy.GetDimensions() == 5);
  assert(w_copy.ValueOf(4) == 2.0);
  assert(w_copy.GetSquaredNorm() == 40.0);

  vector<float> inner_all(w_copy.NumLabels());
  w_copy.InnerProductAll(x, &inner_all);
  assert(inner_all.size() == 2);
  assert(inner_all[0] == w_copy.InnerProductLabel(x, 0));
  assert(inner_all[1] == w_copy.InnerProductLabel(x, 1));

  std::cout << argv[0] << ": PASS" << std::endl;
}
