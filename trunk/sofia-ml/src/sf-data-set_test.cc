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
#include "sf-data-set.h"

int main (int argc, char** argv) {
  // Test vector string with comment.
  SfDataSet data_set(string("sf-data-set_test.dat"), 5);
  assert(data_set.NumExamples() == 2);
  assert(data_set.VectorAt(0).GetY() == 1);
  assert(data_set.VectorAt(1).GetY() == -1);

  std::cout << argv[0] << ": PASS" << std::endl;
}
