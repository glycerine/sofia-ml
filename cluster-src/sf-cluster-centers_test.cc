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
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu   

#include <assert.h>
#include <iostream>
#include "sf-cluster-centers.h"

int main (int argc, char** argv) {
  SfClusterCenters cluster_centers_1(10);

  SfSparseVector x_1("0.0 1:1 2:2");
  SfSparseVector x_2("0.0 1:2 2:5 3:-1");

  cluster_centers_1.AddClusterCenterAt(x_1);
  cluster_centers_1.AddClusterCenterAt(x_2);

  assert(cluster_centers_1.Size() == 2);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(1) == 1);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(2) == 2);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(3) == 0);
  assert(cluster_centers_1.ClusterCenter(1).ValueOf(1) == 2);
  assert(cluster_centers_1.ClusterCenter(1).ValueOf(2) == 5);
  assert(cluster_centers_1.ClusterCenter(1).ValueOf(3) == -1);

  assert(cluster_centers_1.SqDistanceToCenterId(0, x_1) == 0);
  assert(cluster_centers_1.SqDistanceToCenterId(1, x_2) == 0);

  assert(cluster_centers_1.SqDistanceToCenterId(1, x_1) == 11);
  assert(cluster_centers_1.SqDistanceToCenterId(0, x_2) == 11);

  SfSparseVector* x_t =
    cluster_centers_1.MapVectorToCenters(x_1, SQUARED_DISTANCE, 0.0);
  assert(x_t->GetY() == 0.0);
  assert(x_t->FeatureAt(0) == 1);
  assert(x_t->ValueAt(0) == 0);
  assert(x_t->FeatureAt(1) == 2);
  assert(x_t->ValueAt(1) == 11);

  SfSparseVector* x_t_2 =
    cluster_centers_1.MapVectorToCenters(x_1, RBF_KERNEL, 1.0);
  assert(x_t_2->GetY() == 0.0);
  assert(x_t_2->FeatureAt(0) == 1);
  assert(x_t_2->ValueAt(0) == 1.0);
  assert(x_t_2->FeatureAt(1) == 2);
  assert(x_t_2->ValueAt(1) > 1.6e-5 && x_t_2->ValueAt(1) < 1.7e-5);

  cluster_centers_1.MutableClusterCenter(0)->AddVector(x_1, 1.0);
  assert(cluster_centers_1.Size() == 2);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(1) == 2);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(2) == 4);
  assert(cluster_centers_1.ClusterCenter(0).ValueOf(3) == 0);
  
  string expected_output_string("0 2 4 0 0 0 0 0 0 0\n"
				"0 2 5 -1 0 0 0 0 0 0\n");
  assert(cluster_centers_1.AsString() == expected_output_string);

  SfClusterCenters cluster_centers_2("sf-cluster-centers_test.dat");
  assert(cluster_centers_2.Size() == 2);
  assert(cluster_centers_2.ClusterCenter(0).ValueOf(1) == 1);
  assert(cluster_centers_2.ClusterCenter(0).ValueOf(2) == 2);
  assert(cluster_centers_2.ClusterCenter(0).ValueOf(3) == 3);
  assert(cluster_centers_2.ClusterCenter(0).ValueOf(4) == 4);
  assert(cluster_centers_2.ClusterCenter(1).ValueOf(1) == -1);
  assert(cluster_centers_2.ClusterCenter(1).ValueOf(2) == 0);
  assert(cluster_centers_2.ClusterCenter(1).ValueOf(3) == 1);
  assert(cluster_centers_2.ClusterCenter(1).ValueOf(4) == 0);
  assert(cluster_centers_2.ClusterCenter(1).ValueOf(5) == 1);
  
  std::cout << argv[0] << ": PASS" << std::endl;
}
