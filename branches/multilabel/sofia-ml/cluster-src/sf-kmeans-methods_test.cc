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
#include <cstdlib>
#include <iostream>
#include "sf-kmeans-methods.h"

int main (int argc, char** argv) {
  srand(101);  // Use a fixed seed for deterministic testing.
  SfDataSet data_set("sf-kmeans-methods_test.dat", 1, false);
  
  SfClusterCenters cluster_centers_1(10);
  sofia_cluster::InitializeWithKRandomCenters(9, data_set, &cluster_centers_1);
  assert(cluster_centers_1.Size() == 9);

  SfClusterCenters* cluster_centers_2 = new SfClusterCenters(10, 1);
  sofia_cluster::OneBatchKmeansOptimization(data_set, cluster_centers_2);
  assert(cluster_centers_2->ClusterCenter(0).ValueOf(1) < 0.67 &&
	 cluster_centers_2->ClusterCenter(0).ValueOf(1) > 0.66);
  assert(cluster_centers_2->ClusterCenter(0).ValueOf(2) < 0.34 &&
	 cluster_centers_2->ClusterCenter(0).ValueOf(2) > 0.33);
  assert(cluster_centers_2->ClusterCenter(0).ValueOf(3) < 0.34 &&
	 cluster_centers_2->ClusterCenter(0).ValueOf(3) > 0.33);
  
  SfClusterCenters* cluster_centers_3 = new SfClusterCenters(10);
  srand(105);  // Use a fixed seed for deterministic testing.
  sofia_cluster::InitializeWithKRandomCenters(3, data_set, cluster_centers_3);
  sofia_cluster::BatchKmeans(5, data_set, cluster_centers_3);
  assert(cluster_centers_3->ClusterCenter(0).ValueOf(1) == 1.0);
  assert(cluster_centers_3->ClusterCenter(0).ValueOf(2) == 0.0);
  assert(cluster_centers_3->ClusterCenter(0).ValueOf(3) == 1.0);
  assert(cluster_centers_3->ClusterCenter(1).ValueOf(1) == 1.0);
  assert(cluster_centers_3->ClusterCenter(1).ValueOf(2) == 0.0);
  assert(cluster_centers_3->ClusterCenter(1).ValueOf(3) == 0.0);
  assert(cluster_centers_3->ClusterCenter(2).ValueOf(1) == 0.0);
  assert(cluster_centers_3->ClusterCenter(2).ValueOf(2) == 1.0);
  assert(cluster_centers_3->ClusterCenter(2).ValueOf(3) == 0.0);

  SfClusterCenters* cluster_centers_4 = new SfClusterCenters(5);
  srand(105);  // Use a fixed seed for deterministic testing.
  sofia_cluster::InitializeWithKRandomCenters(3, data_set, cluster_centers_4);
  sofia_cluster::SGDKmeans(1000, data_set, cluster_centers_4);
  assert(cluster_centers_4->ClusterCenter(0).ValueOf(1) < 1.01 &&
	 cluster_centers_4->ClusterCenter(0).ValueOf(1) > 0.99);
  assert(cluster_centers_4->ClusterCenter(0).ValueOf(2) == 0.0);
  assert(cluster_centers_4->ClusterCenter(0).ValueOf(3) < 1.01 &&
	 cluster_centers_4->ClusterCenter(0).ValueOf(3) > 0.99);

  assert(cluster_centers_4->ClusterCenter(1).ValueOf(1) < 1.01 &&
	 cluster_centers_4->ClusterCenter(1).ValueOf(1) > 0.99);
  assert(cluster_centers_4->ClusterCenter(1).ValueOf(2) == 0.0);
  assert(cluster_centers_4->ClusterCenter(1).ValueOf(3) == 0.0);

  assert(cluster_centers_4->ClusterCenter(2).ValueOf(1) < 0.01 &&
	 cluster_centers_4->ClusterCenter(2).ValueOf(1) >= 0);
  assert(cluster_centers_4->ClusterCenter(2).ValueOf(2) < 1.01 &&
	 cluster_centers_4->ClusterCenter(2).ValueOf(2) > 0.99);
  assert(cluster_centers_4->ClusterCenter(2).ValueOf(3) == 0.0);

  srand(100);
  SfClusterCenters* cluster_centers_5 = new SfClusterCenters(5);
  sofia_cluster::ClassicKmeansPlusPlus(3, data_set, cluster_centers_5);
  assert(cluster_centers_5->ClusterCenter(0).ValueOf(1) == 0);
  assert(cluster_centers_5->ClusterCenter(0).ValueOf(2) > 1.09 &&
	 cluster_centers_5->ClusterCenter(0).ValueOf(2) < 1.11);
  assert(cluster_centers_5->ClusterCenter(0).ValueOf(3) == 0);
  assert(cluster_centers_5->ClusterCenter(1).ValueOf(1) > 1.09 &&
	 cluster_centers_5->ClusterCenter(1).ValueOf(1) < 1.11);
  assert(cluster_centers_5->ClusterCenter(1).ValueOf(2) == 0);
  assert(cluster_centers_5->ClusterCenter(1).ValueOf(3) == 0);
  assert(cluster_centers_5->ClusterCenter(2).ValueOf(1) > 1.09 &&
	 cluster_centers_5->ClusterCenter(2).ValueOf(1) < 1.11);
  assert(cluster_centers_5->ClusterCenter(2).ValueOf(2) == 0);
  assert(cluster_centers_5->ClusterCenter(2).ValueOf(3) > 0.89 &&
	 cluster_centers_5->ClusterCenter(2).ValueOf(3) < 0.91);

  SfClusterCenters* cluster_centers_6 = new SfClusterCenters(5);
  sofia_cluster::OptimizedKmeansPlusPlus(3, data_set, cluster_centers_6);
  assert(cluster_centers_6->ClusterCenter(0).ValueOf(1) > 0.89 &&
	 cluster_centers_6->ClusterCenter(0).ValueOf(1) < 0.91);
  assert(cluster_centers_6->ClusterCenter(0).ValueOf(2) == 0);
  assert(cluster_centers_6->ClusterCenter(0).ValueOf(3) == 0);
  assert(cluster_centers_6->ClusterCenter(1).ValueOf(1) == 0);
  assert(cluster_centers_6->ClusterCenter(1).ValueOf(2) > 1.09 &&
	 cluster_centers_6->ClusterCenter(1).ValueOf(2) < 1.11);
  assert(cluster_centers_6->ClusterCenter(1).ValueOf(3) == 0);
  assert(cluster_centers_6->ClusterCenter(2).ValueOf(1) > 0.89 &&
	 cluster_centers_6->ClusterCenter(2).ValueOf(1) < 0.91);
  assert(cluster_centers_6->ClusterCenter(2).ValueOf(2) == 0);
  assert(cluster_centers_6->ClusterCenter(2).ValueOf(3) > 1.09 &&
	 cluster_centers_6->ClusterCenter(2).ValueOf(3) < 1.11);

  SfClusterCenters* cluster_centers_7 = new SfClusterCenters(5);
  sofia_cluster::SamplingKmeansPlusPlus(3, 10, data_set, cluster_centers_7);
  assert(cluster_centers_7->ClusterCenter(0).ValueOf(1) > 0.89 &&
	 cluster_centers_7->ClusterCenter(0).ValueOf(1) < 0.91);
  assert(cluster_centers_7->ClusterCenter(0).ValueOf(2) == 0);
  assert(cluster_centers_7->ClusterCenter(0).ValueOf(3) == 0);
  assert(cluster_centers_7->ClusterCenter(1).ValueOf(1) == 0);
  assert(cluster_centers_7->ClusterCenter(1).ValueOf(2) > 1.09 &&
	 cluster_centers_7->ClusterCenter(1).ValueOf(2) < 1.11);
  assert(cluster_centers_7->ClusterCenter(1).ValueOf(3) == 0);
  assert(cluster_centers_7->ClusterCenter(2).ValueOf(1) > 0.99 &&
	 cluster_centers_7->ClusterCenter(2).ValueOf(1) < 1.01);
  assert(cluster_centers_7->ClusterCenter(2).ValueOf(2) == 0);
  assert(cluster_centers_7->ClusterCenter(2).ValueOf(3) > 0.99 &&
	 cluster_centers_7->ClusterCenter(2).ValueOf(3) < 1.01);

  float kmeans_objective_7 = 
   sofia_cluster::KmeansObjective(data_set, *cluster_centers_7);
  assert(kmeans_objective_7 > 0.139 && kmeans_objective_7 < 0.141);
  
  sofia_cluster::SGDKmeans(1000, data_set, cluster_centers_7);
  float improved_kmeans_objective_7 = 
    sofia_cluster::KmeansObjective(data_set, *cluster_centers_7);
  assert(improved_kmeans_objective_7 < kmeans_objective_7);
 
  std::cout << argv[0] << ": PASS" << std::endl;
}


