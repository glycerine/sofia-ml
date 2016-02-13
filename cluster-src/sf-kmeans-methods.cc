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

#include "sf-kmeans-methods.h"

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <iostream>
#include <map>
#include <set>

namespace sofia_cluster {

  // ---------------------------------------------------
  //         Helper functions (Not exposed in API)
  // ---------------------------------------------------

  int RandInt(int num_vals) {
    return static_cast<int>(rand()) % num_vals;
  }

  float RandFloat() {
    return static_cast<float>(rand() / static_cast<float>(RAND_MAX));
  }

  const SfSparseVector& RandomExample(const SfDataSet& data_set) {
    int num_examples = data_set.NumExamples();
    int i = static_cast<int>(rand()) % num_examples;
    if (i < 0) {
      i += num_examples;
    }
    return data_set.VectorAt(i);
  }

  // ---------------------------------------------------
  //          Kmeans Initialization Functions
  // ---------------------------------------------------

  void InitializeWithKRandomCenters(int k,
                                    const SfDataSet& data_set,
                                    SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    std::set<int> selected_centers;
    // Sample k centers uniformly at random, with replacement.
    for (int i = 0; i < k; ++i) {
      cluster_centers->AddClusterCenterAt(RandomExample(data_set));
    }
  }

  void SamplingFarthestFirst(int k,
			     int sample_size,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point.
    int id = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(id));
    // Get the next k - 1 points.
    int center_id;
    for (int i = 1; i < k; ++i) {
      int best_distance = 0;
      int best_center = 0;
      for (int j = 0; j < sample_size; ++j) {
	int temp_id = RandInt(data_set.NumExamples());
	float temp_distance = cluster_centers->
	  SqDistanceToClosestCenter(data_set.VectorAt(temp_id), &center_id);
	if (temp_distance > best_distance) {
	  best_distance = temp_distance;
	  best_center = temp_id;
	}
      }
      cluster_centers->AddClusterCenterAt(data_set.VectorAt(best_center));
    }
  }

  void ClassicKmeansPlusPlus(int k,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point.
    int id = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(id));
    // Get the next k - 1 points.
    for (int i = 1; i < k; ++i) {
      // First, compute the total distance-mass, and distance for each point.
      float total_distance_mass = 0.0;
      std::map<float, int> distance_for_points;
      int center_id;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance =
	  cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(j),
						     &center_id);
	if (distance > 0) {
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
    }
  }

  void OptimizedKmeansPlusPlus(int k,
                               const SfDataSet& data_set,
                               SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));
    vector<float> best_center_ids(data_set.NumExamples(), 0);
    vector<float> best_distances(data_set.NumExamples(), FLT_MAX);
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      best_distances[i] =
	cluster_centers->SqDistanceToCenterId(0, data_set.VectorAt(i));
    }

    // Get the next (k - 1) points.
    for (int i = 1; i < k; ++i) {
      float total_distance_mass = 0.0;
      std::map<float, int> distance_for_points;
      int recently_added_center = i - 1;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance =
	  cluster_centers->SqDistanceToCenterId(recently_added_center,
						data_set.VectorAt(j));
	if (distance < best_distances[j]) {
	  best_distances[j] = distance;
	  best_center_ids[j] = recently_added_center;
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	} else {
	  distance_for_points[best_distances[j] + total_distance_mass] = j;
	  total_distance_mass += best_distances[j];
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
    }
  }

  void OptimizedKmeansPlusPlusTI(int k,
				 const SfDataSet& data_set,
				 SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));
    vector<float> best_center_ids(data_set.NumExamples(), 0);
    vector<float> best_distances(data_set.NumExamples());
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      best_distances[i] =
	cluster_centers->SqDistanceToCenterId(0, data_set.VectorAt(i));
    }

    vector<float> inter_center_distances;
    // Get the next (k - 1) points.
    for (int i = 1; i < k; ++i) {
      float total_distance_mass = 0.0;
      int recently_added_center = i - 1;
      std::map<float, int> distance_for_points;
      for (int j = 0; j < data_set.NumExamples(); ++j) {
	float distance;
	if (i >= 2 &&
	    inter_center_distances[best_center_ids[j]] >
	    2.0 * best_distances[j]) {
	  distance = best_distances[j];
	} else {
	  distance =
	    cluster_centers->SqDistanceToCenterId(recently_added_center,
						  data_set.VectorAt(j));
	}
	if (distance < best_distances[j]) {
	  best_distances[j] = distance;
	  best_center_ids[j] = recently_added_center;
	  distance_for_points[distance + total_distance_mass] = j;
	  total_distance_mass += distance;
	} else {
	  distance_for_points[best_distances[j] + total_distance_mass] = j;
	  total_distance_mass += best_distances[j];
	}
      }
      // Get an example with D^2 weighting.
      // Note that we're breaking ties arbitrarily.
      float sample_distance = RandFloat() * total_distance_mass;
      std::map<float, int>::iterator distance_iter =
	distance_for_points.lower_bound(sample_distance);
      if (distance_iter == distance_for_points.end()) {
	std::cerr << "No unique points left for cluster centers." << std::endl;
	exit(1);
      }
      // Add the new cluster center and update the inter-cluster distances.
      cluster_centers->AddClusterCenterAt(
        data_set.VectorAt(distance_iter->second));
      inter_center_distances.clear();
      for (int j = 0; j < cluster_centers->Size() - 1; ++j) {
	inter_center_distances.push_back(cluster_centers->
	  SqDistanceToCenterId(j,
			       data_set.VectorAt(distance_iter->second)));
      }
    }
  }

  void SamplingKmeansPlusPlus(int k,
                              int sample_size,
                              const SfDataSet& data_set,
                              SfClusterCenters* cluster_centers) {
    assert(k > 0 && k <= data_set.NumExamples());
    assert(sample_size > 0);
    // Get first point, and initialize best distances.
    int cluster_center = RandInt(data_set.NumExamples());
    cluster_centers->AddClusterCenterAt(data_set.VectorAt(cluster_center));

    int cluster_id;
    for (int i = 1; i < k; ++i) {
      int selected_center = 0;
      float total_distance_mass = 0.0;
      for (int j = 0; j < sample_size; ++j) {
	int proposed_cluster_center = RandInt(data_set.NumExamples());
	float distance = cluster_centers->SqDistanceToClosestCenter(
	  data_set.VectorAt(proposed_cluster_center),
	  &cluster_id);
	total_distance_mass += distance;
	if (RandFloat() < distance / total_distance_mass) {
	  selected_center = proposed_cluster_center;
	}
      }
      cluster_centers->AddClusterCenterAt(data_set.VectorAt(selected_center));
    }

  }

  // ---------------------------------------------------
  //          Kmeans Optimization Functions
  // ---------------------------------------------------

  void ProjectToL1Ball(float L1_lambda,
		       float L1_epsilon,
		       SfClusterCenters* cluster_centers) {
    if (L1_lambda > 0) {
      for (int i = 0; i < cluster_centers->Size(); ++i) {
	if (L1_epsilon == 0.0) {
	  cluster_centers->MutableClusterCenter(i)->ProjectToL1Ball(L1_lambda);
	} else {
	  cluster_centers->MutableClusterCenter(i)->
	    ProjectToL1Ball(L1_lambda, L1_epsilon);
	}
      }
    }
  }

  void BatchKmeans(int num_iterations,
		   const SfDataSet& data_set,
		   SfClusterCenters* cluster_centers,
		   float L1_lambda,
		   float L1_epsilon) {
    for (int i = 0; i < num_iterations; ++i) {
      OneBatchKmeansOptimization(data_set, cluster_centers);
      ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }
  }

  void SGDKmeans(int num_iterations,
                 const SfDataSet& data_set,
                 SfClusterCenters* cluster_centers,
		 float L1_lambda,
		 float L1_epsilon) {
    vector<int> per_center_step_counts;
    per_center_step_counts.resize(cluster_centers->Size());
    for (int i = 0; i < num_iterations; ++i) {
      OneStochasticKmeansStep(RandomExample(data_set),
			      cluster_centers,
			      &per_center_step_counts);
      if (i % 100 == 50)
	ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }    
    ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
  }

  void MiniBatchKmeans(int num_iterations,
		       int mini_batch_size,
		       const SfDataSet& data_set,
		       SfClusterCenters* cluster_centers,
		       float L1_lambda,
		       float L1_epsilon) {
    vector<int> per_center_step_counts;
    per_center_step_counts.resize(cluster_centers->Size());
    for (int i = 0; i < num_iterations; ++i) {
      OneMiniBatchKmeansOptimization(data_set,
				     cluster_centers,
				     mini_batch_size,
				     &per_center_step_counts);
      ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
    }    
    ProjectToL1Ball(L1_lambda, L1_epsilon, cluster_centers);
  }

  void OneBatchKmeansOptimization(const SfDataSet& data_set,
				  SfClusterCenters* cluster_centers) {
    assert(cluster_centers->Size() > 0);
    SfClusterCenters new_centers(cluster_centers->GetDimensionality(),
				 cluster_centers->Size());
    vector<int> examples_per_cluster(cluster_centers->Size(), 0);

    // Sum the vectors for each center.
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(i),
						 &closest_center);
      new_centers.MutableClusterCenter(closest_center)->
	AddVector(data_set.VectorAt(i), 1.0);
      ++examples_per_cluster[closest_center];
    }

    // Scale each center by 1/number of vectors.
    for (int i = 0; i < cluster_centers->Size(); ++i) {
      if (examples_per_cluster[i] > 0) {
	new_centers.MutableClusterCenter(i)->
	  ScaleBy(1.0 / examples_per_cluster[i]);
      }
    }
    // Swap in the new centers.
    cluster_centers->Clear();
    for (int i = 0; i < new_centers.Size(); ++i) {
      cluster_centers->AddClusterCenter(new_centers.ClusterCenter(i));
    }
  }

  void OneStochasticKmeansStep(const SfSparseVector& x,
                               SfClusterCenters* cluster_centers,
                               vector<int>* per_center_step_counts) {
    // Find the closest center.
    int closest_center;
    cluster_centers->SqDistanceToClosestCenter(x, &closest_center);
    
    // Take the step.
    float c = 1.0;
    float eta = c / (++(*per_center_step_counts)[closest_center] + c);
    cluster_centers->MutableClusterCenter(closest_center)->
      ScaleBy(1.0 - eta);
    cluster_centers->MutableClusterCenter(closest_center)->
      AddVector(x, eta);
  }
  
  void OneMiniBatchKmeansOptimization(const SfDataSet&  data_set,
				      SfClusterCenters* cluster_centers,
				      int mini_batch_size,
				      vector<int>* per_center_step_counts) {
    // Compute closest centers for a mini-batch.
    vector<vector<int> > mini_batch_centers(cluster_centers->Size());
    for (int i = 0; i < mini_batch_size; ++i) {
      // Find the closest center for a random example.
      int x_id = RandInt(data_set.NumExamples());
      int closest_center;
      cluster_centers->SqDistanceToClosestCenter(data_set.VectorAt(x_id),
						 &closest_center);
      mini_batch_centers[closest_center].push_back(x_id);
    }
    // Apply the mini-batch.
    for (unsigned int i = 0; i < mini_batch_centers.size(); ++i) {
      for (unsigned int j = 0; j < mini_batch_centers[i].size(); ++j) {
	float c = 1.0;
	float eta = c / (++(*per_center_step_counts)[i] + c);
	cluster_centers->MutableClusterCenter(i)->ScaleBy(1.0 - eta);
	cluster_centers->MutableClusterCenter(i)->
	  AddVector(data_set.VectorAt(mini_batch_centers[i][j]), eta);
      }
    }
  }
  
  // ---------------------------------------------------
  //          Kmeans Evaluation Functions
  // ---------------------------------------------------

  float KmeansObjective(const SfDataSet& data_set,
		       const SfClusterCenters& cluster_centers) {
    if (cluster_centers.Size() == 0) return FLT_MAX;
    int center_id;
    float total_sq_distance = 0.0;
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      total_sq_distance +=
	cluster_centers.SqDistanceToClosestCenter(data_set.VectorAt(i),
						  &center_id);
    }
    return total_sq_distance;
  }

}  // namespace sofia_cluster
