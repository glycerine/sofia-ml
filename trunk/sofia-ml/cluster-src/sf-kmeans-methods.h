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
//
// Various fast methods for performing initialization of kmeans clustering
// and local search for kmeans optimization.  These are implemented as
// non-member functions in the sofia_cluster namespace.

#ifndef SF_KMEANS_METHODS_H__
#define SF_KMEANS_METHODS_H__

#include <vector>
#include "sf-cluster-centers.h"
#include "../src/sf-data-set.h"

namespace sofia_cluster {

  // ---------------------------------------------------
  //          Kmeans Initialization Functions
  // ---------------------------------------------------

  // Draw k unique samples from data_set uniformly at random and use them
  // as the seed values for the cluster_centers.
  void InitializeWithKRandomCenters(int k,
				    const SfDataSet& data_set,
				    SfClusterCenters* cluster_centers);

  // Seed the cluster centers with k centers, sampled using the kmeans++
  // sampling algorithm, naively implemented.  That is, to perform the
  // D^2 sampling, each point is compared to all active centers on
  // each sampling round.
  void ClassicKmeansPlusPlus(int k,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers);

  // An optimized implementation of the kmeans++ sampling algorithm,
  // which cache's the distance of the closest center to each point
  // for faster D^2 sampling.  Here, each point is only compared to the
  // newest active center on each sampling round.
  void OptimizedKmeansPlusPlus(int k,
			       const SfDataSet& data_set,
			       SfClusterCenters* cluster_centers);

  // A further optimization of kmeans++, which only compares the
  // a point to the newest center if triangle inequality shows that the
  // newest center might be closer to this point than it's old nearest
  // center.
  void OptimizedKmeansPlusPlusTI(int k,
				 const SfDataSet& data_set,
				 SfClusterCenters* cluster_centers);

  // A sampling-based variant of kmeans++, in which each new center
  // is drawn from D^2 sampling on a subsample of size sample_size,
  // rather than D^2 sampling across the whole data set.
  void SamplingKmeansPlusPlus(int k,
			      int sample_size,
			      const SfDataSet& data_set,
			      SfClusterCenters* cluster_centers);

  // A sampling-based variant of the farthest-first sampling method,
  // in which we select the next center as the farthest point from all
  // current centers out of the points in a sample of size sample_size.
  void SamplingFarthestFirst(int k,
			     int sample_size,
			     const SfDataSet& data_set,
			     SfClusterCenters* cluster_centers);
  
  // ---------------------------------------------------
  //          Kmeans Optimization Functions
  // ---------------------------------------------------

  // Lloyd's classic batch kmeans algorithm.  If L1_lambda is set
  // to a positive value, each center is projected to an L1 ball of
  // radius at most lambda.  If L1_epsilon is set to a positive
  // value, we use an approximate projection which projects each
  // center to a radius of between lambda and (1+epsilon)*lambda.
  void BatchKmeans(int num_iterations,
		   const SfDataSet& data_set,
		   SfClusterCenters* cluster_centers,
		   float L1_lambda = -1.0,
		   float L1_epsilon = 0.0);

  // The online kmeans of Bottou and Bengio, using per-center learning
  // rates for fast convergence.  If L1_lambda is set
  // to a positive value, each center is projected to an L1 ball of
  // radius at most lambda.  If L1_epsilon is set to a positive
  // value, we use an approximate projection which projects each
  // center to a radius of between lambda and (1+epsilon)*lambda.
  void SGDKmeans(int num_iterations,
		 const SfDataSet& data_set,
		 SfClusterCenters* cluster_centers,
		 float L1_lambda = -1.0,
		 float L1_epsilon = 0.0);

  // A mini-batch variant of kmeans.  Each mini-batch round is performed
  // in two steps.  First, a mini batch of size mini_batch_size is sampled,
  // and the closest center for each point is the batch is cache'd.  Then,
  // each point is used to update its closest center using the per-center
  // learning rate update rule of SGDKmeans, above.  If L1_lambda is set
  // to a positive value, each center is projected to an L1 ball of
  // radius at most lambda.  If L1_epsilon is set to a positive
  // value, we use an approximate projection which projects each
  // center to a radius of between lambda and (1+epsilon)*lambda.
  void MiniBatchKmeans(int num_iterations,
		       int mini_batch_size,
		       const SfDataSet& data_set,
		       SfClusterCenters* cluster_centers,
		       float L1_lambda = -1.0,
		       float L1_epsilon = 0.0);

  void OneBatchKmeansOptimization(const SfDataSet& data_set,
				 SfClusterCenters* cluster_centers);

  void OneMiniBatchKmeansOptimization(const SfDataSet& data_set,
				      SfClusterCenters* cluster_centers,
				      int mini_batch_size,
				      vector<int>* per_center_step_counts);

  void OneStochasticKmeansStep(const SfSparseVector& x,
			       SfClusterCenters* cluster_centers,
			       vector<int>* per_center_step_counts);

  // ---------------------------------------------------
  //          Kmeans Evaluation Functions
  // ---------------------------------------------------

  // Compute the value of the kmeans objective function on the data set,
  // with respect to the given set of cluster centers.
  float KmeansObjective(const SfDataSet& data_set,
			const SfClusterCenters& cluster_centers);

}  // namespace sofia_cluster

#endif  // SF_KMEANS_METHODS_H__
