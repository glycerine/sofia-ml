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
// sf-cluster-center.h
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// A class for containing cluster centers used by clustering algorithms
// such as kmeans.  Each cluster center is represented by an SfWeightVector
// instance.

#ifndef SF_CLUSTER_CENTERS_H__
#define SF_CLUSTER_CENTERS_H__

#include <string>
#include <vector>

#include "../src/sf-sparse-vector.h"
#include "../src/sf-weight-vector.h"

using std::string;

enum ClusterCenterMappingType {
  SQUARED_DISTANCE,
  RBF_KERNEL
};

class SfClusterCenters {
 public:
  // Construct an empty object with no centers defined.  The dimensionality
  // argument gives the maximum dimensionality (max feature id + 1) that we
  // will encounter on this data.
  SfClusterCenters(int dimensionality);

  // As above, but create num_clusters empty cluster centers of the given
  // dimensionality.
  SfClusterCenters(int dimensionality, int num_clusters);

  // Initialize a set of SfClusterCenters from a file.  This file is
  // assumed to contain one string-represented SfWeightVector per line.
  // Will die on failure.
  SfClusterCenters(const string& file_name);

  // Copy the new_center into the set of cluster centers.  This increases the
  // number of cluster centers by one.  Also increases the dimensionality_
  // to the max of the current dimensionality_ and the dimensionality of
  // the new_center.
  void AddClusterCenter(const SfWeightVector& new_center);

  // Create a new cluster center at the location given by SfSparseVector x.
  void AddClusterCenterAt(const SfSparseVector& x);

  // Returns the squared Euclidean distance from x to the nearest cluster
  // center, and fills the closest_center_id output argument with the id
  // of the closest center.  This method will cause failure/death if there
  // are no centers defined, or if closest_center_id is null.
  float SqDistanceToClosestCenter(const SfSparseVector& x,
				  int* closest_center_id) const;

  // Returns the Eucliean distance from x to the specified cluster center.
  // This will fail/die if the specified center_id does not exist.
  float SqDistanceToCenterId(int center_id, const SfSparseVector& x) const;

  // Maps example x to a new transformed vector x', where each coordinate
  // i (ranging from 1..k+1) of the returned x' corresponds to cluster_center
  // i-1.  The value of coordinate i is given by f(x, cluster_center(i-1)),
  // where f is determined by the ClusterCenterMappingType type:
  //   SQUARED_DISTANCE: f(x, c) = ||x - c|| ^ 2
  //   RBF_KERNEL: f(x, c) = exp(-p * ||x - c|| ^ 2)
  SfSparseVector* MapVectorToCenters(const SfSparseVector& x,
				     ClusterCenterMappingType type,
				     float p) const;

  // Accessors.
  // Returns a reference to the given cluster center.  Behavior is undefined
  // (but surely unsafe) if the speficied center does not exist.
  const SfWeightVector& ClusterCenter(int center_id) const;

  // Returns a mutable reference to the given cluster center.  Behavior is
  // undefined if the speficied center does not exist.
  SfWeightVector* MutableClusterCenter(int center_id);

  int GetDimensionality() { return dimensionality_; }

  // Utility methods.
  // Returns a string representation of this object, with one
  // string-reresented SfWeightVector per line.
  string AsString();
  
  // Empties the set of cluster centers.
  void Clear() { cluster_centers_.clear(); }

  // Return the number of cluster centers.
  int Size() const { return cluster_centers_.size(); }

 protected:
  float SqDistance(int center_id, const SfSparseVector& x); 
  
  // The set of cluster centers.
  vector<SfWeightVector> cluster_centers_;

  // Maximum dimensionality of any weight vector; used to initialized
  // weight vector objects.
  int dimensionality_;
};

#endif  // SF_CLUSTER_CENTERS_H__
