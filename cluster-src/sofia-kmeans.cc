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
// Main executable for running kmeans.

#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "sf-cluster-centers.h"
#include "sf-kmeans-methods.h"
#include "../src/simple-cmd-line-helper.h"

using std::string;

void CommandLine(int argc, char** argv) {
  AddFlag("--training_file", "File to be used for training.", string(""));
  AddFlag("--test_file", "File to be used for testing.", string(""));
  AddFlag("--model_in", "Read in a model from this file.", string(""));
  AddFlag("--model_out", "Write the model to this file.", string(""));
  AddFlag("--cluster_assignments_out",
	  "Assign each example in the --test_file to its closest cluster\n"
	  "    center, and write these results to this file.  Format of the \n"
	  "    file is <nearest center id>TAB<true label (if any)>."
	  "    Default: no output file.",
	  string(""));
  AddFlag("--cluster_mapping_out",
	  "Transform each vector in --test_file by mapping it onto the set \n"
	  "    of cluster centers.  Each example x is mapped to a new \n"
	  "    transformed vector x', where each coordinate i (ranging \n"
	  "    from  1..k+1) of  x' corresponds to cluster_center  i-1.\n"
	  "    The value of coordinate i is given by f(x, c(i-1))\n"
	  "    where f is --cluster_mapping_type.\n"
	  "    Default: no mapping output file.",
	  string(""));
  AddFlag("--cluster_mapping_type",
	  "The mapping function to use to create the --cluster_mapping_out \n"
	  "    file.  The value p is given by --cluster_mapping_param.\n"
	  "    Options are:\n"
	  "      squared_distance        f(x, c) = ||x - c|| ^ 2\n"
	  "      rbf_kernel              f(x, c) = exp(-p * ||x - c|| ^ 2)\n"
	  "    Default: squared_distance",
	  string("squared_distance"));
  AddFlag("--cluster_mapping_param",
	  "   The parameter value to use in --cluster_mapping_out.",
	  float(1.0));
  AddFlag("--random_seed",
          "When set to non-zero value, use this seed instead of seed \n"
	  "    from system clock. This can be useful for parameter tuning \n"
	  "    in cross-validation, as setting a fixed seed by hand forces \n"
	  "    examples to be sampled in the same order.  However\n"
          "    for actual training/test, this should never be used.\n"
          "    Default: 0",
          int(0));
  AddFlag("--k",
	  "The number of cluster centers to find.  Must be set.\n",
	  int(0));
  AddFlag("--init_type",
	  "Initialization procedure for seeding the kmeans optimization.\n"
	  "    Options are:\n"
	  "      random          random selection of cluster centers\n"
	  "      kmeans_pp       kmeans++ initialization method (naive)\n"
	  "      optimized_kmeans_pp   optimized kmeans++\n"
	  "    Default: random",
	  string("random"));
  AddFlag("--opt_type",
	  "Optimization procedure for kmeans objective.\n"
	  "    Options are: batch_kmeans, sgd_kmeans, mini_batch_kmeans\n"
	  "     Default: mini_batch_kmeans",
	  string("mini_batch_kmeans"));
  AddFlag("--sample_size",
          "When using sampling_kmeans_pp, the number of examples to sample on "
	  "each round.\n"
          "    Default: 1000",
          int(1000));
  AddFlag("--mini_batch_size",
          "When using mini_batch_kmeans, the number of examples to sample on "
	  "each round.\n"
          "    Default: 100",
          int(100));
  AddFlag("--iterations",
          "Number of optimization iterations to take.\n"
          "    Default: 1000",
          int(100000));
  AddFlag("--buffer_mb",
          "Size of buffer to use in reading/writing to files, in MB.\n"
          "    Default: 40",
          int(40));
  AddFlag("--dimensionality",
          "Index value of largest feature index in training data set. \n"
          "    Default: 2^17 = 131072",
          int(2<<16));
  AddFlag("--no_bias_term",
          "When set, causes a bias term x_0 to be set to 0 for every \n"
          "    feature vector loaded from files, rather than the default \n"
	  "    of x_0 = 1.\n"
          "    Default: set.",
          bool(true));
  AddFlag("--objective_after_init",
          "Compute value of the kmeans objective function on training data,\n"
	  "    after initializing the cluster centers.\n"
          "    Default is not to do this.",
          bool(false));
  AddFlag("--objective_after_training",
          "Compute value of the kmeans objective function on training data,\n"
	  "    after completing training the cluster centers.\n"
          "    Default is not to do this.",
          bool(false));
  AddFlag("--objective_on_test",
          "Compute value of the kmeans objective function on test data.\n"
          "    Default is not to do this.",
          bool(false));
  AddFlag("--L1_lambda",
	  "When set to a positive value, forces each cluster center to\n"
	  "    lie within a ball with L1 radius at most --L1_lambda.\n"
	  "    Default is not to enforce this constraint.",
	  float(-1.0));
  AddFlag("--L1_epsilon",
	  "When set to a positive value, we use an approximate L1 projection\n"
	  "    rather than an exact L1 projection.  The projection results\n"
	  "    in each center lying within a ball with L1 radius of between\n"
	  "    --L1_lambda and (1 + --L1_epsilon) * --L1_lambda.  Default is\n"
	  "    to perform exact projection.",
	  float(0.0));

  ParseFlags(argc, argv);
}

void PrintElapsedTime(clock_t start, const string& message) {
  float num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
  std::cout << message << num_secs << std::endl;
}

void InitializeCenters(const SfDataSet& data_set,
		       SfClusterCenters* cluster_centers) {
  if (CMD_LINE_INTS["--k"] <= 0) {
    std::cerr << "--k must be greater than 0." << std::endl;
    exit(0);
  }
  assert(cluster_centers != NULL);
  clock_t initialize_start = clock();
  
  if (CMD_LINE_STRINGS["--init_type"] == "random") {
    sofia_cluster::InitializeWithKRandomCenters(CMD_LINE_INTS["--k"],
						data_set,
						cluster_centers);
  } else if (CMD_LINE_STRINGS["--init_type"] == "kmeans_pp") {
    sofia_cluster::ClassicKmeansPlusPlus(CMD_LINE_INTS["--k"],
					 data_set,
					 cluster_centers);
  } else if (CMD_LINE_STRINGS["--init_type"] == "optimized_kmeans_pp") {
    sofia_cluster::OptimizedKmeansPlusPlus(CMD_LINE_INTS["--k"],
					   data_set,
					   cluster_centers);
  } else if (CMD_LINE_STRINGS["--init_type"] == "optimized_kmeans_pp_ti") {
    sofia_cluster::OptimizedKmeansPlusPlusTI(CMD_LINE_INTS["--k"],
					   data_set,
					   cluster_centers);
  } else if (CMD_LINE_STRINGS["--init_type"] == "sampling_kmeans_pp") {
    sofia_cluster::SamplingKmeansPlusPlus(CMD_LINE_INTS["--k"],
					  CMD_LINE_INTS["--sample_size"],
					  data_set,
					  cluster_centers);
  } else if (CMD_LINE_STRINGS["--init_type"] == "sampling_farthest") {
    sofia_cluster::SamplingFarthestFirst(CMD_LINE_INTS["--k"],
					 CMD_LINE_INTS["--sample_size"],
					 data_set,
					 cluster_centers);
  } else { 
    std::cerr << "--init_type " << CMD_LINE_STRINGS["--init_type"]
	      << " not supported." << std::endl;
    exit(0);
  }

  PrintElapsedTime(initialize_start, "Time to initialize cluster centers: ");
}


void OptimizeCenters(const SfDataSet& data_set,
		       SfClusterCenters* cluster_centers) {
  if (CMD_LINE_INTS["--iterations"] < 0) {
    std::cerr << "--iterations must be non-negative." << std::endl;
    exit(0);
  }
  assert(cluster_centers != NULL);
  clock_t optimize_start = clock();
  
  if (CMD_LINE_STRINGS["--opt_type"] == "batch_kmeans") {
    sofia_cluster::BatchKmeans(CMD_LINE_INTS["--iterations"],
			       data_set,
			       cluster_centers,
			       CMD_LINE_FLOATS["--L1_lambda"],
			       CMD_LINE_FLOATS["--L1_epsilon"]);
  } else if (CMD_LINE_STRINGS["--opt_type"] == "sgd_kmeans") {
    sofia_cluster::SGDKmeans(CMD_LINE_INTS["--iterations"],
			     data_set,
			     cluster_centers,
			     CMD_LINE_FLOATS["--L1_lambda"],
			     CMD_LINE_FLOATS["--L1_epsilon"]);
  } else if (CMD_LINE_STRINGS["--opt_type"] == "mini_batch_kmeans") {
    sofia_cluster::MiniBatchKmeans(CMD_LINE_INTS["--iterations"],
				   CMD_LINE_INTS["--mini_batch_size"],
				   data_set,
				   cluster_centers,
				   CMD_LINE_FLOATS["--L1_lambda"],
				   CMD_LINE_FLOATS["--L1_epsilon"]);
  } else { 
    std::cerr << "--opt_type " << CMD_LINE_STRINGS["--opt_type"]
	      << " not supported." << std::endl;
    exit(0);
  }

  PrintElapsedTime(optimize_start, "Time to optimize cluster centers: ");
}

float ComputeObjective(const SfDataSet& data_set,
		      const SfClusterCenters& cluster_centers,
		      const string& objective_type) {
  clock_t objective_start = clock();
  float objective_value = sofia_cluster::KmeansObjective(data_set,
							 cluster_centers); 
  std::cout << "Objective function value for " << objective_type << ": " 
	    <<  objective_value << std::endl;
  PrintElapsedTime(objective_start,
		   "Time to compute objective function: ");  
  return objective_value;
}

SfDataSet* NewDataSet(const string& file_name) {
  std::cerr << "Reading data from: " << file_name << std::endl;
  clock_t read_data_start = clock();
  SfDataSet* data_set = new SfDataSet(file_name,
				      CMD_LINE_INTS["--buffer_mb"],
				      !CMD_LINE_BOOLS["--no_bias_term"]);
  PrintElapsedTime(read_data_start,
		   "Time to read training data from " + file_name + ": ");  
  return data_set;
}

void LoadModelFromFile(const string& file_name,
		       SfClusterCenters** cluster_centers) {
  if (*cluster_centers != NULL) delete *cluster_centers;

  *cluster_centers = new SfClusterCenters(file_name);
  assert(*cluster_centers != NULL);
}

void SaveModelToFile(const string& file_name,
		     SfClusterCenters* cluster_centers) {
  std::fstream model_stream;
  model_stream.open(file_name.c_str(), std::fstream::out);
  if (!model_stream) {
    std::cerr << "Error opening model output file " << file_name << std::endl;
    exit(1);
  }
  std::cerr << "Writing model to: " << file_name << std::endl;
  model_stream << cluster_centers->AsString();
  model_stream.close();
  std::cerr << "   Done." << std::endl;
}

int main (int argc, char** argv) {
  CommandLine(argc, argv);
  
  if (CMD_LINE_INTS["--random_seed"] == 0) {
    srand(time(NULL));
  } else {
    std::cerr << "Using random_seed: "
	      << CMD_LINE_INTS["--random_seed"] << std::endl;
    srand(CMD_LINE_INTS["--random_seed"]);
  }

  SfClusterCenters* cluster_centers =
    new SfClusterCenters(CMD_LINE_INTS["--dimensionality"]);
  
  // Load model (overwriting empty model), if needed.
  if (!CMD_LINE_STRINGS["--model_in"].empty()) {
    LoadModelFromFile(CMD_LINE_STRINGS["--model_in"], &cluster_centers); 
  }

  // Train model, if needed.
  if (!CMD_LINE_STRINGS["--training_file"].empty()) {
    SfDataSet* training_data = NewDataSet(CMD_LINE_STRINGS["--training_file"]);

    InitializeCenters(*training_data, cluster_centers);
    if (CMD_LINE_BOOLS["--objective_after_init"]) {
      ComputeObjective(*training_data, *cluster_centers, "initialization");
    }

    OptimizeCenters(*training_data, cluster_centers);
    if (CMD_LINE_BOOLS["--objective_after_training"]) {
      ComputeObjective(*training_data, *cluster_centers, "training");
    }
  }    

  // Save cluster centers, if needed.
  if (!CMD_LINE_STRINGS["--model_out"].empty()) {
    SaveModelToFile(CMD_LINE_STRINGS["--model_out"], cluster_centers);
  }

  // Test cluster centers, if needed.
  if (!CMD_LINE_STRINGS["--test_file"].empty()) {
    SfDataSet* test_data = NewDataSet(CMD_LINE_STRINGS["--test_file"]);
    if (CMD_LINE_BOOLS["--objective_on_test"]) {
      ComputeObjective(*test_data, *cluster_centers, "test");
    }

    if (!CMD_LINE_STRINGS["--cluster_assignments_out"].empty()) {
      std::fstream assignment_stream;
      assignment_stream.open(CMD_LINE_STRINGS["--cluster_assignments_out"].
			     c_str(),
			     std::fstream::out);
      if (!assignment_stream) {
	std::cerr << "Error opening cluster assingments output file " 
		  << CMD_LINE_STRINGS["--cluster_assignments_out"]
		  << std::endl;
	exit(1);
      }
      std::cerr << "Writing cluster assignments to: "
		<< CMD_LINE_STRINGS["--cluster_assignments_out"] << std::endl;
      for (int i = 0; i < test_data->NumExamples(); ++i) {
	int closest_center;
	cluster_centers->SqDistanceToClosestCenter(test_data->VectorAt(i),
						   &closest_center);
	assignment_stream << closest_center << "\t" 
			  << test_data->VectorAt(i).GetY() << std::endl;
      }
      assignment_stream.close();
    }

    if (!CMD_LINE_STRINGS["--cluster_mapping_out"].empty()) {
      ClusterCenterMappingType type;
      if (CMD_LINE_STRINGS["--cluster_mapping_type"] == "squared_distance") {
	type = SQUARED_DISTANCE;
      } else if (CMD_LINE_STRINGS["--cluster_mapping_type"] == "rbf_kernel") {
	type = RBF_KERNEL;
      } else {
	std::cerr << "Cluster Mapping Type: "
		  << CMD_LINE_STRINGS["--cluster_mapping_type"]
		  << " is not supported.";
	exit(1);
      }
      float p = CMD_LINE_FLOATS["--cluster_mapping_param"];

      std::fstream mapping_stream;
      mapping_stream.open(CMD_LINE_STRINGS["--cluster_mapping_out"].
			     c_str(),
			     std::fstream::out);
      if (!mapping_stream) {
	std::cerr << "Error opening cluster mappings output file " 
		  << CMD_LINE_STRINGS["--cluster_mapping_out"]
		  << std::endl;
	exit(1);
      }
      std::cerr << "Writing cluster mappings to: "
		<< CMD_LINE_STRINGS["--cluster_mapping_out"] << std::endl;
      for (int i = 0; i < test_data->NumExamples(); ++i) {
	SfSparseVector* x_t =
	  cluster_centers->MapVectorToCenters(test_data->VectorAt(i), type, p);
	mapping_stream << x_t->AsString() << std::endl;
	delete x_t;
      }
      mapping_stream.close();
    }
  }

  std::cerr << "   Done." << std::endl;
}
