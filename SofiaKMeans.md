# Introduction #

The `sofia-kmeans` package allows k-means clustering on large scale data.  Options are included for using both random cluster initialization and the k-means++ algorithm for cluster seeding.  There are optimization methods for batch k-means (Lloyd's classic kmeans algorithm), Bottou and Bengio's online k-means variant (sgd k-means), and a very fast mini-batch stochastic gradient descent method for large scale data sets.



---


---


---


**Step 1** Check out the code:

```
> svn checkout http://sofia-ml.googlecode.com/svn/trunk/sofia-ml sofia-ml-read-only
```

**Step 2** Compile the code:
```
> cd sofia-ml-read-only/cluster-src/
> make
> ls ../sofia-kmeans
# Executable should be in main sofia-ml-read-only directory.

# If the above did not succeed, run the unit tests to help locate the problem:
> make clean
> make all_test
```

**Step 3** Test the code:
```
> cd ..
> ./sofia-kmeans
# This should display the set of commandline flags and descriptions.


# Learn 5 cluster centers on the training data, using mini_batch_kmeans, initialized with randomly selected points.  Also, compute the value of the k-means objective function on the training data.
> ./sofia-kmeans --k 5 --init_type random --opt_type mini_batch_kmeans --mini_batch_size 100 --iterations 500 --objective_after_init --objective_after_training --training_file demo/demo.train --model_out demo/clusters.txt
# Should display the following.
Reading data from: demo/demo.train
Time to read training data from demo/demo.train: 0.046837
Time to initialize cluster centers: 0.015064
Objective function value for initialization: 191727
Time to compute objective function: 0.002865
Time to optimize cluster centers: 0.224908
Objective function value for training: 128422
Time to compute objective function: 0.002808
Writing model to: demo/clusters.txt
   Done.
   Done.

# Apply the learned cluster centers to test data, assigning each point to the nearest cluster center.  Also compute the value of the kmeans objective function on the test file.
> ./sofia-kmeans --model_in demo/clusters.txt --test_file demo/demo.test --objective_on_test --cluster_assignments_out demo/assignments.txt
# Should display the following.
Reading data from: demo/demo.test
Time to read training data from demo/demo.test: 0.046725
Objective function value for test: 129601
Time to compute objective function: 0.003934
Writing cluster assignments to: demo/assignments.txt
   Done.
# Look at the cluster assignments.  Format is <assigned center id> <class label from training file>.  (Actual results will vary.)
> head -5 demo/assignments.txt 
0	1
0	1
2	-1
0	-1
2	-1

# Map the data in one file to a new feature space, based on the cluster centers.  Use the rbf kernel as the mapping function.
> ./sofia-kmeans --model_in demo/clusters.txt --test_file demo/demo.test --cluster_mapping_out demo/mapped_test_data.txt --cluster_mapping_type rbf_kernel --cluster_mapping_param 0.001
Reading data from: demo/demo.test
Time to read training data from demo/demo.test: 0.046866
Writing cluster mappings to: demo/mapped_data.txt
   Done.
# Look at the mapped test data.
> head -4 demo/mapped_test_data.txt 
1 1:0.847188 2:0.803948 3:0.812755 4:0.714623 5:0.807991 
1 1:0.871236 2:0.827893 3:0.836631 4:0.758054 5:0.837016 
-1 1:0.855617 2:0.837934 3:0.884961 4:0.715338 5:0.829957 
-1 1:0.917343 2:0.874639 3:0.889385 4:0.771052 5:0.881724 
```

Note that you can do this mapping for test and training data, by first learning cluster centers on the training data, and then creating mapped data sets for both the training and test set.  You can then use =sofia-ml= to learn a linear classifier on the mapped training data, and test it on the mapped test data.  This linear classifier will be learning a kernelized model that is non-linear in the original feature space, but much faster than traditional kernelized SVM's.


---


---


---


## Command line flag options: ##

# File Options #

  * **--training\_file**       File to be used for training.  Optional.

  * **--test\_file**           File to be used for testing/application.  Optional.

  * **--model\_in**            Read in a model from this file before training/testing.  Optional.

  * **--model\_out**           Write the model to this file after training.  Optional.

  * **--buffer\_mb** Size of buffer to use in reading/writing to files, in MB.
> > Default: 40

  * **--dimensionality**      Index value of largest feature index in training data set.
> > Default: 2^17 = 131072

  * **--no\_bias\_term**        When set, causes a bias term x\_0 to be set to 0 for every feature vector loaded from files, rather than the default  of x\_0 = 1.
> > Default: set.

### k-Means Initialization Options ###

  * **--k**                   The number of cluster centers to find.  Must be set.

  * **--init\_type**           Initialization procedure for seeding the kmeans optimization.
> > Options are:
    * **random**          Random selection of cluster centers
    * **kmeans\_pp**       kmeans++ initialization method (naive)
    * **optimized\_kmeans\_pp**   Optimized kmeans++; this is identical to kmeans++ but is faster.
> > Default: random

  * **--random\_seed**         When set to non-zero value, use this seed instead of seed from the system clock. This can be useful for parameter tuning in cross-validation, as setting a fixed seed by hand forces examples to be sampled in the same order.  However for actual training/test, this should never be used.
> > Default: 0 (i.e., not used)

  * **--objective\_after\_init**  Compute value of the k-means objective function on training data, after initializing the cluster centers.
> > Default is not to do this.

### k-Means Training Options ###

  * **--opt\_type**            Optimization procedure for kmeans objective.
> > Options are:
    * **mini\_batch\_kmeans**  The mini\_batch kmeans algorithm in [2010](Sculley.md).  This uses a mini\_batch of size --mini\_batch\_size to compute gradients for updates on each iteration.   Coverges very quickly to solution nearly as good as batch\_kmeans.
    * **batch\_kmeans**   Lloyd's classic batch kmeans algorithm.
    * **sgd\_kmeans**  The online stochastic gradient descent method of Bottou and Bengio.  This converges very quickly, but tends to find lower quality solutions compared to batch\_kmeans or mini\_batch\_kmeans.
> > > Default: mini\_batch\_kmeans

  * **--iterations**          Number of optimization iterations to take.  This value will be very different for mini\_batch\_kmeans (10<sup>3 - 10</sup>5), batch\_kmeans (10<sup>1 - 10</sup>3), and sgd\_kmeans (10<sup>5 - 10</sup>8).  Of course, the cost of each iteration is also very different.

> > Default: 1000

  * **--mini\_batch\_size**     When using mini\_batch\_kmeans, the number of examples to sample on each round.  Larger --mini\_batch\_size values can find slightly better solutions, but are slower.  Reasonable values are often between 100 and 10000 for large data sets.
> > Default: 100

  * **--L1\_epsilon**  When set to a positive value, we use an approximate L1 projection rather than an exact L1 projection.  The projection results in each center lying within a ball with L1 radius of between --L1\_lambda and (1 + --L1\_epsilon) `*` --L1\_lambda.
> > Default is to perform exact projection.

  * **--L1\_lambda** When set to a positive value, forces each cluster center to lie within a ball with L1 radius at most --L1\_lambda.
> > Default is not to enforce this constraint.

  * **--objective\_after\_training**  Compute value of the kmeans objective function on training data, after completing training the cluster centers.
> > Default is not to do this.


### Test and Application Options ###

  * **--cluster\_assignments\_out**  Assign each example in the --test\_file to its closest cluster center, and write these results to this file.  Format of the  file is <nearest center id>TAB<true label (if any)>.
> > Default: no output file.

  * **--cluster\_mapping\_out**  Transform each vector in --test\_file by mapping it onto the set of cluster centers.  Each example x is mapped to a new transformed vector x', where each coordinate i (ranging from  1..k+1) of  x' corresponds to cluster\_center  i-1. The value of coordinate i is given by f(x, c(i-1))  where f is --cluster\_mapping\_type.
> > Default: no mapping output file.

  * **--cluster\_mapping\_param** The parameter value to use in --cluster\_mapping\_out.
> > Default is: 1.0.

  * **--cluster\_mapping\_type**  The mapping function to use to create the --cluster\_mapping\_out file.  The value p is given by --cluster\_mapping\_param.
> > Options are:
    * squared\_distance        f(x, c) = `|``|`x - c`|``|` ^ 2
    * rbf\_kernel              f(x, c) = exp(-p `*` `|``|`x - c`|``|` ^ 2)
> > Default: squared\_distance

  * **--objective\_on\_test**   Compute value of the kmeans objective function on test data.
> > Default is not to do this.


---


---


---


## Data Format ##

This package uses the popular SVM-light sparse data format.
```
<class-label> <feature-id>:<feature-value> ... <feature-id>:<feature-value>\n
<class-label> qid:<optional-query-id> <feature-id>:<feature-value> ... <feature-id>:<feature-value>\n
<class-label> <feature-id>:<feature-value> ... <feature-id>:<feature-value># Optional comment or extra data, following the optional "#" symbol.\n
```

The feature id's are expected to be in ascending numerical order. The lowest allowable feature-id is 1 (0 is reserved for the bias term internally.)  Any feature not specified is assumed to have value 0 to allow for sparse representation.

The class label for test data is required but not used; it's okay to put in a dummy placeholder value such as 0 for test data.  For binary-class classification problems, the training labels should be 1 or -1.  For ranking problems, the labels may be any numeric value, with higher values being judged as "more preferred".

Currently, the comment string is not used.  However, it is available for use in other algorithms, and can also be useful to aid in bookkeeping of data files.

Examples:
```
# Class label is 1, feature 1 has value 1.2, feature 2 (not listed) has value 0,
# and feature 3 has value -0.5.
1 1:1.2 3:-0.5

# Class label is -1, belongs to qid 3, and all feature values are zero except
# for feature 5011 with value 1.2.
-1 qid:3 5011:1.2

# Class label is -1, feature 1 has value 7, comment string is
# "This example is especially interesting."
-1 1:7 3:-0.5#This example is especially interesting.
```