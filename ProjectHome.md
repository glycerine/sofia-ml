# sofia-ml #

The suite of fast incremental algorithms for machine learning (sofia-ml) can be used for training models for classification, regression, ranking, or combined regression and ranking.  Several different techniques are available.  This release is intended to aid researchers and practitioners who require fast methods for classification and ranking on large, sparse data sets.

Supported classification, regression, and ranking learners include:
  * Pegasos SVM
  * Stochastic Gradient Descent (SGD) SVM
  * Passive-Aggressive Perceptron
  * Perceptron with Margins
  * ROMMA
  * Logistic Regression (with Pegasos Projection)


These learners can be configured for binary classification, ranking, and optimizing ROC area, through the use of several available sampling methods for stochastic gradient descent.

This implementation is very fast.  For example, 100,000 Pegasos SVM training iterations can be performed on data from the CCAT task from the RCV1 benchmark data set (with roughly 780,000 examples) in 0.1 CPU seconds on an ordinary 2.4GHz laptop, with no loss in classification performance compared with other SVM methods.  On LETOR learning to rank benchmark tasks, training time with 100,000 Pegasos SVM rank steps complete 0.2 CPU seconds on an ordinary laptop.

The primary computational bottleneck is actually reading the data off of disk; sofia-ml reads and parses data from disk substantially faster than other SVM packages we tested.  For example, sofia-ml can read and parse data nearly 10 times faster than the reference Pegasos implementation by Shalev-Shwartz, and nearly 3 times faster than `svm_perf` by Joachims.

This package provides a commandline utility for training models and using them to predict on new data, and also exposes an API for model training and prediction that can be used in new applications.  The underlying libraries for data sets, weight vectors, and example vectors are also provided for researchers wishing to use these classes to implement other algorithms.

# sofia-kmeans #

There is also an unsupervised clustering package for k-means clustering.  This includes a mini-batch k-means variant that is exceptionally fast for large, sparse data sets, and an optimized version of the k-means++ seeding algorithm.

This pacakage can be used for learning cluster centers, for assigning new data to the learned cluster centers, and for mapping a given data set onto a new feature space based on the learned cluster centers.

For more information on this subpackage, see [SofiaKMeans](http://code.google.com/p/sofia-ml/wiki/SofiaKMeans).


---


---


---


## Quick Start ##

These quick-start instructions assume the use of the unix/linux commandline, with g++ installed.  There are no external code dependencies.

**Step 1** Check out the code:

```
> svn checkout http://sofia-ml.googlecode.com/svn/trunk/sofia-ml sofia-ml-read-only
```

**Step 2** Compile the code:
```
> cd sofia-ml-read-only/src/
> make
> ls ../sofia-ml
# Executable should be in main sofia-ml-read-only directory.

# If the above did not succeed, run the unit tests to help locate the problem:
> make clean
> make all_test
```

**Step 3** Test the code:
```
> cd ..
> ./sofia-ml
# This should display the set of commandline flags and descriptions.

# Train a model on the demo training data.
> ./sofia-ml --learner_type pegasos --loop_type stochastic --lambda 0.1 --iterations 100000 --dimensionality 150000 --training_file demo/demo.train --model_out demo/model
# This should display something like the following:
Reading training data from: demo/demo.train
Time to read training data: 0.056134
Time to complete training: 0.075364
Writing model to: demo/model
   Done.

# Test the model on the demo data.
> ./sofia-ml --model_in demo/model --test_file demo/demo.train --results_file demo/results.txt
# Should display the following:
Reading model from: demo/model
   Done.
Reading test data from: demo/demo.train
Time to read test data: 0.046729
Time to make test prediction results: 0.000844
Writing test results to: demo/results.txt
   Done.

# Examine a few results in the results file:
> head -5 demo/results.txt
# Format is: <prediction value>\t<label from test file>.  Each line in the results
# file corresponds to the same line (in order) in the test file.
1.02114	1
1.18046	1
-1.24609	-1
-1.12822	-1
-1.41046	-1
# Note that exact results may vary slightly because these algorithms train
# by randomly sampling one example at a time.

# Evaluate the results:
> perl eval.pl demo/results.txt
# Should display something like:

Results for demo/results.txt: 

Accuracy  0.9880  (using threshold 0.00) (988/1000)
Precision 0.9719  (using threshold 0.00) (311/320)
Recall    0.9904  (using threshold 0.00) (311/314)
ROC area: 0.999406 

Total of 1000 trials. 

# Note that this evaluation script has limited functionality.  For more
# options, we recommend using the perf software by Rich Caruana (developed for
# the KDD Cup 2004), available at: http://kodiak.cs.cornell.edu/kddcup/software.html
```


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


---


---


---


## Commandline Details ##

### File Input and Output ###

**--model\_in** _file_
  * Read in a model from this file before doing any training or testing.

**--model\_out** _file_
  * Write the model to this file when done with everything.

**--training\_file** _file_
  * File to be used for training.  When set, causes model training to occur.

**--test\_file** _file_
  * File to be used for testing.  When set, causes current model (either loaded from `--model_in` or trained from `--training_file` to be tested on test data.

**--results\_file** _file_
  * File to which to write predictions, when `--test_file` is used.  Results for each line are in the format `<prediction>\t<label from test file>\n` and correspond line-by-line with the examples form the `--test_file`.

### Learning Options ###

**--learner\_type** _type_
  * Type of learner to use.
  * Options are:
    * `pegasos` Use the Pegasos SVM learning algorithm.  `--lambda` sets the regularization parameter, with values closer to zero giving less regularization.  Note that Pegasos enforces a hard constraint that the model weight vector must lie within an L2 ball of radius at most 1/sqrt(lambda).  Also relies on `--eta_type`.
    * `sgd-svm` Use the SGD-SVM learning algorithm.  `--lambda` sets the regularization parameter, with values closer to zero giving less regularization.  Also relies on `--eta_type`
    * `passive-aggressive`  Use the Passive Aggressive Perceptron learning algorithm.  `--passive-aggressive-c` sets the largest step size to be taken on any update step; this operates as a capacity term with values closer to zero encouraging simpler models.  `--passive-aggressive-lambda` will force the model weight vector to lie within an L2 ball of radius 1/sqrt(passive-aggressive-lambda)
    * `margin-perceptron`  Use the Perceptron with Margins algorithm.  `--perceptron-margin-size` sets the update margin.  When set to 0, this is exactly equivalent to the classical Perceptron by Rosenblatt.  When set to 1, this is equivalent to optimizing SVM hinge-loss without regularization.  Increasing values may give additional tolerance to noise.  Also relies on `--eta_type`.
    * `romma`  Use the ROMMA algorithm.  No parametert to set.
    * `logreg-pegasos`  Use Logistic Regression with Pegasos updates; we optimize logistic loss and enforce Pegasos-style regularization and constraints, with `--lambda` being the regularization parameter.  Also relies on `--eta_type`.  Note that the classification values provided by this method regression are logodds, and can be converted to probabilties using: exp(p) / (1 + exp(p)).
  * Default: pegasos

**--loop\_type** _type_
  * Type of sampling loop to use for training, controlling how examples are selected.
  * Options are:
    * `stochastic`  Perform normal stochastic sampling for stochastic gradient descent, for training binary classifiers.  On each iteration, pick a new example uniformly at random from the data set.
    * `balanced-stochastic` Perform a balanced sampling from positives and negatives in data set.  For each iteration, samples one positive example uniformly at random from the set of all positives, and samples one negative example uniformly at random from the set of all negatives.  This can be useful for training binary classifiers with a minority-class distribution.
    * `rank`  Perform indexed sampling of candidate pairs for pairwise learning to rank.  Useful when there are examples from several different `qid` groups.
    * `roc`  Perform indexed sampling to optimize ROC Area.
    * `query-norm-rank`  Perform sampling of candidate pairs, giving equal weight to each `qid` group regardless of its size.  Currently this is implemented with rejection sampling rather than indexed sampling, so this may run more slowly.
    * `combined-ranking`  Performs CRR algorithm for combined regression and ranking.  Alternates between pairwise rank-based steps and standard stochastic gradient steps on single examples.  Relies on --rank\_step\_probability to balance between these two kinds of updates.
    * `combined-roc`  Performs CRR algorithm for combined regression and ROC area optimization.  Alternates between pairwise roc-optimization-based steps and standard stochastic gradient steps on single examples.  Relies on --rank\_step\_probability to balance between these two kinds of updates.  This can be faster than the combined-ranking option when there are exactly two classes.
  * Default: stochastic

**--eta\_type** _type_
  * Type of update for learning rate to use.
  * Options are:
    * `basic`  On the i-th iteration, the learning rate eta is set to: 1000 / (i + 1000)
    * `pegasos`  On the i-th iteration, the learning rate eta is set to: 1 / (i `*` lambda)
    * `constant`  Always use learning rate eta of 0.02.
  * Default: pegasos

**--prediction\_type** _type_
  * Type of predictions to compute on test data.
  * Options are:
    * `linear` Use standard linear dot product <w,x> for predictions.
    * `logistic` Use prediction function of exp(<w,x>) / (1 + exp(<w,x>)) for prediction, in the manner of logistic regression.
> > Default: linear

**--no\_bias\_term**
  * When set, causes a bias term x\_0 to be set to 0 for every feature vector loaded from files, rather than the default of x\_0 = 1.
  * Setting this flag is equivalent to forcing a decision threshold of exactly 0 to be used.
  * The same setting of this flag should be used for training and testing.
  * Note that this flag as no effect for rank and roc optimzation.
  * Default: not set.

**--dimensionality** _int_
  * Index id of largest feature index in training data set, plus one.
  * Default: 131072 (131072 = 2^17)

**--iterations** _int_
  * Number of stochastic gradient steps to take.
  * Default: 100000

**--lambda** _float_
  * Value of lambda for SVM regularization, used by both Pegasos SVM and SGD-SVM.
  * Default: 0.1

**--passive\_aggressive\_c** _float_
  * Maximum size of any step taken in a single passive-aggressive update.

**--passive\_aggressive\_lambda** _float_
  * Lambda for pegasos-style projection for passive-aggressive update.
  * When set to 0 (default) no projection is performed.

**--perceptron\_margin\_size** _float_
  * Width of margin for perceptron with margins.
  * Default of 1 is equivalent to unregularized SVM-loss.

**--rank\_step\_probability** _float_
  * Probability that we will take a rank step (as opposed to a
> > standard stochastic gradient step) in a combined-ranking or combined-roc loop.
  * Default: 0.5


**--hash\_mask\_bits** _int_
  * When set to a non-zero value, causes the use of a hashed weight vector with hashed cross product features.
  * This allows learning on conjunction of features, at some increase in computational cost.
  * Note that this flag must be set both in training and testing to function properly.
  * The size of the hash table is set to 2^--hash\_mask\_bits.
  * Default value of 0 shows that hash cross products are not used.

### Other Options ###

**--random\_seed** _int_
  * When set to non-zero value, use this seed instead of seed from system clock.
  * This can be useful in testing and in parameter tuning.
  * Default: 0

**--training\_objective**
  * When set, computes the value of the standard SVM objective function on training data, after training.
  * Default is not to do this.

## References ##

If you use this source code for scientific research, please cite one of the following:

  * D. Sculley.  Combined Regression and Ranking.  Proceedings of the 16th Annual SIGKDD Conference on Knowledge Discover and Data Mining, 2010.

  * D. Sculley.  Web-Scale K-Means Clustering.  Proceedings of the 19th international conference on World Wide Web, 2010.

  * D. Sculley.  Large Scale Learning to Rank.  NIPS Workshop on Advances in Ranking, 2009.  _Presents the indexed sampling methods used learning to rank, including the `rank` and `roc` loops._

Additional reading and references:

  * K. Crammer, O. Dekel, J. Keshet, S. Shalev-Shwartz, and Y. Singer. Online passive-aggressive algorithms. J. Mach. Learn. Res., 7, 2006.  _Presents the Passive-Aggressive Perceptron algorithm._

  * T. Joachims. Optimizing search engines using clickthrough data. In KDD ’02: Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, 2002.  _Presents the RankSVM objective function, a pairwise objective function used by the `rank` loop method in `sofia-ml`._

  * Y. Li and P. M. Long. The relaxed online maximum margin algorithm. Mach. Learn., 46(1-3), 2002.  _Presents the ROMMA algorithm._

  * S. Shalev-Shwartz, Y. Singer, and N. Srebro. Pegasos: Primal estimated sub-gradient solver for SVM. In ICML ’07: Proceedings of the 24th international conference on Machine learning, 2007.  _Presents the Pegasos SVM solver._

  * T. Zhang. Solving large scale linear prediction problems using stochastic gradient descent algorithms. In ICML ’04: Proceedings of the twenty-ﬁrst international conference on Machine learning, 2004.  _Presents SGD SVM._

  * http://leon.bottou.org/projects/sgd  _Leon Bottou's SGD page, including experiments with SGD SVM._

  * W. Krauth and M. M´ezard. Learning algorithms with optimal stability in neural networks. Journal of Physics A, 20(11):745–752, 1987.   _Presents Perceptron with Margins._