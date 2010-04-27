#============================================================================#
# Copyright 2009 Google Inc.                                                 #
#                                                                            # 
# Licensed under the Apache License, Version 2.0 (the "License");            #
# you may not use this file except in compliance with the License.           #
# You may obtain a copy of the License at                                    #
#                                                                            #
#      http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                            #
# Unless required by applicable law or agreed to in writing, software        #
# distributed under the License is distributed on an "AS IS" BASIS,          #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   #
# See the License for the specific language governing permissions and        #
# limitations under the License.                                             #
#============================================================================#

GCC= g++ -O3 -lm -Wall

sofia-kmeans:
	$(GCC) -o sofia-kmeans sofia-kmeans.cc sf-cluster-centers.cc sf-kmeans-methods.cc ../src/sf-weight-vector.cc  ../src/sf-data-set.cc  ../src/sf-sparse-vector.cc
	cp sofia-kmeans ..

all_test: sf-cluster-centers_test sf-kmeans-methods_test

sf-cluster-centers_test:
	$(GCC) -o sf-cluster-centers_test sf-cluster-centers_test.cc sf-cluster-centers.cc ../src/sf-weight-vector.cc ../src/sf-sparse-vector.cc
	./sf-cluster-centers_test

sf-kmeans-methods_test:
	$(GCC) -o sf-kmeans-methods_test sf-kmeans-methods_test.cc sf-kmeans-methods.cc sf-cluster-centers.cc ../src/sf-weight-vector.cc ../src/sf-sparse-vector.cc ../src/sf-data-set.cc 
	./sf-kmeans-methods_test

clean:
	rm -f ../sofia-kmeans
	rm -f sofia-kmeans
	rm -f sf-cluster-centers_test
	rm -f sf-kmeans-methods_test
