#================================================================================#
# Copyright 2009 Google Inc.                                                     #
#                                                                                # 
# Licensed under the Apache License, Version 2.0 (the "License");                #
# you may not use this file except in compliance with the License.               #
# You may obtain a copy of the License at                                        #
#                                                                                #
#      http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                                #
# Unless required by applicable law or agreed to in writing, software            #
# distributed under the License is distributed on an "AS IS" BASIS,              #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       #
# See the License for the specific language governing permissions and            #
# limitations under the License.                                                 #
#================================================================================#

GCC= g++ -O3 -lm -Wall

#================================================================================#
#                           Main Make Commands                                   #
#================================================================================#

# Primary executable binary.
sofia-ml:
	$(GCC) -o sofia-ml sofia-ml.cc sofia-ml-methods.cc sf-weight-vector.cc sf-sparse-vector.cc sf-data-set.cc sf-hash-weight-vector.cc sf-hash-inline.cc
	cp sofia-ml ..

# Build and execute all unit tests.
all_test: sf-sparse-vector_test sf-data-set_test sf-hash-inline_test sf-weight-vector_test simple-cmd-line-helper_test sofia-ml-methods_test

# Remove all executable binaries (including tests).
clean:
	rm -f sofia-ml
	rm -f ../sofia-ml
	rm -f sf-sparse-vector_test
	rm -f sf-data-set_test
	rm -f sf-hash-inline_test
	rm -f sf-weight-vector_test
	rm -f simple-cmd-line-helper_test
	rm -f sofia-ml-methods_test

#================================================================================#
#                           Individual Unit Tests                                #
#================================================================================#

sf-sparse-vector_test:
	$(GCC) -o sf-sparse-vector_test sf-sparse-vector_test.cc sf-sparse-vector.cc
	./sf-sparse-vector_test

sf-data-set_test:
	$(GCC) -o sf-data-set_test sf-data-set_test.cc sf-data-set.cc sf-sparse-vector.cc
	./sf-data-set_test

sf-hash-inline_test:
	$(GCC) -o sf-hash-inline_test sf-hash-inline_test.cc sf-hash-inline.cc
	./sf-hash-inline_test

sf-weight-vector_test:
	$(GCC) -o sf-weight-vector_test sf-weight-vector_test.cc sf-weight-vector.cc sf-sparse-vector.cc
	./sf-weight-vector_test

simple-cmd-line-helper_test:
	$(GCC) -o simple-cmd-line-helper_test simple-cmd-line-helper_test.cc
	./simple-cmd-line-helper_test

sofia-ml-methods_test:
	$(GCC) -o sofia-ml-methods_test sf-weight-vector.cc sf-sparse-vector.cc sf-data-set.cc sofia-ml-methods.cc sofia-ml-methods_test.cc
	./sofia-ml-methods_test