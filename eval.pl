#!/usr/bin/perl
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

my $file = shift;
my $t = shift;

die "use: ./eval.pl [data file] [decision threshold (default 0.0)] \n"
    if (not $file);

my $threshold = (not $t) ? 0 : $t;

open IN, "< $file" 
    or die "Can't open file $file $! \n";
my @lines = <IN>;
close IN;

my $truePos = 0;
my $falsePos = 0;

my $tp = 0;
my $fp = 0;
my $tn = 0;
my $fn = 0;

my $rocArea = 0;

my $totalTrue = 0;
my $totalFalse = 0;

@sortedLines = sort byScore @lines;

for $line (@sortedLines) {
    $eval = evaluate ($line);

    if ($eval eq 'tp') {
	$totalTrue++;
    };
    if ($eval eq 'fp') {
	$totalFalse++;
    }
}

$totalCorrect = 0;
$totalTrials = 0;

for $line (@sortedLines) {

    $eval = evaluate ($line);
    $totalCorrect += correct ($line);
    $totalTrials++;

    $pred = prediction($line);

    if ($eval eq 'tp') {
	$truePos++;
    };
    if ($eval eq 'fp') {
	$rocArea += $truePos;
	$falsePos++;
    }
    $tp++ if ($eval eq 'tp' and $pred == "1"); 
    $tn++ if ($eval eq 'fp' and $pred == "-1"); 
    $fp++ if ($eval eq 'fp' and $pred == "1");
    $fn++ if ($eval eq 'tp' and $pred == "-1");
}

$rocArea = $rocArea / ($truePos * $falsePos);

die "Bad threshold -- all values are on one side of threshold.\n" 
    if ($tp + $fn == 0 or $tp + $fp == 0);

my $recall = $tp / ($tp + $fn);
my $prec = $tp / ($tp + $fp);

print "\nResults for $file: \n\n";
printf "Accuracy  %1.4f  (using threshold %2.2f) (%i/%i)\n", 
    ($totalCorrect/$totalTrials), $threshold, $totalCorrect, $totalTrials;
printf "Precision %1.4f  (using threshold %2.2f) (%i/%i)\n", 
    $prec, $threshold, $tp, ($tp + $fp);
printf "Recall    %1.4f  (using threshold %2.2f) (%i/%i)\n", 
    $recall, $threshold, $tp, ($tp + $fn);
printf "ROC area: %1.6f \n\n", $rocArea;

printf "Total of %i trials. \n\n", scalar (@sortedLines);


sub evaluate {
    my ($inLine) = @_;
    chomp ($inLine);
    my @fields = split "\t", $inLine; 
    my $truth = $fields[1];
    my $eval;
    $eval = 'tp' if ($truth eq '1');
    $eval = 'fp' if ($truth eq '-1');
    return $eval;
}

sub prediction {
    my ($inLine) = @_;
    chomp ($inLine);

    my @fields = split "\t", $inLine; 
    my $prediction = ($fields[0] > $threshold) ? 1 : -1;
    
    return $prediction;
}

sub correct {
    my ($inLine) = @_;
    chomp ($inLine);

    my @fields = split "\t", $inLine; 
    my $score = (($fields[0] > $threshold and $fields[1] == 1) or
		 ($fields[0] <= $threshold and $fields[1] == -1)) ? 1 : 0;
    
    return $score;
}

sub byScore {
    @aFields = split "\t", $a;
    @bFields = split "\t", $b;
    $bFields[0] <=> $aFields[0];
}
