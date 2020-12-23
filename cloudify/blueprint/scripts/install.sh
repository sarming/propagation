#!/usr/bin/env bash

git_repo=$1
git_branch=$2
graph_url=$3
source_map=$4
stats_url=$5

# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR

pip install --user pandas
pip install --user scipy
pip install --user profilehooks
pip install --user networkx

git clone -b $git_branch $git_repo src

mkdir input
mkdir output

wget $graph_url -P input
wget $source_map -P input
wget $stats_url -P input

