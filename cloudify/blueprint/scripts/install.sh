#!/usr/bin/env bash

git_repo=$1
git_branch=$2
graph_url=$3
source_map=$8
stats_url=$9
topic=$4
job_id=$5
param_samples=$6
param_epsilon=$7

# DYNAMIC VARIABLES
cd $CURRENT_WORKDIR

git clone -b $git_branch $git_repo src

mkdir input
mkdir output

wget $graph_url -P input

if [ -z "$source_map" ]
    then
        echo "no source map as input"
    else
        wget $source_map -P input
fi

if [ -z "$stats_url" ]
    then
        echo "no stats file as input"
    else
        wget $stats_url -P input
fi


#write config file for execution

configfile="config.txt"
touch $configfile
echo "GIT_REPO=$git_repo" >> $configfile
echo "GIT_BRANCH=$git_branch" >> $configfile
echo "TOPIC=$topic" >> $configfile
echo "JOB_ID=$job_id" >> $configfile
echo "GRAPH_URL=$graph_url" >> $configfile
echo "SOURCE_MAP_URL=$source_map" >> $configfile
echo "STATS_URL=$stats_url" >> $configfile
echo "PARAM_SAMPLES=$param_samples" >> $configfile
echo "PARAM_EPSILON=$param_epsilon" >> $configfile


