#!/usr/bin/env bash
git_repo=$1
git_branch=$2
graph_type=$3
sourcemap_file=$4
stats_file=$5
topic=$6
job_id=$7
tasks_per_node=$8
param_samples=$9
param_epsilon=${10}
sim_features=${11}
sim_sources=${12}
sim_samples=${13}

# dynamic variable
cd $CURRENT_WORKDIR

git clone -b $git_branch $git_repo src

mkdir output

if [ "$graph_type" == "outer" ]; then
  gzip -dk "input/anon_graph_${graph_type}_${topic}.metis.gz"
fi

graph_file="anon_graph_${graph_type}_${topic}.metis"

#write config file for execution
configfile="config.txt"
if [ -f "$configfile" ]; then
  echo "config is already present"
else
  touch $configfile
  echo "GIT_REPO=$git_repo" >>$configfile
  echo "GIT_BRANCH=$git_branch" >>$configfile
  echo "TOPIC=$topic" >>$configfile
  echo "JOB_ID=$job_id" >>$configfile
  echo "TASKS_PER_NODE=$tasks_per_node" >>$configfile
  echo "CURRENT_WORKDIR=$CURRENT_WORKDIR" >>$configfile
  echo "GRAPH_FILE=$graph_file" >>$configfile
  echo "SOURCEMAP_FILE=$sourcemap_file" >>$configfile
  echo "STATS_FILE=$stats_file" >>$configfile
  echo "PARAM_SAMPLES=$param_samples" >>$configfile
  echo "PARAM_EPSILON=$param_epsilon" >>$configfile
  echo "SIM_FEATURES=$sim_features" >>$configfile
  echo "SIM_SOURCES=$sim_sources" >>$configfile
  echo "SIM_SAMPLES=$sim_samples" >>$configfile
fi
