#!/usr/bin/env bash

git_repo=$1
git_branch=$2
graph_type=$3
sourcemap_url=$4
stats_url=$5
topic=$6
job_id=$7
tasks_per_node=$8
param_samples=$9
param_epsilon=${10}
sim_features=${11}
sim_sources=${12}
sim_samples=${13}
ckan_api_key=${14}

# dynamic variable
cd $CURRENT_WORKDIR

git clone -b $git_branch $git_repo src

mkdir input
mkdir output

if [ "$topic" == "neos" ] || [ "$topic" == "fpoe" ] || [ "$topic" == "political" ]; then
  graph_date="20201110"
elif [ "$topic" == "schalke" ] || [ "$topic" == "bvb" ] || [ "$topic" == "vegan" ]; then
  graph_date="20201110"
else
  echo "not a valid topic"
  graph_date=""
fi

if [ "$graph_type" == "outer" ]; then
  graph_url="https://ckan.hidalgo-project.eu/dataset/1ee0bda5-86bb-4adf-9179-4af087467b86/resource/b2bda13c-7f53-4742-85f5-9458dbdf6700/download/anon_graph_outer_${topic}_${graph_date}.metis.gz"
elif [ "$graph_type" == "inner" ]; then
  graph_url="https://ckan.hidalgo-project.eu/dataset/543a67f1-7648-43c0-be84-566e38b0fa54/resource/5eb0bb39-3bc6-4875-a084-93010e34f6e7/download/anon_graph_inner_${topic}_${graph_date}.metis"
else
  graph_url=""
  echo "graph_url not valid"
fi

graph_file="anon_graph_${graph_type}_${topic}.metis"

tweets_url="https://ckan.hidalgo-project.eu/dataset/a71570f1-45fc-43ee-be7f-65f189f5e7b9/resource/8b913a69-beed-4ad3-ba44-f53f7b41c1c7/download/sim_features_${topic}_${graph_date}.csv"
tweets_file="sim_features_${topic}.csv"

#download files
if [ "$graph_type" == "outer" ]; then
  wget -nc $graph_url -O input/$graph_file".gz"
  gzip -dk input/$graph_file".gz"

elif [ "$graph_type" == "inner" ]; then
  wget -nc $graph_url -O input/$graph_file
else
  echo "not a valid graph type"
fi

wget -N $tweets_url -O input/$tweets_file

if [ -z "$sourcemap_url" ] || [ "$sourcemap_url" == "default" ]; then
  echo "no source map as input"
  sourcemap_file=""
else
  sourcemap_file="$(basename -- $sourcemap_url)"
  curl -H"Authorization: $ckan_api_key" $sourcemap_url --output input/$sourcemap_file
fi

if [ -z "$stats_url" ] || [ "$stats_url" == "default" ]; then
  echo "no stats file as input"
  stats_file=""
else
  stats_file="$(basename -- $stats_url)"
  curl -H"Authorization: $ckan_api_key" $stats_url --output input/$stats_file
fi

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
