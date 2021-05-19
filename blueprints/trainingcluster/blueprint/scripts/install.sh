#!/usr/bin/env bash

git_repo=$1
git_branch=$2
graph_type=$3
source_map=$4
stats_url=$5
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

mkdir input
mkdir output


if [ "$topic" == "neos" ] || [ "$topic" == "fpoe" ]
    then
        topic_date="${topic}_20200311"
elif  [ "$topic" == "schalke" ] || [ "$topic" == "bvb" ]
    then
        topic_date="${topic}_20200409"
elif  [ "$topic" == "vegan" ]
    then
        topic_date="${topic}_20200407"
else
    echo "not a valid topic"
    topic_date=""
fi

if [ "$graph_type" == "outer" ]
    then
        graph_url="https://ckan.hidalgo-project.eu/dataset/02ef431b-7fb5-4fe5-9ea2-828e2038b395/resource/e369c14a-4732-497d-a5f3-0807e874108c/download/anonymized_outer_graph_${topic_date}.adjlist"
elif [ "$graph_type" == "inner" ]
    then
        graph_url="https://ckan.hidalgo-project.eu/dataset/89b3941a-23a3-4073-868c-717fafa7e50a/resource/ebc224ac-1a59-4ee4-9154-9be4cd552741/download/anonymized_inner_graph_${topic_date}.adjlist"
else
    graph_url=""
    echo "not a valid graph_url"
fi

graph_file_out="anonymized_${graph_type}_graph_${topic}.adjlist"

tweets_url="https://ckan.hidalgo-project.eu/dataset/7ca9e0d7-3ec0-4c13-8d6e-9aeb37e50c8e/resource/3a309fc4-bc99-477f-ba19-8c73b947b8ae/download/sim_features_${topic_date}.csv"
tweets_file_out="sim_features_${topic}.csv"

#download files
wget -O $graph_file_out -N $graph_url -P input
wget -O $tweets_file_out -N $tweets_url -P input

if [ -z "$source_map" ] || [ "$source_map" == "default" ]
    then
        echo "no source map as input"
        source_map_file=""
    else
        wget -N $source_map -P input
        source_map_file="$(basename -- $source_map)"
fi

if [ -z "$stats_url" ] || [ "$stats_url" == "default" ]
    then
        echo "no stats file as input"
        stats_file=""
    else
        wget -N $stats_url -P input
        stats_file="$(basename -- $stats_url)"
fi


#write config file for execution
configfile="config.txt"
if [ -f "$configfile" ]
    then
        echo "config is already present"
    else
        touch $configfile
        echo "GIT_REPO=$git_repo" >> $configfile
        echo "GIT_BRANCH=$git_branch" >> $configfile
        echo "TOPIC=$topic" >> $configfile
        echo "JOB_ID=$job_id" >> $configfile
        echo "TASKS_PER_NODE=$tasks_per_node" >> $configfile
        echo "CURRENT_WORKDIR=$CURRENT_WORKDIR" >> $configfile
        echo "GRAPH_FILE=$graph_file_out" >> $configfile
        echo "SOURCEMAP_FILE=$source_map_file" >> $configfile
        echo "STATS_FILE=$stats_file" >> $configfile
        echo "PARAM_SAMPLES=$param_samples" >> $configfile
        echo "PARAM_EPSILON=$param_epsilon" >> $configfile
        echo "SIM_FEATURES=$sim_features" >> $configfile
        echo "SIM_SOURCES=$sim_sources" >> $configfile
        echo "SIM_SAMPLES=$sim_samples" >> $configfile
fi
