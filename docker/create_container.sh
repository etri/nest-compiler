#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "Direction: ./docker_build.sh SHARED_DIR CONTAINER_NAME"
    exit
fi

export NEST_COMPILER_HOME=$PWD/$1



docker build  -t nest_compiler:0.0 $NEST_COMPILER_HOME/utils/docker/
docker run -it -v $NEST_COMPILER_HOME:/root/dev/nestc_workspace --name=$2 nest_compiler:0.0 /bin/bash
