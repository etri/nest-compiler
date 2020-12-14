#!/bin/bash

if [[ $# -ne 1 ]]
then
    echo "Direction: ./docker_build.sh CONTAINER_NAME"
    exit
fi

export NEST_COMPILER_HOME=$PWD/$1

docker container rm $1
