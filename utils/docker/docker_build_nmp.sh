#!/bin/bash

if [[ $# -ne 3 ]]
then
    echo "Direction: ./docker_build.sh NEST_COMPILER_HOME_DIR BUILD_DIR_NAME NMP_HOME"
    exit
fi

export NEST_COMPILER_HOME=$PWD/$1

echo $NEST_COMPILER_HOME
mkdir -p $NEST_COMPILER_HOME/$2

export NMP_HOME=$PWD/$3 

docker build  -t nest_compiler:0.0 $NEST_COMPILER_HOME/utils/docker/
docker create -i -v $NMP_HOME:/root/dev/nmp_workspace  -v $NEST_COMPILER_HOME:/root/dev/nestc_workspace --name=nest_container nest_compiler:0.0 /bin/bash
docker start nest_container
echo "#######Docker submodule update#######"
docker exec nest_container sh -c "cd nestc_workspace; ls;git submodule update --init --recursive"
echo "#######Docker Nest Compiler Cmaked#######"
docker exec nest_container sh -c "cd nestc_workspace; ls;rm -rf $2/*; cd $2; ls; cmake -G Ninja ../ -DCMAKE_BUILD_TYPE=Release -DNESTC_WITH_NMP=ON"
echo "#######Docker Nest Compiler Build#######"
docker exec nest_container sh -c "cd nestc_workspace; cd $2; ninja all;"
