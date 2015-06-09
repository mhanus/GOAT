#!/usr/bin/env bash

if [ "$#" -gt 2 ] || [ "$#" -eq 0 ]; then
    echo "Illegal number of parameters"
fi

problem=$1

if [ "$#" -eq 2 ]; then
    mesh=$2
else
    mesh=$problem
fi

dos2unix $problem/$mesh.msh

if [ $? -ne 0 ]; then
    echo "Dos2unix not performed."
fi

dolfin-convert $problem/$mesh.msh $problem/$mesh.xml

if [ $? -ne 0 ]; then
    echo "Fatal error occured - GOAT meshes not produced."
    exit -1
fi

python get_material_names.py $problem -m $mesh

if [ $? -ne 0 ]; then
    echo "Fatal error occured - GOAT meshes not produced."
    exit -1
fi