#!/usr/bin/env bash

if [ "$#" -gt 2 ] || [ "$#" -eq 0 ]; then
    echo "Illegal number of parameters"
fi

mydir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
problem=$1
problem_dir=${mydir}/${problem}

if [ "$#" -eq 2 ]; then
    mesh=$2
else
    mesh=${problem}
fi

dos2unix ${problem_dir}/${mesh}.msh

if [ $? -ne 0 ]; then
    echo "Dos2unix not performed."
fi

dolfin-convert ${problem_dir}/${mesh}.msh ${problem_dir}/${mesh}.xml

if [ $? -ne 0 ]; then
    echo "Fatal error occured - GOAT meshes not produced."
    exit -1
fi

python ${mydir}/convert.py ${problem_dir}/${mesh} 2

if [ $? -ne 0 ]; then
    echo "Conversion of GOAT meshes to a parallel-friendly format unsuccessful."
    echo "Only sequential runs will be possible."
fi

python ${mydir}/get_material_names.py ${problem} -m ${mesh}

if [ $? -ne 0 ]; then
    echo "Fatal error occured - GOAT meshes not produced."
    exit -1
fi