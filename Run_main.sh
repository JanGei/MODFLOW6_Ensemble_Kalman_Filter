#!/bin/bash
#PBS -N Test
#PBS -l nodes=1:ppn=10:geo2
#PBS -q geo2
#PBS -l walltime=55:00:00

eval "$(conda shell.bash hook)"
conda activate Flopy

# create a status text file to indicate the start of the job
echo 'started' > .statusStarted.txt

# go back to original working directory
#cd $PBS_O_WORKDIR/
cd /home-link/epajg01/Python/NeckarDIS

# load matlab-HOW TO LOAD PYTHON PACKAGES?
# module load math/matlab/2019b

# run Python file
python "main.py"

# create a status text file to indicate the end of the job
echo 'finished' >.statusFinished.txt

# print some diagnostic output
echo $PBS_O_WORKDIR/
echo $PBS_O_HOST
echo $PBS_QUEUE
echo $PBS_NODEFILE
echo $TMPDIR
