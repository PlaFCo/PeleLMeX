#!/bin/bash
#SBATCH --job-name=N2_10slm_600W
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=8
#SBATCH --time=48:00:00

# Load/activate whatever environment you normally use here.
source ~/spack/share/spack/setup-env.sh
spack load petsc
spack load /k6o2x7p
spack load openmpi %gcc@12.2.0
export LD_LIBRARY_PATH=$(spack location -i petsc)/lib:$LD_LIBRARY_PATH

# Run the code
mpirun -np 8 ./PeleLMeX2d.gnu.MPI.AXISWIRL.ex \
                input.2d-regt

