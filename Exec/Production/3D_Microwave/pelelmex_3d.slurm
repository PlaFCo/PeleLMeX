#!/bin/bash
#SBATCH --job-name=3D_N2O2
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=64
#SBATCH --time=0-4:00:00

# Load/activate whatever environment you normally use here.
source ~/spack/share/spack/setup-env.sh
spack load petsc
spack load hypre
spack load openmpi
export LD_LIBRARY_PATH=$(spack location -i petsc)/lib:$(spack location -i hypre)/lib:$LD_LIBRARY_PATH

# make -j4 USE_HYPRE=TRUE      HYPRE_DIR=$(spack location -i hypre)      USE_SUNDIALS=TRUE      SUNDIALS_DIR=/home/P70094444/PeleLMeX/Submodules/PelePhysics/ThirdParty/INSTALL/gnu      AMREX_SUNDIALS_HOME=/home/P70094444/PeleLMeX/Submodules/PelePhysics/ThirdParty/INSTALL/gnu

# Run the code
mpirun -np 64 ./PeleLMeX3d.gnu.MPI.ex \
                input.3d-regt

