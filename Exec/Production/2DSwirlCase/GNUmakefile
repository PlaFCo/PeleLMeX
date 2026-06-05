# AMReX
DIM = 2
COMP = gnu
PRECISION = DOUBLE
USE_EB = FALSE
USE_HYPRE = FALSE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_SYCL = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE
# PeleLMeX
USE_EFIELD = FALSE
USE_AXISWIRL = TRUE

# PelePhysics
Chemistry_Model = air
Eos_Model = Fuego
Transport_Model = Simple

PELE_HOME ?= ../../..

# HARPS - MW EFIELD SOLVER
HARPS_SRC_DIR = ../../../../power_coupling/harps/src

CEXE_sources += harps.cpp

VPATH_LOCATIONS   += $(HARPS_SRC_DIR)
INCLUDE_LOCATIONS += $(HARPS_SRC_DIR)

CEXE_sources += CartesianCoordinates.cpp
CEXE_sources += ConfigParser.cpp
CEXE_sources += MatrixSolver.cpp

# PETSc Dependencies
PETSC_DIR := $(shell spack location -i petsc)

INCLUDE_LOCATIONS += $(PETSC_DIR)/include
LIBRARY_LOCATIONS += $(PETSC_DIR)/lib
LIBRARIES         += -lpetsc

include $(PELE_HOME)/Exec/Make.PeleLMeX
