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
USE_MWSOLVER = FALSE
USE_SUNDIALS = TRUE

# PelePhysics
Chemistry_Model = airThermal		#airThermal
Eos_Model = Fuego
Transport_Model = Simple

PELE_HOME ?= ../../..

ifeq ($(USE_MWSOLVER),TRUE)
  HARPS_SRC_DIR = ../../../../harps/src
  CEXE_sources += harps.cpp CartesianCoordinates.cpp ConfigParser.cpp MatrixSolver.cpp
  VPATH_LOCATIONS   += $(HARPS_SRC_DIR)
  INCLUDE_LOCATIONS += $(HARPS_SRC_DIR)

  PETSC_DIR := $(shell spack location -i petsc)
  INCLUDE_LOCATIONS += $(PETSC_DIR)/include
  LIBRARY_LOCATIONS += $(PETSC_DIR)/lib
  LIBRARIES         += -lpetsc
endif

include $(PELE_HOME)/Exec/Make.PeleLMeX
