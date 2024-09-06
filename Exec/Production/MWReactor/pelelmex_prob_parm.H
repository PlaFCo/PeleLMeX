#ifndef PELELM_PROB_PARM_H
#define PELELM_PROB_PARM_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct ProbParm
{
  amrex::Real P_mean = 101325.0_rt;
  amrex::Real T_mean = 300.0_rt;
  amrex::Real vmax = -1.0_rt;
  amrex::Real vmean_tan = 0.0_rt;
  amrex::Real vmean_rad = 0.0_rt;
  amrex::Real R = 0.45_rt;
  amrex::Real cyl_R = 0.0;
  amrex::Real T_tan = 300.0;
  amrex::Real T_rad = 300.0;
  amrex::Real zrad = 0.0;
  amrex::Real ztan = 0.0;
};
#endif
