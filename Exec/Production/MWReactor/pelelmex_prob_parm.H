#ifndef PELELM_PROB_PARM_H
#define PELELM_PROB_PARM_H

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct ProbParm
{
  amrex::Real P_mean = 101325.0_rt;
  amrex::Real T_mean = 300.0_rt;
  amrex::Real T_jet =  300.0_rt;
  amrex::Real V_jet = 1.0_rt;
  amrex::Real jet_rad = 0.25_rt;
  amrex::Real bl_thickness = 0.01_rt;
  amrex::Real vmax = -1.0_rt;
  amrex::Real tanvmax = 0.0_rt;
  amrex::Real vmean_tan = 0.0_rt;
  amrex::Real vmean_rad = 0.0_rt;
  amrex::Real R = 0.45_rt;
  amrex::Real rtrans = 0.5_rt;
  amrex::Real ztan = 0.0;
  amrex::Real cylt_R = 0.0;
  amrex::Real cyl_R = 0.0;
  amrex::Real T_tan = 300.0;
  amrex::Real T_rad = 300.0;
  amrex::Real zrad = 0.0;
  amrex::Real YCH4 = 0.0;
  amrex::Real rtan = 0.0;
  amrex::Real spark_power    = 0.0;
  amrex::Real spark_volume   = 0.0;
  amrex::Real spark_locationx = 0.0;
  amrex::Real spark_locationz = 0.0;
  amrex::Real spark_radiusx = 0.0;
  amrex::Real spark_radiusz = 0.0;

};
#endif
