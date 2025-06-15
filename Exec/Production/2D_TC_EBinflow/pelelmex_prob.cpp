#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>
#include <math.h>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);

  amrex::ParmParse pp2("eb2");
  pp2.query("rtan", prob_parm->rtan);
  pp2.query("sphere_radius", prob_parm->cyl_R);

  amrex::Real tanflux_sccm = 0.0;
  amrex::Real cubicmeter_minute = 0.0;
  pp.query("tanflux_sccm",tanflux_sccm);
  if (tanflux_sccm>0) {
    cubicmeter_minute = tanflux_sccm*1e-6*prob_parm->T_mean*100000/273.15/prob_parm->P_mean;
    prob_parm->vmean_tan = cubicmeter_minute/4.0/(3.14159*prob_parm->rtan*prob_parm->rtan)/60.0; // 4 injectors
  }
}

void
PeleLM::freeProbParm()
{
}	
	
