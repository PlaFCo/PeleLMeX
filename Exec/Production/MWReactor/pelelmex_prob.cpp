#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);
  pp.query("T_tan", prob_parm->T_tan);
  pp.query("T_rad", prob_parm->T_rad);
  pp.query("vmax", prob_parm->vmax);
  pp.query("R", prob_parm->R);
  pp.query("vmean_tan", prob_parm->vmean_tan);
  pp.query("vmean_rad", prob_parm->vmean_rad);

  amrex::ParmParse pp2("eb2");
  pp2.query("cyl_R", prob_parm->cyl_R);
  pp2.query("zrad", prob_parm->zrad);
  pp2.query("ztan", prob_parm->ztan);
}
