#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_init", prob_parm->T_init);
  pp.query("is_reactive", prob_parm->is_reactive);
  pp.query("has_coreflow", prob_parm->has_coreflow);
  pp.query("phi", prob_parm->phi);
  //pp.query("Y_fuel", prob_parm->Y_fuel);
  //pp.query("Y_oxid", prob_parm->Y_o2);
  pp.query("T_hot", prob_parm->T_hot);
  //pp.query("T_wall", prob_parm->Twall);
  pp.query("EBinflow_pmain", prob_parm->EBinflow_pm);
  pp.query("EBinflow_pcore", prob_parm->EBinflow_pc);
  pp.query("EBinflow_T", prob_parm->EBinflow_T);
  pp.query("EB_isoTwall", prob_parm->EB_isoTwall);
  pp.query("turb_mag", prob_parm->turb_mag);
  pp.query("use_parabolic_inflow", prob_parm->use_parabolic);
}

void
PeleLM::freeProbParm()
{
}
