#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  std::string type;
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("standoff", PeleLM::prob_parm->standoff);
  pp.query("pertmag", PeleLM::prob_parm->pertmag);
  pp.query("solidBody", PeleLM::prob_parm->solidBody);
  pp.query("p", PeleLM::prob_parm->p);
  pp.query("sin", PeleLM::prob_parm->sin);

  pp.query("utheta", PeleLM::prob_parm->utheta);
  pp.query("uz", PeleLM::prob_parm->uz);
  pp.query("ur", PeleLM::prob_parm->ur);

  pp.query("inlet_center", PeleLM::prob_parm->inletcenter);
  pp.query("inlet_delta", PeleLM::prob_parm->inletdelta);

}

void
PeleLM::freeProbParm()
{
}
