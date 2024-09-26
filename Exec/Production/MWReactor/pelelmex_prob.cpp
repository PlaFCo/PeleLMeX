#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm() // NOLINT(readability-make-member-function-const)
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);
  pp.query("T_jet", prob_parm->T_jet);
  pp.query("T_tan", prob_parm->T_tan);
  pp.query("T_rad", prob_parm->T_rad);
  pp.query("V_jet", prob_parm->V_jet);
  pp.query("jet_rad", prob_parm->jet_rad);
  pp.query("bl_thickness", prob_parm->bl_thickness);
  pp.query("vmax", prob_parm->vmax);
  pp.query("R", prob_parm->R);
  pp.query("tanvmax", prob_parm->tanvmax);
  pp.query("vmean_tan", prob_parm->vmean_tan);
  pp.query("vmean_rad", prob_parm->vmean_rad);
  pp.query("rtrans", prob_parm->rtrans);
  pp.query("vmean_tan", prob_parm->vmean_tan);
  pp.query("vmean_rad", prob_parm->vmean_rad);
  pp.query("YCH4", prob_parm->YCH4);
  std::cout << "rtrans query->" << prob_parm->rtrans;

  amrex::ParmParse pp2("eb2");
  pp2.query("ztan", prob_parm->ztan);
  pp2.query("cylt_R", prob_parm->cylt_R);
  pp2.query("cyl_R", prob_parm->cyl_R);
  pp2.query("zrad", prob_parm->zrad);
  pp2.query("rtan", prob_parm->rtan);
  
  amrex::Vector<amrex::Real> spark_location = amrex::Vector<amrex::Real>(AMREX_SPACEDIM);
  amrex::Vector<amrex::Real> spark_radius = amrex::Vector<amrex::Real>(AMREX_SPACEDIM);

  amrex::ParmParse pps("peleLM.spark1");
  pps.get("power",   prob_parm->spark_power);
  pps.getarr("radius", spark_radius, 0, AMREX_SPACEDIM);

  pps.getarr("location", spark_location, 0, AMREX_SPACEDIM);
  prob_parm->spark_locationx = spark_location[0];
  prob_parm->spark_locationz = spark_location[2];
  prob_parm->spark_radiusx = spark_radius[0];
  prob_parm->spark_radiusz = spark_radius[2];
  prob_parm->spark_volume = 8.0 * 3.14 
	  * spark_radius[1] 
	  * spark_radius[2] 
	  * spark_radius[3] / 15.0;
}
