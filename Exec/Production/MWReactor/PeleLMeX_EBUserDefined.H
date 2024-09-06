#ifndef EBUSERDEFINED_H
#define EBUSERDEFINED_H

using namespace amrex;

#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
void
EBUserDefined(
  const Geometry& geom,
  const int required_coarsening_level,
  const int max_coarsening_level)
{
  // ParmParse geometry parameters
  amrex::ParmParse pp("eb2");

  amrex::Real cyl_R = 0.5;
  pp.query("cyl_R",cyl_R);

  amrex::Real cyl_dir = 1;
  pp.query("cyl_dir",cyl_dir);

  amrex::Real cyl_x = 0.0;
  pp.query("cyl_x",cyl_x);

  amrex::Real cyl_y = 0.0;
  pp.query("cyl_y",cyl_y);

  amrex::Real cyl_z = 0.0;
  pp.query("cyl_z",cyl_z);

  amrex::Real cylt_R = 0.1;
  pp.query("cylt_R",cylt_R);

  amrex::Real cylt_dir = 1;
  pp.query("cylt_dir",cylt_dir);

  amrex::Real cylt_x = 0.0;
  pp.query("cylt_x",cylt_x);

  amrex::Real cylt_y = 0.0;
  pp.query("cylt_y",cylt_y);

  amrex::Real rtan = 0.002;
  pp.query("rtan", rtan);

  amrex::Real ztan = 0.0;
  pp.query("ztan", ztan);

  amrex::Real zrad = 0.0;
  pp.query("zrad", zrad);
  
  amrex::Real zhi = 0.0;
  zhi = geom.ProbHi(2);

  // Build geometry pieces using EB2::* methods
  // First cylinder (true is the flag for internal flows) 
  EB2::CylinderIF cylinder1(cyl_R,  cyl_dir, 
		  {cyl_x,  cyl_y,  cyl_z},             true);

  EB2::CylinderIF cylinder4(cylt_R, cylt_dir, 
		  {cylt_x, cylt_y, zhi + ztan + zrad}, true);

  EB2::BoxIF boxA({0.0, cyl_R - rtan - 0.0035, zhi + ztan}, 
		  {0.02, cyl_R - 0.0035, zhi + ztan + rtan}, true);

  EB2::BoxIF boxB({-0.02, -cyl_R + 0.0035, zhi + ztan}, 
		  {0.0, -cyl_R + rtan + 0.0035, zhi + ztan + rtan}, true);

  // Make union ("makeIntersection") of geometries 
  auto MWreactor = EB2::makeIntersection(cylinder1, cylinder4, boxA, boxB);
  
  // Build your geometry shop using EB2::makeShop
  auto gshop = EB2::makeShop(MWreactor);
  
  // Build geom using EB2::Build
  EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
setEBState(
  const amrex::Real xEBface[AMREX_SPACEDIM],
  amrex::Real s_ext[NVAR],
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParm const& /*prob_parm*/)
{
  // if (xEBface[1] > 0.00) {
  //   s_ext[TEMP] = 300.0;
  // } else {
  //   s_ext[TEMP] = 300.0;
  // }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
setEBType(
  const amrex::Real* /*xEBface[AMREX_SPACEDIM]*/,
  amrex::Real& EBflagType,
  amrex::GeometryData const& /*geomdata*/,
  ProbParm const& /*prob_parm*/)
{
  // EBflagType = 1.0;
}
#endif
#endif
