#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#endif

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMFData.H>
#include <PelePhysics.H>

#include <math.h>


struct MyProbParm : public ProbParmDefault
{
  amrex::Real P_mean = 101325.0; 
  amrex::Real T_mean = 300.0;
  amrex::Real rtan = 0.0;
  amrex::Real cyl_R = 0.0;
  amrex::Real vmean_tan = 0.0;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void initdata(
    int i,
    int j,
    int k,
    int /*is_incompressible*/,
    amrex::Array4<amrex::Real> const& state,
    amrex::Array4<amrex::Real> const& /*aux*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) = -1.0);

    state(i, j, k, TEMP) = prob_parm.T_mean;
    amrex::Real P_cgs = prob_parm.P_mean * 10.0;
    massfrac[N2_ID]  = 1.0;
    
    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
 
    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);
 
    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
    }
  }


  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal_eb(
    const amrex::Real xEBface[AMREX_SPACEDIM],
    const amrex::Real* /*xEBfaceNormal[AMREX_SPACEDIM]*/,
    const amrex::CellData<const amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const amrex::Real /*time*/,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm)
  {
    auto eos = pele::physics::PhysicsType::eos();
    // Default IsoT temperature
    s_ext[TEMP] = 300.0;

    // Inflow EB
	  //amrex::Real zhi  = geomdata.ProbHi(2);
    //amrex::Real z_center = zhi+prob_parm.ztan;
    amrex::Real xy_center = prob_parm.cyl_R * 0.5;
    amrex::Real xgeom_center = geomdata.ProbLo(0) + 0.5 * (geomdata.ProbHi(0) - geomdata.ProbLo(0));
    amrex::Real ygeom_center = geomdata.ProbLo(1) + 0.5 * (geomdata.ProbHi(1) - geomdata.ProbLo(1));

    amrex::Real r_yz_xpl = std::sqrt( (xEBface[1] - ( - xy_center ))*(xEBface[1] - ( - xy_center )) );//+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_yz_xmn = std::sqrt( (xEBface[1] - xy_center)*(xEBface[1] - xy_center) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_xz_ypl = std::sqrt( (xEBface[0] - xy_center)*(xEBface[0] - xy_center) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_xz_ymn = std::sqrt( (xEBface[0] - ( - xy_center ))*(xEBface[0] - ( - xy_center )) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));

    amrex::Real rad = prob_parm.rtan;
    amrex::Real Pcsg = prob_parm.P_mean * 10.0;
    amrex::Real rho_cgs{0.0};
    amrex::Real RhoH_temp{0.0};

    amrex::Real Y_main[NUM_SPECIES] = {0.0};
    Y_main[N2_ID] = 1.0;

    // Inflow  +X	  
    if ( (r_yz_xpl <= rad) && (xEBface[0] < xgeom_center) ) {
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(Pcsg, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = prob_parm.vmean_tan;
      s_ext[VELY] = 0.0;
    }
    // Inflow  -X	  
    if ( (r_yz_xmn <= rad) && (xEBface[0] > xgeom_center) ) {
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(Pcsg, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = -prob_parm.vmean_tan;
      s_ext[VELY] = 0.0;
    }
    // Inflow  +Y	  
    if ( (r_xz_ypl <= rad) && (xEBface[1] < ygeom_center) ) {
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(Pcsg, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = 0.0;
      s_ext[VELY] = prob_parm.vmean_tan;
    }
    // Inflow  -Y	  
    if ( (r_xz_ymn <= rad) && (xEBface[1] > ygeom_center) ) {
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(Pcsg, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = 0.0;
      s_ext[VELY] = -prob_parm.vmean_tan;
    }
  }


  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bctype_eb(
    const amrex::Real xEBface[AMREX_SPACEDIM],
    int& EBflagType,
    amrex::Real& EBfaceFrac,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm)
  {
    
    // IsoT temperature
    EBflagType = pelelmex::BCTypeEB::wall_isot;
    EBfaceFrac = 1.0; 

    // Inflow EB
	  //amrex::Real zhi  = geomdata.ProbHi(2);
    //amrex::Real z_center = zhi+prob_parm.ztan;
    amrex::Real xy_center = prob_parm.cyl_R * 0.5;
    amrex::Real xgeom_center = geomdata.ProbLo(0) + 0.5 * (geomdata.ProbHi(0) - geomdata.ProbLo(0));
    amrex::Real ygeom_center = geomdata.ProbLo(1) + 0.5 * (geomdata.ProbHi(1) - geomdata.ProbLo(1));

    amrex::Real r_yz_xpl = std::sqrt( (xEBface[1] - ( - xy_center ))*(xEBface[1] - ( - xy_center )) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_yz_xmn = std::sqrt( (xEBface[1] - xy_center)*(xEBface[1] - xy_center) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_xz_ypl = std::sqrt( (xEBface[0] - xy_center)*(xEBface[0] - xy_center) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));
    amrex::Real r_xz_ymn = std::sqrt( (xEBface[0] - ( - xy_center ))*(xEBface[0] - ( - xy_center )) ); //+ (xEBface[2] - z_center)*(xEBface[2] - z_center));

    amrex::Real rad = prob_parm.rtan;

    // Inflow  +X	  
    if ( (r_yz_xpl <= rad) && (xEBface[0] < xgeom_center) ) {
      EBflagType = pelelmex::BCTypeEB::inflow;
      EBfaceFrac = 1.0;
    }
     // Inflow  -X	  
     if ( (r_yz_xmn <= rad) && (xEBface[0] > xgeom_center) ) {
      EBflagType = pelelmex::BCTypeEB::inflow;
      EBfaceFrac = 1.0;
     }
     // Inflow  +Y	  
     if ( (r_xz_ypl <= rad) && (xEBface[1] < ygeom_center) ) {
      EBflagType = pelelmex::BCTypeEB::inflow;
      EBfaceFrac = 1.0;
     }
     // Inflow  -Y	  
     if ( (r_xz_ymn <= rad) && (xEBface[1] > ygeom_center) ) {
      EBflagType = pelelmex::BCTypeEB::inflow;
      EBfaceFrac = 1.0;
     }
  }


  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void zero_visc(
    int i,
    int j,
    int k,
    amrex::Array4<amrex::Real> const& beta,
    amrex::GeometryData const& geomdata,
    amrex::Box const& domainBox,
    const int dir,
    const int beta_comp,
    const int nComp,
    MyProbParm const& /*prob_parm*/)
  {
     amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
  }

};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
