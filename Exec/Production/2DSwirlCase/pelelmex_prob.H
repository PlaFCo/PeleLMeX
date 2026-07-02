#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>

#include <petscksp.h>

#include <harps.h>

struct MyProbParm : public ProbParmDefault
{

  amrex::Real uz = 0.0;
  amrex::Real ur = 0.0;
  amrex::Real inletcenter = 0.01;
  amrex::Real inletdelta = 0.002;

  amrex::Real standoff = 0.0;
  amrex::Real utheta = 0.0;
  amrex::Real pertmag = 0.0004;
  int meanFlowDir = 1;
  amrex::Real solidBody = 0.0;

  amrex::Real total_power = 0.0;
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
    amrex::Array4<amrex::Real> const& aux,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();
    

    AMREX_D_TERM(
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
      , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
      , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];);

    AMREX_D_TERM(
      const amrex::Real Lx = prob_hi[0] - prob_lo[0];
      , const amrex::Real Ly = prob_hi[1] - prob_lo[1];
      , const amrex::Real Lz = prob_hi[2] - prob_lo[2]);

    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    auto eos = pele::physics::PhysicsType::eos();
    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    amrex::Real pert;
    amrex::Real y1, y2;

    pert = 0.0;
    state(i, j, k, TEMP) = 300.0; //pmf_vals[0];
    molefrac[N2_ID] = 0.79;
    molefrac[O2_ID] = 0.21;
    eos.X2Y(molefrac, massfrac);

    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;
                 , state(i, j, k, VELY) = 0.01; //pmf_vals[1] * 1e-2;
                 , state(i, j, k, VELZ) = 0.0);

    amrex::Real P_cgs = prob_parm.P_mean * 10.0;

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
    amrex::Real r = (i + 0.5) * dx[0];

    // Start with a slow solid body swirl
    aux(i, j, k, aux.nComp()-1) = (0.01*r)*r*state(i,j,k,DENSITY);
    //aux(i, j, k, aux.nComp()-1) = (30*r*std::exp(-2*r*r*2e4))*r*state(i,j,k,DENSITY);                  // odd normal
    //aux(i, j, k, aux.nComp()-1) = (30*0.005*(r/0.005)/(1+(r/0.005)*(r/0.005)))*r*state(i,j,k,DENSITY); // decreasing velocity but increasing l_z
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal_aux(
    const amrex::Real x[AMREX_SPACEDIM],
    amrex::CellData<amrex::Real>& aux_ext,
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* pmf_data)
  {
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};
    int naux = aux_ext.ncomp-1;

    amrex::Real center = 0.0;
    center = prob_parm.inletcenter;
    amrex::Real deltac = deltac;
    deltac = prob_parm.inletdelta; 

    if (idir == 1 && sgn == 1) {
      aux_ext[naux] = 0.0;
      amrex::Real utheta = 0.0;
      amrex::Real ymin = center-deltac;
      amrex::Real ymax = center+deltac;
      amrex::Real r = x[0];
      if (x[0]<ymax && x[0]>ymin){
        utheta = prob_parm.utheta; //*std::cos(0.5*3.14159*(x[1]-center)/deltac); //prob_parm.utheta;
      }
      
      amrex::Real temp = 300;
      molefrac[N2_ID] = 0.79;
      molefrac[O2_ID] = 0.21;
      eos.X2Y(molefrac, massfrac);

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      eos.PYT2R(P_cgs, massfrac, temp, rho_cgs);
      amrex::Real density = rho_cgs * 1.0e3;
      
      aux_ext[naux] = density * r * utheta;
      //aux_ext[naux] = (20*r/0.01)*r*density;
    } else if (idir == 0 && sgn == -1){
      aux_ext[naux] = 0.0;

      if (x[1]>0.025 && x[1]<0.03){
        amrex::Real utheta = 0.0;
        amrex::Real temp = 300;
        molefrac[N2_ID] = 0.79;
        molefrac[O2_ID] = 0.21;
        eos.X2Y(molefrac, massfrac);

        amrex::Real rho_cgs, P_cgs, RhoH_temp;
        P_cgs = prob_parm.P_mean * 10.0;

        eos.PYT2R(P_cgs, massfrac, temp, rho_cgs);
        amrex::Real density = rho_cgs * 1.0e3;

        utheta = 20;
        aux_ext[naux] = density * 0.014 * utheta;
      }
    }
  }

  AMREX_GPU_DEVICE
  AMREX_FORCE_INLINE
  static void bcnormal(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::CellData<amrex::Real>& /*s_in*/,
    amrex::Real s_ext[NVAR],
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata,
    MyProbParm const& prob_parm,
    pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
  {
    const amrex::Real* prob_lo = geomdata.ProbLo();
    amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
    amrex::Real molefrac[NUM_SPECIES] = {0.0};
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    auto eos = pele::physics::PhysicsType::eos();

        
    if (idir == 1 && sgn == 1) {
      amrex::Real r = x[0];
      
      AMREX_D_TERM(
          s_ext[VELX] = 0.0;, s_ext[VELY] = 0.0;
          , s_ext[VELZ] = 0.0);
     
      amrex::Real center = prob_parm.inletcenter;
      amrex::Real deltac = prob_parm.inletdelta;
      amrex::Real ymin = center-deltac;
      amrex::Real ymax = center+deltac;


      if (x[0]<ymax && x[0]>ymin){
        AMREX_D_TERM(
            s_ext[VELX] = prob_parm.ur; //*std::cos(0.5*3.14159*(x[1]-center)/deltac);
          , s_ext[VELY] = prob_parm.uz; //*std::sin(0.5*3.14159*(x[1]-center)/deltac);
          , s_ext[VELZ] = 0.0);
      }
      
      s_ext[TEMP] = 300;// pmf_vals[0];
                        //
      molefrac[N2_ID] = 0.79;
      molefrac[O2_ID] = 0.21;
      eos.X2Y(molefrac, massfrac);

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;

      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] =
        RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
      }
    }else if (idir == 0){
      AMREX_D_TERM(
          s_ext[VELX] = 0.0;, s_ext[VELY] = 0.0;
          , s_ext[VELZ] = 0.0);

      if (x[1]>0.025 && x[1]<0.03){
        AMREX_D_TERM(
            s_ext[VELX] = -8; //*std::cos(0.5*3.14159*(x[1]-center)/deltac);
          , s_ext[VELY] = 0.0; //*std::sin(0.5*3.14159*(x[1]-center)/deltac);
          , s_ext[VELZ] = 0.0);
      }
      
      s_ext[TEMP] = 300;// pmf_vals[0];
                        //
      molefrac[N2_ID] = 0.79;
      molefrac[O2_ID] = 0.21;
      eos.X2Y(molefrac, massfrac);

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;

      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
      }
    }
}

  static void modify_ext_sources(
    amrex::Real time,
    amrex::Real /*dt*/,
    const amrex::MultiFab& state_old,
    const amrex::MultiFab& state_new,
    std::unique_ptr<amrex::MultiFab>& ext_src,
    const amrex::GeometryData& geomdata,
    const MyProbParm* prob_parm_d,
    const int sdcIter);
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
