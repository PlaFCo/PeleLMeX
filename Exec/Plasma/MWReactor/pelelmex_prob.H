#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <pelelmex_prob_parm.H>
#include <PMFData.H>
#include <PelePhysics.H>

// Species
#ifndef N2_ID
#define N2_ID 1
#endif

#ifndef O2_ID
#define O2_ID 0
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pelelmex_initdata(
  int i,
  int j,
  int k,
  int /*is_incompressible*/,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& /*aux*/,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real vmax_l = 0;
  amrex::Real rz = 0;

  amrex::Real x[3] = {
    prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
    prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
    prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

  vmax_l = prob_parm.vmax;
  state(i, j, k, TEMP) = prob_parm.T_mean;
  rz = sqrt((x[0]*x[0]+x[1]*x[1]));

  if ( rz < prob_parm.cyl_R+0.0005 or x[2] > 0.12 ) {
   
    massfrac[O2_ID] = 0.0;
    massfrac[N2_ID] = 1.0-massfrac[O2_ID];
  
    if (rz<prob_parm.cyl_R) {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else if (x[0]>0.0){
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
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
  }
  else{
    massfrac[O2_ID] = 1.0;
    massfrac[N2_ID]  = 1.0-massfrac[O2_ID];
  
    if (rz<prob_parm.cyl_R) {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else if (x[0]>0.0){
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
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
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const int /*m_nAux*/,
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real X_main[NUM_SPECIES] = {0.0};
  amrex::Real Y_main[NUM_SPECIES] = {0.0};
  amrex::Real X_rad[NUM_SPECIES] = {0.0};
  amrex::Real Y_rad[NUM_SPECIES] = {0.0};
  amrex::Real rad_pos = 0.0;
  amrex::Real zradeff = 0.0;
  amrex::Real rho_cgs = 0.0;
  amrex::Real RhoH_temp = 0.0;
  amrex::Real p_cgs = prob_parm.P_mean * 10.0;
  amrex::Real massfrac_ch[NUM_SPECIES] = {0.0};

  // Top inlet
  Y_main[O2_ID] = 0.0;
  Y_main[N2_ID] = 1.0 - Y_main[O2_ID];

  // Bottom inlet
  Y_rad[O2_ID]= 1.0;
  Y_rad[N2_ID] = 1.0 - Y_rad[O2_ID];

  if (idir == 0) {
    // Fill defaults: no slip wall, sort of.
    zradeff = geomdata.ProbHi(2) + prob_parm.ztan - prob_parm.zrad;
    rad_pos = sqrt((x[1] - 0.0) * (x[1] - 0.0) + (x[2] + zradeff) * (x[2] + zradeff));
    if (x[2] < 0.15){
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_rad, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_rad, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_rad[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = sgn * prob_parm.vmean_rad;
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;
    }
    else{
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = sgn * prob_parm.vmean_tan;
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& beta,
  amrex::GeometryData const& /*geomdata*/,
  amrex::Box const& domainBox,
  const int dir,
  const int beta_comp,
  const int nComp)
{
  // amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
  // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
  // otherwise this routine could be called for other face diffusivity (Temp,
  // velocity, ...)
  if (beta_comp == 0 && nComp == NUM_SPECIES) {
    if (dir == 2 && k <= domainBox.smallEnd(dir)) {

      // Just over whole inlet
      for (int n = 0; n < nComp; n++) {
        beta(i, j, k, n) = 0.0;
      }
    }
  }
}

#endif
