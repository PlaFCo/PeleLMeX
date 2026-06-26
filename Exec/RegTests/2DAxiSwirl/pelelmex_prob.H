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
  amrex::Real p = 1.0;
  amrex::Real sin = 0.0;

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
    molefrac[N2_ID] = 1.0;
    eos.X2Y(molefrac, massfrac);

    AMREX_D_TERM(state(i, j, k, VELX) = 0.0;
                 , state(i, j, k, VELY) = 0.0; //pmf_vals[1] * 1e-2;
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
    const amrex::Real r = prob_lo[0] + (i + 0.5) * dx[0];
    const amrex::Real z =
      prob_lo[1] + (j + 0.5) * dx[1];

    const amrex::Real Lz = prob_hi[1] - prob_lo[1];

    const amrex::Real Omega = prob_parm.solidBody;
    const int mode = 1;
    const amrex::Real k_wave = 2.0 * Pi * mode / Lz;

    const amrex::Real q = Omega * prob_parm.sin*std::sin(k_wave * (z - prob_lo[1]));

    const amrex::Real utheta = std::pow(r, prob_parm.p) * q;

    aux(i, j, k, aux.nComp()-1) =
      state(i, j, k, DENSITY) * r * utheta;

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

  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
