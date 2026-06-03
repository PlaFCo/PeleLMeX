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
    amrex::Real r = (i + 0.5) * dx[0];

    // impose radial Gaussian profile in all auxiliaries
    aux(i, j, k, aux.nComp()-1) = 0.01*r*r*state(i,j,k,DENSITY);

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
      if (x[0]<ymax && x[0]>ymin){
        utheta = prob_parm.utheta; //*std::cos(0.5*3.14159*(x[1]-center)/deltac); //prob_parm.utheta;
      }
      amrex::Real r = x[0];
      
      amrex::Real temp = 300;
      molefrac[N2_ID] = 1.0;
      eos.X2Y(molefrac, massfrac);

      amrex::Real rho_cgs, P_cgs, RhoH_temp;
      P_cgs = prob_parm.P_mean * 10.0;

      eos.PYT2R(P_cgs, massfrac, temp, rho_cgs);
      amrex::Real density = rho_cgs * 1.0e3;
      
      aux_ext[naux] = density * r * utheta;
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
      molefrac[N2_ID] = 1.0;
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
    const int sdcIter)
  {
    const auto prob_lo = geomdata.ProbLo();
    const amrex::Real* dx = geomdata.CellSize();
    auto ext_src_rhoh_a = ext_src->arrays();
    auto ext_src_a = ext_src->arrays();
    auto const& state_old_a = state_old.const_arrays();
    auto const& state_new_a = state_new.const_arrays();

    const amrex::Real pi = 3.141592653589793;

    amrex::Real total_power  = 1*300;
    amrex::Real z_center     = 0.075;  // 7.5 cm 
    amrex::Real z_half_width = 0.025;  // 2.5 cm (spans 5.0 cm to 10.0 cm)
    amrex::Real r_max        = 0.008;  // 0.8 cm

    const amrex::Real P_0 = (3.0 * total_power) / (2.0 * pi * r_max * r_max * z_half_width);

    bool testing_SDC = 1;
    bool print_P_in = 1;
    int option = 2; // 1 for parabolic, 2 proportional to temperature
    
    if (time < 0.15){
      option = 1;
    } else {
      option = 2;
    }

    if (option == 1) {
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
          amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];

          bool inside_r = (r <= r_max);
          bool inside_z = (amrex::Math::abs(z - z_center) <= z_half_width);

          if (inside_r && inside_z) {
            amrex::Real r_norm = r / r_max;
            amrex::Real z_norm = (z - z_center) / z_half_width;

            ext_src_rhoh_a[box_no](i, j, k, RHOH) = P_0 * (1.0 - r_norm*r_norm) * (1.0 - z_norm*z_norm);
          } else {
            ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0; 
          }
        }
      );
    } else if (option == 2) {
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
          amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];

          bool inside_r = (r <= r_max);
          bool inside_z = (amrex::Math::abs(z - z_center) <= z_half_width);

          if (inside_r && inside_z) {
            amrex::Real r_norm = r / r_max;
            amrex::Real z_norm = (z - z_center) / z_half_width;

            amrex::Real Tg_temp = 0;

            if(sdcIter > 0) {
              Tg_temp = (state_old_a[box_no](i, j, k, TEMP)+state_new_a[box_no](i, j, k, TEMP))*0.5;
            } else {
              Tg_temp = state_old_a[box_no](i, j, k, TEMP);
            }

            ext_src_rhoh_a[box_no](i, j, k, RHOH) = P_0 * (1.0 - r_norm*r_norm) * (1.0 - z_norm*z_norm)/(1+std::exp((2500-Tg_temp) / 500.0));
          } else {
            ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0; 
          }
        }
      );

      amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
      amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename amrex::ReduceData<amrex::Real>::Type;

      for (amrex::MFIter mfi(*ext_src); mfi.isValid(); ++mfi) {
          const amrex::Box& bx = mfi.validbox();
          auto const& ext_src_a = ext_src->array(mfi);

          reduce_op.eval(bx, reduce_data,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
          {
              // Compute the cell-centered radius 'r' for this specific cell
              amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
              
              // Calculate the volume of this specific cylindrical ring shell
              amrex::Real cell_vol = 2.0 * pi * r * dx[0] * dx[1];
              
              // Return the energy scaled by this cell's individual volume
              return ext_src_a(i, j, k, RHOH) * cell_vol;
          });
      }

      // Gather the sum across all local GPU threads
      amrex::Real deposited_power = amrex::get<0>(reduce_data.value());

      // Sum the results across all MPI ranks (parallel processor communication)
      amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

      amrex::Real scaling_factor = total_power / deposited_power;

      // Now apply the scaling factor to the ext_src to ensure the total deposited power matches the target
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          ext_src_rhoh_a[box_no](i, j, k, RHOH) = std::min(ext_src_rhoh_a[box_no](i, j, k, RHOH)*scaling_factor, 3e8);
        }
      );
    }
    amrex::Gpu::streamSynchronize();
    
    if (print_P_in) {
      amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
      amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename amrex::ReduceData<amrex::Real>::Type;

      for (amrex::MFIter mfi(*ext_src); mfi.isValid(); ++mfi) {
          const amrex::Box& bx = mfi.validbox();
          auto const& ext_src_a = ext_src->array(mfi);

          reduce_op.eval(bx, reduce_data,
          [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
          {
              // Compute the cell-centered radius 'r' for this specific cell
              amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
              
              // Calculate the volume of this specific cylindrical ring shell
              amrex::Real cell_vol = 2.0 * pi * r * dx[0] * dx[1];
              
              // Return the energy scaled by this cell's individual volume
              return ext_src_a(i, j, k, RHOH) * cell_vol;
          });
      }

      // Gather the sum across all local GPU threads
      amrex::Real deposited_power = amrex::get<0>(reduce_data.value());

      // Sum the results across all MPI ranks (parallel processor communication)
      amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

      if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print()
          << "Deposited power = " << deposited_power
          << " ; target = " << total_power
          << "\n";
      }
    }
    
    if (testing_SDC){
      amrex::ParallelFor(
        *ext_src,
        [=, state_old_a=state_old_a, state_new_a=state_new_a]
        AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
          if (i == 16 && j == 256 && k == 0) {
            std::cout << "Inside cell (16, 256, 0) at SDC iteration " << sdcIter << std::endl;
            std::cout << "Tg_n: " << state_old_a[box_no](i, j, k, TEMP) << std::endl;
            std::cout << "Tg_k: " << state_new_a[box_no](i, j, k, TEMP) << std::endl;
          }
        }
      );
    }
    

  };
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
