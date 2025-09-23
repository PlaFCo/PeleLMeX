
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

struct MyProbParm : public ProbParmDefault
{
  // Shared params
  amrex::Real P_mean = 101325.0;
  amrex::Real T_init = 300.0;
  bool is_reactive = false;
  bool has_coreflow = false;
  amrex::Real phi = 0.2;
  //amrex::Real Y_fuel = 0.0445;
  //amrex::Real Y_o2 = 0.223;
  amrex::Real T_hot = 1800.0;
  //amrex::Real Twall = 300.0_rt;
  amrex::Real EBinflow_pm = -1.0;
  amrex::Real EBinflow_pc = -1.0;
  amrex::Real EBinflow_T = 300.0;
  amrex::Real EB_isoTwall = 300.0;
  amrex::Real turb_mag = 0.1;
  bool use_parabolic = false;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void initdata(
      int i,
      int j,
      int k,
      int is_incompressible,
      amrex::Array4<amrex::Real> const& state,
      amrex::Array4<amrex::Real> const& /*aux*/,
      amrex::GeometryData const& geomdata,
      MyProbParm const& prob_parm,
      pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
    {
      const amrex::Real* prob_lo = geomdata.ProbLo();
      const amrex::Real* dx = geomdata.CellSize();
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
      const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];

      auto eos = pele::physics::PhysicsType::eos();

      amrex::Real massfrac[NUM_SPECIES] = {0.0};
      if (!(is_incompressible == 0)) {
        return;
      }


      if (prob_parm.is_reactive) {
          // Start with air everywhere
          massfrac[O2_ID] = 0.233;
          massfrac[N2_ID] = 1.0 - massfrac[O2_ID];
          state(i, j, k, TEMP) = prob_parm.T_init;


          // Add fuel injection at top
          if (y > 0.048) {
              massfrac[H2_ID] = 0.00866;
              massfrac[O2_ID] = 0.233 * (1.0 - massfrac[H2_ID]);
              massfrac[N2_ID] = 0.767 * (1.0 - massfrac[H2_ID]);
          }

	  // Hot spot near tip
          if ((y > 0.030) && (y < 0.048) && (x > 0.202) && (x < 0.212)) {
              state(i, j, k, TEMP) = prob_parm.T_hot;
          }

          // Inital Velocity
          AMREX_D_TERM(state(i, j, k, VELX) = -10.0;
                 , state(i, j, k, VELY) = 0.0;
                 , state(i, j, k, VELZ) = 0.0);

     } else {
         // Cold flow (air only)
         massfrac[O2_ID] = 0.233;
         massfrac[N2_ID] = 1.0 - massfrac[O2_ID];
         state(i, j, k, TEMP) = prob_parm.T_init;

         // Velocity for cold flow - same as reactive
         AMREX_D_TERM(state(i, j, k, VELX) = -10.0;
                      , state(i, j, k, VELY) = 0.0;
                      , state(i, j, k, VELZ) = 0.0);
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

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void bcnormal(
      const amrex::Real x[AMREX_SPACEDIM],
      const amrex::CellData<amrex::Real>& /*s_in*/,
      amrex::Real s_ext[NVAR],
      const int idir,
      const int sgn,
      const amrex::Real /*time*/,
      amrex::GeometryData const& /*geomdata*/,
      MyProbParm const& prob_parm,
      pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
    {
      if (sgn == -1 && idir == 0) {
        if (prob_parm.has_coreflow){
          amrex::Real massfrac[NUM_SPECIES] = {0.0};

          auto eos = pele::physics::PhysicsType::eos();

          if (prob_parm.use_parabolic) {
            amrex::Real p_parab = prob_parm.EBinflow_pc*1e-7;
            amrex::Real h_parab = 0.006; // based on 0.012 m channel inlet
            amrex::Real k_parab = -h_parab*h_parab/(4.0 * p_parab);
            amrex::Real norm_y = (0.032 - x[1]); // channel top at 0.08 --> gets noorm_y bet 0 and 0.01 
            amrex::Real velo_xdir = 1.0 + (norm_y-h_parab)*(norm_y-h_parab)/(4.0 * p_parab) + k_parab; 
            AMREX_D_TERM(s_ext[VELX] = - velo_xdir;, s_ext[VELY] = 0.0;
                         , s_ext[VELZ] = 0.0);
          } else {
            amrex::Real velo_xdir = 10.2; 
            AMREX_D_TERM(s_ext[VELX] = - velo_xdir;, s_ext[VELY] = 0.0;
                         , s_ext[VELZ] = 0.0);
          }

          // input air
	  massfrac[O2_ID] = 0.233;
          massfrac[N2_ID] = 1.0 - massfrac[O2_ID];
	  

          s_ext[TEMP] = prob_parm.EBinflow_T;

          amrex::Real rho_cgs, P_cgs, RhoH_temp;
          P_cgs = prob_parm.P_mean * 10.0;

          eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
          s_ext[DENSITY] = rho_cgs * 1.0e3;

          eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
          s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

          for (int n = 0; n < NUM_SPECIES; n++) {
            s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
          }
        } else {
            //amrex::Abort("Core flow deactivated: change BC to wall !");
        }


      }
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void bcnormal_eb(
      const amrex::Real xEBface[AMREX_SPACEDIM],
      const amrex::Real* /*xEBfaceNormal[AMREX_SPACEDIM]*/,
      const amrex::CellData<const amrex::Real>& s_in,
      amrex::Real s_ext[NVAR],
      const amrex::Real time,
      amrex::GeometryData const& geomdata,
      MyProbParm const& prob_parm)
    {

      constexpr amrex::Real Pi = 3.14159265359;
      // Isothermal wall

	if ((xEBface[0] < 0.203) && (xEBface[1] > 0.043) && (xEBface[1] < 0.05)) {
        amrex::Real x = xEBface[0];  // in meters
        //amrex::Real T_K = -3.748e+04 * x * x * x
        //                + 5.051e+03 * x * x
        //                + 9.580e+02 * x
        //                + 487.680;   // Kelvin
	amrex::Real T_K = 300; //Kelvin

        s_ext[TEMP] = T_K;  // set temperature in K
	//#ifndef AMREX_USE_GPU
        // to check if temp function is applied
        //amrex::Print() << "[BC DEBUG] Wall BC applied at x=" << x
        //               << " m, T=" << T_K << " K\n";
	//#endif

    }


      // Inflow
      if ((xEBface[0] >= 0.039) && (xEBface[0] <= 0.202) && (xEBface[1] >= 0.050 - 1e-12)){

          const amrex::Real y = xEBface[1];

          auto eos = pele::physics::PhysicsType::eos();

          // compute parabolic velo profile -- channel height is 10 mm
          if (prob_parm.use_parabolic) {
            amrex::Real p_parab = prob_parm.EBinflow_pm*1e-7;//-6.164e-07; // PUT THIS AS INPUT !! 
            amrex::Real h_parab = 0.005; // based on 0.01 m channel inlet
            amrex::Real k_parab = -h_parab*h_parab/(4.0 * p_parab);
            amrex::Real norm_y = (0.08 - y); // channel top at 0.08 --> gets noorm_y bet 0 and 0.01 
            amrex::Real velo_xdir = 1.0 + (norm_y-h_parab)*(norm_y-h_parab)/(4.0 * p_parab) + k_parab; 
            //amrex::Real r1 = 3.0e-3;//std::rand() / (RAND_MAX);
            //amrex::Real r2 = 7.0e-3;//std::rand() / (RAND_MAX);
            //r1 = std::max(0.1, r1);  //random number bet 0.1-1
            //r2 = std::max(0.1, r2);  //random number bet 0.1-1
            //amrex::Real velo_pert  = prob_parm.turb_mag * velo_xdir * std::sin(2.0*Pi*(time-0.0058)/r1) * std::sin(2.0*Pi*norm_y/(2*h_parab));
            //amrex::Real velo_pert2 = prob_parm.turb_mag * velo_xdir * std::sin(2.0*Pi*time/r2) * std::sin(3.0*2.0*Pi*norm_y/(2*h_parab));
            //amrex::Real velo_pert  = prob_parm.turb_mag * velo_xdir * std::sin(2.0*Pi*norm_y/(2*h_parab));
            //amrex::Real velo_pert2 = prob_parm.turb_mag * velo_xdir * std::sin(3.0*2.0*Pi*norm_y/(2*h_parab));
            //amrex::Real velo = std::max(0.0, velo_xdir + velo_pert + velo_pert2);
            //amrex::Real velo = velo_xdir + velo_pert + velo_pert2;
            AMREX_D_TERM(s_ext[VELX] = velo_xdir;, s_ext[VELY] = 0.0;
                         , s_ext[VELZ] = 0.0);
          } else {
            amrex::Real velo_xdir = 10.2;
            AMREX_D_TERM(s_ext[VELX] = velo_xdir;, s_ext[VELY] = 0.0;
                         , s_ext[VELZ] = 0.0);
          }
          //std::cout << velo_xdir << " " <<  velo << std::endl;
          // phi of 0p2 - fuel is always pure H2
          amrex::Real massfrac[NUM_SPECIES] = {0.0};
          amrex::Real Xt[NUM_SPECIES] = {0.0};
          amrex::Real X_H2 = prob_parm.phi;
          amrex::Real X_O2 = 0.5;
          amrex::Real X_N2 = 3.76*X_O2;
          Xt[O2_ID] = X_O2 /(X_H2 + X_O2 + X_N2);
          Xt[H2_ID] = X_H2 /(X_H2 + X_O2 + X_N2); 
          Xt[N2_ID] = 1.0 - Xt[O2_ID] - Xt[H2_ID]; 
          eos.X2Y(Xt, massfrac);

	  amrex::Real P_cgs = prob_parm.P_mean * 10.0;
    
          s_ext[TEMP] = prob_parm.EBinflow_T;
          amrex::Real rho_cgs, RhoH_temp;
          eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
          s_ext[DENSITY] = rho_cgs * 1.0e3;
    
          eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
          s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];
    
          for (int n = 0; n < NUM_SPECIES; n++) {
              s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
          }
      }
    }
 
    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    static void bctype_eb(
      const amrex::Real xEBface[AMREX_SPACEDIM],
      int& EBflagType,
      amrex::Real& EBfaceFrac,
      amrex::GeometryData const& /*geomdata*/,
      MyProbParm const& prob_parm)
    {
      //for now all other walls are ADIAB - not true for top one, also have T profiles there
      if ((xEBface[0] >= 0.039) && (xEBface[0] <= 0.202) && (xEBface[1] >= 0.050 - 1e-12)) {
        EBflagType = pelelmex::BCTypeEB::inflow;
      }
      if ((xEBface[0] < 0.203) && (xEBface[1] > 0.043) && (xEBface[1] < 0.05)) {
        EBflagType = pelelmex::BCTypeEB::wall_isot;
      }
    }

    static void patchFlowVariables(
      const amrex::Geometry& geom,
      MyProbParm const& lprobparm,
      amrex::MultiFab& a_mf)
    {

      amrex::Print() << "\nPatching flow variables..";
      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo =
        geom.ProbLoArray();
      const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx =
        geom.CellSizeArray();

      for (amrex::MFIter mfi(a_mf, amrex::TilingIfNotGPU()); mfi.isValid();
           ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& temp_arr = a_mf.array(mfi, TEMP);
        amrex::ParallelFor(
          bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            amrex::Real x[3] = {
              prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
              prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
              prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

            amrex::ignore_unused(x);
            /*User can define how to patch flow variables here.*/
            temp_arr(i, j, k) = lprobparm.T_hot;

          });
      }

      amrex::Print() << "Done\n";
    }
    static void EBUserDefined(
        const amrex::Geometry& geom,
        const int required_coarsening_level,
        const int max_coarsening_level)
     {
    // ParmParse your geometry parameters
      amrex::ParmParse pp("eb2");  //
      bool use_sharp = false;      //default if not mentioned in input file
      pp.query("sharp", use_sharp);

    // Define all the boxes (fluid regions)
    amrex::RealArray outer_lo = {AMREX_D_DECL(0.00, 0.00, 0.00)};
    amrex::RealArray outer_hi = {AMREX_D_DECL(0.24, 0.06, 0.031)}; // Outer domain

    amrex::RealArray big_lo = {AMREX_D_DECL(0.00, 0.00, 0.00)};
    amrex::RealArray big_hi = {AMREX_D_DECL(0.172, 0.046, 0.031)}; // Left bottom

    amrex::RealArray main_lo = {AMREX_D_DECL(0.040, 0.05, 0.00)};
    amrex::RealArray main_hi = {AMREX_D_DECL(0.202, 0.06, 0.031)}; // top inlet

    amrex::RealArray center_lo = {AMREX_D_DECL(0.172, 0.016, 0.00)};
    amrex::RealArray center_hi = {AMREX_D_DECL(0.202, 0.046, 0.031)}; // Center block

    amrex::RealArray core_lo = {AMREX_D_DECL(0.172, 0.00, 0.00)};
    amrex::RealArray core_hi = {AMREX_D_DECL(0.24, 0.012, 0.031)}; // bottom inlet

    amrex::RealArray tipbox_lo = {AMREX_D_DECL(0.195, 0.046, 0.00)};
    amrex::RealArray tipbox_hi = {AMREX_D_DECL(0.202, 0.05, 0.031)}; // tip block


    // Create BoxIFs for all regions
    amrex::EB2::BoxIF outer_box(outer_lo, outer_hi, false); // Outer domain (solid)
    amrex::EB2::BoxIF big_box(big_lo, big_hi, false); // Big duct (fluid)
    amrex::EB2::BoxIF main_box(main_lo, main_hi, false); // Main duct (fluid)
    amrex::EB2::BoxIF center_box(center_lo, center_hi, false); // Center block (fluid)
    amrex::EB2::BoxIF core_box(core_lo, core_hi, false); // Core duct (fluid)
    amrex::EB2::BoxIF tip_box(tipbox_lo, tipbox_hi, false); //tip block

    // Create the half-sphere (fluid)
    amrex::Real radius = 0.022;
    amrex::Array<amrex::Real, AMREX_SPACEDIM> center{AMREX_D_DECL(0.202, 0.038, 0.015)};

    #if AMREX_SPACEDIM == 2
    amrex::EB2::SphereIF sphere(radius, center, false); // Fluid inside sphere
    // Create a half-plane to cut the sphere
    amrex::EB2::PlaneIF cutter({AMREX_D_DECL(0.202, 0.038, 0.0)},
                              {AMREX_D_DECL(1.0, 0.0, 0.0)}); // Vertical plane at x=0.175
    auto half_sphere = amrex::EB2::makeIntersection(sphere, cutter);
    #else
    //amrex::EB2::CylinderIF cylinder(radius, 100.0, 2, center, false);
    //auto half_sphere = cylinder; // In 3D, we just use the full cylinder
    #endif

    amrex::Real angle_degrees = 60.0; // angle from x axis
    amrex::Real angle_radians = angle_degrees * 3.14159265 / 180.0;
    amrex::EB2::PlaneIF tip_cutter(
	{AMREX_D_DECL(0.202, 0.047, 0.0)}, // Point on the cutting plane
        {AMREX_D_DECL(cos(angle_radians), sin(angle_radians), 0.0)}); // Normal vector at 60deg

    // Apply the tip cutter to the tip_box
    auto sharp_main_box = amrex::EB2::makeIntersection(tip_box, tip_cutter);


    if (use_sharp) {

    // Combine all fluid regions
    auto fluid_boxes = amrex::EB2::makeUnion(big_box, main_box, sharp_main_box, center_box, core_box);
    auto all_fluid = amrex::EB2::makeUnion(fluid_boxes, half_sphere);

    // Subtract fluid regions from outer domain to get solid geometry
    auto solid_geometry = amrex::EB2::makeDifference(outer_box, all_fluid);

    // Build your geometry shop
    auto gshop = amrex::EB2::makeShop(solid_geometry);

    // Build geom using EB2::Build
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
    amrex::Print() << "sharp geometry built successfully." << std::endl;
    } else {
      auto fluid_boxes = amrex::EB2::makeUnion(big_box, main_box, center_box, core_box);
      auto all_fluid = amrex::EB2::makeUnion(fluid_boxes, half_sphere);

      auto solid_geometry = amrex::EB2::makeDifference(outer_box, all_fluid);
      auto gshop = amrex::EB2::makeShop(solid_geometry);
      amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
      amrex::Print() << "blunt geometry built successfully." << std::endl;
      }
  }
};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
