#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>
#include <math.h>

#include <AMReX_PlotFileUtil.H>

#ifdef PELE_USE_MWSOLVER

#include "lut_plasma_props.h"

#endif

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  pp.query("P_mean", prob_parm->P_mean);
  pp.query("T_mean", prob_parm->T_mean);
  pp.query("T_wall", prob_parm->T_wall);
  
  pp.query("power", prob_parm->power);
  pp.query("power_time", prob_parm->power_time);
  pp.query("power_shape", prob_parm->power_shape);
  pp.query("power_radiusx", prob_parm->power_radiusx);
  pp.query("power_radiusy", prob_parm->power_radiusy);
  pp.query("power_radiusz", prob_parm->power_radiusz);
  pp.query("z_center_waveguide", prob_parm->z_center_waveguide);
  pp.query("power_shape", PeleLM::prob_parm->power_shape);

  #ifdef PELE_USE_MWSOLVER

  pp.query("EM_t_start", PeleLM::prob_parm->EM_t_start);
  pp.query("flag_solve_EM", PeleLM::prob_parm->flag_solve_EM);
  pp.query("flag_optimize_reflector", PeleLM::prob_parm->flag_optimize_reflector);
  pp.query("plot_harps_int", PeleLM::prob_parm->plot_harps_int);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);
  pp.query("normalize_power", PeleLM::prob_parm->normalize_power);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);

  std::string plot_harps_file_str; 
  pp.query("plot_harps_file", plot_harps_file_str); std::strncpy(prob_parm->plot_harps_file.data(), plot_harps_file_str.c_str(), 64); prob_parm->plot_harps_file[63] = '\0'; // Ensure null-termination

  #endif

  amrex::ParmParse pp2("eb2");
  pp2.query("ztan", prob_parm->ztan);
  pp2.query("rperp", prob_parm->rperp);
  pp2.query("cyl_R", prob_parm->cyl_R);
  pp2.query("zperp", prob_parm->zperp);
  pp2.query("rtan", prob_parm->rtan);
  pp2.query("ntan", prob_parm->ntan);
  pp2.query("nperp", prob_parm->nperp);

  amrex::Real tanflux_sccm = 0.0;
  amrex::Real cubicmeter_minute = 0.0;
  pp.query("tanflux_sccm", tanflux_sccm);
  if (tanflux_sccm > 0) {
    cubicmeter_minute = tanflux_sccm * 1e-6 * prob_parm->T_mean * 100000 / 273.15 / prob_parm->P_mean;
    prob_parm->vmean_tan = cubicmeter_minute / prob_parm->ntan / (prob_parm->rtan * prob_parm->rtan * M_PI) / 60.0;
  }

  amrex::Real perpflux_sccm = 0.0;
  pp.query("perpflux_sccm", perpflux_sccm);
  if (perpflux_sccm > 0) {
    cubicmeter_minute = perpflux_sccm * 1e-6 * prob_parm->T_mean * 100000 / 273.15 / prob_parm->P_mean;
    prob_parm->vmean_rad = cubicmeter_minute / prob_parm->nperp / (M_PI * prob_parm->rperp * prob_parm->rperp) / 60.0;
  }
  pp.query("x_O2_main", prob_parm->x_O2_main);
  pp.query("x_O2_quench", prob_parm->x_O2_quench);

  amrex::ParmParse ppc("control");
  pp.query("power_radiusz", prob_parm->rz);
  ppc.query("Time_ctrl", prob_parm->Time_ctrl);
  ppc.query("Tctrl_on", prob_parm->Tctrl_on);
  ppc.query("Tctrl_verbose", prob_parm->Tctrl_verbose);
  ppc.query("Tctrl_int", prob_parm->Tctrl_int);
  ppc.query("T_target", prob_parm->T_target);
  ppc.query("tau", prob_parm->tau);
  ppc.query("rz_min", prob_parm->rz_min);
  ppc.query("rz_max", prob_parm->rz_max);
  ppc.query("drz_max", prob_parm->drz_max);
  ppc.query("Kp", prob_parm->Kp);
  ppc.query("Ki", prob_parm->Ki);

  amrex::ParmParse psp("sponge");
  psp.query("on", prob_parm->sponge_on);
  psp.query("zbeg", prob_parm->sponge_zbeg);
  psp.query("zend", prob_parm->sponge_zend);
  psp.query("r", prob_parm->sponge_r);
  psp.query("sigma_para", prob_parm->sponge_sigma_para);
  psp.query("sigma_perp", prob_parm->sponge_sigma_perp);
  psp.query("shape", prob_parm->sponge_shape);

  amrex::Real Ri = 0.013;
  amrex::Real Ro = 0.015;
  amrex::Real ks = 1.4;
  prob_parm->hExt = 10.0;

  amrex::Real Tinf = 300.0;
  amrex::Real kg = 0.03;

  prob_parm->sigma = 5.670374419e-8;
  prob_parm->eps = 0.85;

  amrex::Real dxmin = 0.0004375;
  amrex::Real dn = 0.5 * dxmin;

  amrex::Real Rcond = Ri * std::log(Ro / Ri) / ks;
  amrex::Real Rconv = Ri / (Ro * prob_parm->hExt);

  prob_parm->Gout = 1.0 / (Rcond + Rconv);
  prob_parm->Ggas = kg / dn;
  prob_parm->Tinf = Tinf;
  prob_parm->Ri = Ri;
  prob_parm->Ro = Ro;
  prob_parm->Rcond = Rcond;
  prob_parm->wall_a = prob_parm->Ggas / (prob_parm->Ggas + prob_parm->Gout);
  prob_parm->wall_b = prob_parm->Gout * prob_parm->Tinf / (prob_parm->Ggas + prob_parm->Gout);
}

#ifdef PELE_USE_MWSOLVER

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int find_nonuniform_index(const double* arr, int size, double val) {
    int lo = 0, hi = size - 2;
    int ans = 0;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] <= val) {
            ans = mid;
            lo = mid + 1; // Look right
        } else {
            hi = mid - 1; // Look left
        }
    }
    return ans;
}

#endif

void ProblemSpecificFunctions::modify_ext_sources(
  amrex::Real time,
  amrex::Real /*dt*/,
  const amrex::MultiFab& state_old,
  const amrex::MultiFab& state_new,
  std::unique_ptr<amrex::MultiFab>& ext_src,
  const amrex::GeometryData& geomdata,
  const MyProbParm* prob_parm_d,
  const int sdcIter,
  const int nSDCmax)
{
  const auto prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  auto ext_src_rhoh_a = ext_src->arrays();
  auto ext_src_a = ext_src->arrays();
  auto const& state_old_a = state_old.const_arrays();
  auto const& state_new_a = state_new.const_arrays();

  const amrex::Real pi = 3.141592653589793;
 
  const int power_shape = prob_parm_d->power_shape;
  const amrex::Real power = prob_parm_d->power;
  const amrex::Real power_time = prob_parm_d->power_time;
  amrex::Real total_power = 0.0;
  const amrex::Real radiusx = prob_parm_d->power_radiusx;
  const amrex::Real radiusy = prob_parm_d->power_radiusy;
  amrex::Real radiusz = prob_parm_d->power_radiusz;
  const amrex::Real z_center_waveguide = prob_parm_d->z_center_waveguide;
  
  
  const amrex::GpuArray<amrex::Real, 3> ic = {{
    (0.0- prob_lo[0]) / dx[0] - 0.5,
    (0.0- prob_lo[1]) / dx[1] - 0.5,
    (z_center_waveguide - prob_lo[2]) / dx[2] - 0.5
  }};

  auto addsponge = prob_parm_d->sponge_on;                                                                                                                                               
  const amrex::Real zbeg = prob_parm_d->sponge_zbeg;
  const amrex::Real zend = prob_parm_d->sponge_zend;
  const amrex::Real sponge_r = prob_parm_d->sponge_r;
  const amrex::Real sponge_sigma_perp = prob_parm_d->sponge_sigma_perp;
  const amrex::Real sponge_sigma_para = prob_parm_d->sponge_sigma_para;
  const amrex::Real sponge_T_sigma = prob_parm_d->sponge_T_sigma;
  const amrex::Real sponge_T_ref = prob_parm_d->sponge_T_ref;
  const int sponge_shape = prob_parm_d->sponge_shape;

  if (addsponge) {
    ParallelFor(
      *ext_src,
      [=, dx=dx, prob_lo=prob_lo,
       state_old_a=state_old_a, ext_src_a=ext_src_a]
      AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {    
        amrex::Real x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
        amrex::Real y = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];
        amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5) * dx[2];
        amrex::Real r2 = x * x + y * y;
        bool in_r = (r2 <= sponge_r * sponge_r);
        bool in_z = (z >= zbeg && z <= zend);
        if (in_r && in_z) {
          amrex::Real eta = (zend - z) / amrex::max(zend - zbeg, 1.0e-12);
          //          amrex::Real eta = (z - zbeg) / amrex::max(zend - zbeg, 1.0e-12);
          //          eta = amrex::max(0.0, amrex::min(1.0, eta));
          amrex::Real ramp = eta;
          if (sponge_shape == 2) {
            ramp = eta * eta;
          } else if (sponge_shape == 3) {
            ramp = eta * eta * eta;
          }
          amrex::Real rho = state_old_a[box_no](i, j, k, DENSITY);
          amrex::Real u   = state_old_a[box_no](i, j, k, VELX);
          amrex::Real v   = state_old_a[box_no](i, j, k, VELY);
          amrex::Real w   = state_old_a[box_no](i, j, k, VELZ);
          amrex::Real T   = state_old_a[box_no](i, j, k, TEMP);
          ext_src_a[box_no](i, j, k, VELX) += -rho * ramp * sponge_sigma_perp * u;
          ext_src_a[box_no](i, j, k, VELY) += -rho * ramp * sponge_sigma_perp * v;
          ext_src_a[box_no](i, j, k, VELZ) += -rho * ramp * sponge_sigma_para * amrex::max(0.0, w);
          if (sponge_T_sigma > 0.0) {
            ext_src_a[box_no](i, j, k, TEMP) +=
              -ramp * sponge_T_sigma * (T - sponge_T_ref);
          }
        }  
      });  
  }  

  total_power = power * amrex::min(time/power_time, 1.0);
  if (radiusx <= 0.0|| radiusy <= 0.0|| radiusz <= 0.0|| total_power == 0.0) {
    amrex::ParallelFor(
      *ext_src,
      [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
      {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0;
      });
    amrex::Gpu::streamSynchronize();
    return;
  }

  const amrex::Box bbox = ext_src->boxArray().minimalBox();

  const int ilo = bbox.smallEnd(0);
  const int ihi = bbox.bigEnd(0);
  const int jlo = bbox.smallEnd(1);
  const int jhi = bbox.bigEnd(1);
  const int klo = bbox.smallEnd(2);
  const int khi = bbox.bigEnd(2);
  
  static amrex::Real normalization_factor = 1;
  bool do_harps = 0; 
  int stabilizing_runs = 10;  // Number of initial runs to stabilize the solution before using HARPS
  static int step_counter = 0;
  static int harps_counter = 0;

  if (sdcIter == 0) step_counter++;

  #ifdef PELE_USE_MWSOLVER

  if (prob_parm_d->flag_solve_EM && time > prob_parm_d->EM_t_start){
    do_harps = 1;
  } else{
    do_harps = 0;
  }

  #endif
  
  if (do_harps == 0 || harps_counter <= stabilizing_runs) {
    if (power_shape == 1) { // Parabolic Profile
      // Exact discrete normalization for
      // max(1-x^2/rx^2,0) * max(1-y^2/ry^2,0) * max(1-z^2/rz^2,0)

      amrex::Real Sx = 0.0;
      amrex::Real Sy = 0.0;
      amrex::Real Sz = 0.0;

      for (int i = ilo; i <= ihi; ++i) {
        const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
        Sx += amrex::max(1.0- (x * x) / (radiusx * radiusx), 0.0);
      }
      for (int j = jlo; j <= jhi; ++j) {
        const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
        Sy += amrex::max(1.0- (y * y) / (radiusy * radiusy), 0.0);
      }
      for (int k = klo; k <= khi; ++k) {
        const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
        Sz += amrex::max(1.0- (z * z) / (radiusz * radiusz), 0.0);
      }

      const amrex::Real volume = dx[0] * dx[1] * dx[2] * Sx * Sy * Sz;

      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
          const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
          const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

          const amrex::Real fx =
            amrex::max(1.0- (x * x) / (radiusx * radiusx), 0.0);
          const amrex::Real fy =
            amrex::max(1.0- (y * y) / (radiusy * radiusy), 0.0);
          const amrex::Real fz =
            amrex::max(1.0- (z * z) / (radiusz * radiusz), 0.0);

          const amrex::Real src =
            (volume > 0.0) ? (total_power / volume) * fx * fy * fz : 0.0;

          ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
        });
    }
    else if (power_shape == 2) { // Constant in z, parabolic in x and y
      amrex::Real Sx = 0.0;
      amrex::Real Sy = 0.0;
      amrex::Real Sz = 0.0;

      for (int i = ilo; i <= ihi; ++i) {
        const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
        Sx += amrex::max(1.0 - (x * x) / (radiusx * radiusx), 0.0);
      }
      for (int j = jlo; j <= jhi; ++j) {
        const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
        Sy += amrex::max(1.0 - (y * y) / (radiusy * radiusy), 0.0);
      }
      // Flat top (constant) in z: 1.0 if |z| <= radiusz, else 0.0
      for (int k = klo; k <= khi; ++k) {
        const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
        if (std::abs(z) <= radiusz) Sz += 1.0;
      }

      const amrex::Real volume = dx[0] * dx[1] * dx[2] * Sx * Sy * Sz;

      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
            const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
            const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

            const amrex::Real fx = amrex::max(1.0 - (x * x) / (radiusx * radiusx), 0.0);
            const amrex::Real fy = amrex::max(1.0 - (y * y) / (radiusy * radiusy), 0.0);
            const amrex::Real fz = (amrex::Math::abs(z) <= radiusz) ? 1.0 : 0.0;

            const amrex::Real src = (volume > 0.0) ? (total_power / volume) * fx * fy * fz : 0.0;

            ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
        }
      );
    }
    else if (power_shape == 3){  // hollow profile (testing)
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
            const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
            const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
            
            const amrex::Real radius = sqrt(x*x + y*y);
            const amrex::Real R_r_sq = radiusx * radiusy;
            const amrex::Real R_z_sq = radiusz * radiusz;

            // Unified ellipsoidal distance metric
            const amrex::Real metric = (radius * radius) / R_r_sq + (z * z) / R_z_sq;

            const amrex::Real src = std::exp(-(metric - 1.0) * (metric - 1.0));
            ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
        }
      );

      // Normalize
      amrex::Real total_src = ext_src->sum(RHOH, false) * dx[0] * dx[1] * dx[2];
      amrex::Real norm_factor = (total_src > 0.0) ? (total_power / total_src) : 0.0;
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
            ext_src_rhoh_a[box_no](i, j, k, RHOH) *= norm_factor;
        }
      );
    }
    else if (power_shape == 0) { // Constant profile
      // Exact discrete normalization for support:
      // (x/rx)^2 + (y/ry)^2 <= 1  AND  (z/rz)^2 <= 1

      long long ncell = 0;

      for (int k = klo; k <= khi; ++k) {
        const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];
        const amrex::Real rz = (z * z) / (radiusz * radiusz);

        if (rz <= 1.0) {
          for (int j = jlo; j <= jhi; ++j) {
            const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
            const amrex::Real yy = (y * y) / (radiusy * radiusy);

            for (int i = ilo; i <= ihi; ++i) {
              const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
              const amrex::Real xx = (x * x) / (radiusx * radiusx);

              if (xx + yy <= 1.0) {
                ++ncell;
              }
            }
          }
        }
      }

      const amrex::Real volume =
        static_cast<amrex::Real>(ncell) * dx[0] * dx[1] * dx[2];

      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          const amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
          const amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
          const amrex::Real z = (static_cast<amrex::Real>(k) - ic[2]) * dx[2];

          const amrex::Real rxy =
            (x * x) / (radiusx * radiusx) +
            (y * y) / (radiusy * radiusy);
          const amrex::Real rz =
            (z * z) / (radiusz * radiusz);

          const amrex::Real inside =
            (rxy <= 1.0&& rz <= 1.0) ? 1.0: 0.0;

          const amrex::Real src =
            (volume > 0.0) ? (total_power / volume) * inside : 0.0;

          ext_src_rhoh_a[box_no](i, j, k, RHOH) = src;
        });
    }
    else {
      amrex::ParallelFor(
        *ext_src,
        [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept
        {
          ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0;
        });
    }
  }

  #ifdef PELE_USE_MWSOLVER

  if (do_harps == 1) {
    harps_counter++;

    double max_power = 10e8;

    // HARPS 2D Setup Geometry
    double y_c = 0.056;
    double R_in = 0.0135;
    double a = 0.08636;
    double L_z = 0.140;
    double z_0 = z_center_waveguide + (L_z/2 - 0.02);
    
    static double y_reflector = 1e6;
    const bool optimize_reflector_enabled = (prob_parm_d->flag_optimize_reflector == 1);
    const bool should_optimize_reflector  = optimize_reflector_enabled && (step_counter % 500 == 10) && (sdcIter == 0);

    std::vector<std::tuple<int, int, int>> plasma_locations;
    std::vector<double> plasma_ne;
    std::vector<double> plasma_mu_re;
    std::vector<double> plasma_mu_im;
    std::vector<double> plasma_pabs;
    std::vector<double> plasma_efield;

    static std::unique_ptr<amrex::MultiFab> E_field_mf;
    static std::unique_ptr<amrex::MultiFab> E_field_new_mf;
    static std::unique_ptr<amrex::MultiFab> n_e_old_mf;

    if (!E_field_mf || E_field_mf->boxArray() != ext_src->boxArray() || E_field_mf->DistributionMap() != ext_src->DistributionMap()) {
      E_field_mf = std::make_unique<amrex::MultiFab>(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
      E_field_new_mf = std::make_unique<amrex::MultiFab>(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
      n_e_old_mf = std::make_unique<amrex::MultiFab>(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
      
      E_field_mf->setVal(5000.0); 
      E_field_new_mf->setVal(5000.0);
      n_e_old_mf->setVal(1e13);
    }

    amrex::MultiFab E_field_raw_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    E_field_raw_mf.setVal(5000.0);
    auto E_field_raw_a = E_field_raw_mf.arrays();

    amrex::MultiFab n_e_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab mu_re_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab mu_im_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab real_cond_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);

    auto n_e_arr = n_e_mf.arrays();
    auto mu_re_arr = mu_re_mf.arrays();
    auto mu_im_arr = mu_im_mf.arrays();
    auto real_cond_arr = real_cond_mf.arrays();
    auto E_field_a = E_field_mf->arrays();
    auto E_field_new_a = E_field_new_mf->arrays();
    auto n_e_old_arr = n_e_old_mf->arrays();

    // For Plotting Purposes
    amrex::MultiFab Te_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab E_over_N_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab ne_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab E_field_plot_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    Te_mf.setVal(300.0); E_over_N_mf.setVal(1.0); ne_mf.setVal(1e11); E_field_plot_mf.setVal(500.0);

    auto Te_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? Te_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    auto E_over_N_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? E_over_N_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    auto ne_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? ne_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    auto E_field_plot_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? E_field_plot_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    
    // Compute conductivity related quantities (n_e, mu)
    amrex::Real E_ion_O2 = 12.07*1.60218e-19; // J
    amrex::Real E_ion_N2 = 15.58*1.60218e-19; // J
    amrex::Real E_ion_NO = 9.26*1.60218e-19;  // J
    amrex::Real E_ion_N = 14.53*1.60218e-19;  // J
    amrex::Real E_ion_O = 13.14*1.60218e-19;  // J
    amrex::Real h_planck = 6.62607015e-34;      // J*s
    amrex::Real k_B = 1.380649e-23;             // J/K
    amrex::Real m_e = 9.10938356e-31;           // kg
    amrex::Real atomic_mass_unit = 1.66053906660e-27; // kg
    
    PlasmaLookupTable lut;
    amrex::Real alpha_under_relaxation = 0.5;

    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5)*dx[2];
      if (z < z_0 && z > z_0 - L_z){
        amrex::Real Tg_mid;
        amrex::Real n_O2_mid; amrex::Real n_N2_mid; amrex::Real n_NO_mid = 0; amrex::Real n_O_mid = 0; amrex::Real n_N_mid = 0;
        amrex::Real n_e_O2; amrex::Real n_e_N2; amrex::Real n_e_NO = 0; amrex::Real n_e_O = 0; amrex::Real n_e_N = 0;
        amrex::Real E_field_mid;

        // Mid Point Rule (using n and n+1,k)
        if(sdcIter > 0) {
          Tg_mid = (state_old_a[box_no](i, j, k, TEMP) + state_new_a[box_no](i, j, k, TEMP))*0.5;
          n_O2_mid = (state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + O2_ID) + state_new_a[box_no](i, j, k, DENSITY)*state_new_a[box_no](i, j, k, FIRSTSPEC + O2_ID))*0.5 / (32.0*atomic_mass_unit);
          n_N2_mid = (state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + N2_ID) + state_new_a[box_no](i, j, k, DENSITY)*state_new_a[box_no](i, j, k, FIRSTSPEC + N2_ID))*0.5 / (28.0*atomic_mass_unit);
          n_NO_mid = (state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + NO_ID) + state_new_a[box_no](i, j, k, DENSITY)*state_new_a[box_no](i, j, k, FIRSTSPEC + NO_ID))*0.5 / (30.0*atomic_mass_unit);
          n_O_mid = (state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + O_ID) + state_new_a[box_no](i, j, k, DENSITY)*state_new_a[box_no](i, j, k, FIRSTSPEC + O_ID))*0.5 / (16.0*atomic_mass_unit);
          n_N_mid = (state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + N_ID) + state_new_a[box_no](i, j, k, DENSITY)*state_new_a[box_no](i, j, k, FIRSTSPEC + N_ID))*0.5 / (14.0*atomic_mass_unit);
          E_field_mid = (E_field_a[box_no](i, j, k) + E_field_new_a[box_no](i, j, k))*0.5;
        } else {
          Tg_mid = state_old_a[box_no](i, j, k, TEMP);
          n_O2_mid = state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + O2_ID) / (32.0*atomic_mass_unit);
          n_N2_mid = state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + N2_ID) / (28.0*atomic_mass_unit);
          n_NO_mid = state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + NO_ID) / (30.0*atomic_mass_unit);
          n_O_mid = state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + O_ID) / (16.0*atomic_mass_unit);
          n_N_mid = state_old_a[box_no](i, j, k, DENSITY)*state_old_a[box_no](i, j, k, FIRSTSPEC + N_ID) / (14.0*atomic_mass_unit);
          E_field_mid = E_field_a[box_no](i, j, k);
        }
        amrex::Real gas_density = prob_parm_d->P_mean/(k_B*Tg_mid); // Ideal Gas

        amrex::Real E_over_N = 1e21*E_field_mid/gas_density; // Field calculated by previous HARPS iteration [Td]

        // Interpolate from LoKI-B table for N2 and O2 at 1013 mbar
        PlasmaProperties elec_props = lut.eval((2*n_O2_mid + n_O_mid)/(2*n_O2_mid + n_O_mid + 2*n_N2_mid + n_N_mid), E_over_N, Tg_mid);

        amrex::Real Te_mid = elec_props.te*11604.52;
        mu_re_arr[box_no](i, j, k) = elec_props.mobi_real;
        mu_im_arr[box_no](i, j, k) = elec_props.mobi_imag;

        // Saha equation
        n_e_O2 = sqrt(n_O2_mid*pow(2*pi*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_O2/(k_B*Te_mid)));
        n_e_N2 = sqrt(n_N2_mid*pow(2*pi*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_N2/(k_B*Te_mid)));
        n_e_NO = sqrt(n_NO_mid*pow(2*pi*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_NO/(k_B*Te_mid)));
        n_e_N = sqrt(n_N_mid*pow(2*pi*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_N/(k_B*Te_mid)));
        n_e_O = sqrt(n_O_mid*pow(2*pi*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_O/(k_B*Te_mid)));

        amrex::Real ne_saha = sqrt(n_e_O2*n_e_O2 + n_e_N2*n_e_N2 + n_e_NO*n_e_NO + n_e_N*n_e_N + n_e_O*n_e_O); //*(1/(1 + pow((Te_mid/Tg_mid - 1)/(1.8 - 1), 4)));  // Non equilibrium correction factor
        
        amrex::Real ne_old = n_e_old_arr[box_no](i,j,k);
        amrex::Real ne_new = ne_old*pow(ne_saha / ne_old, alpha_under_relaxation); // Under relaxation

        n_e_arr[box_no](i,j,k) = ne_new;
        n_e_old_arr[box_no](i,j,k) = ne_new;

        if(i == 16 && j == 16 && k == 333 && prob_parm_d->harps_verbose > 1) {
          std::cout << "(16,16,333) Tg: " << Tg_mid << ", E: " << E_field_mid << "; Te: " << Te_mid << "; ne: " << n_e_arr[box_no](i,j,k) << std::endl;
        }

        if((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)){
          Te_arr[box_no](i, j, k) = Te_mid;
          E_over_N_arr[box_no](i, j, k) = E_field_mid; //E_over_N;
          ne_arr[box_no](i, j, k) = n_e_arr[box_no](i,j,k);
          E_field_plot_arr[box_no](i, j, k) = E_field_mid;
        }
      } else {
        n_e_arr[box_no](i, j, k)   = 1e11;
        mu_re_arr[box_no](i, j, k) = 0.0;
        mu_im_arr[box_no](i, j, k) = 0.0;

        if((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) Te_arr[box_no](i, j, k) = state_old_a[box_no](i, j, k, TEMP);
      }
    });

    if((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)){
      const amrex::BoxArray& ba = ext_src->boxArray();
      const amrex::DistributionMapping& dmap = ext_src->DistributionMap();
      amrex::MultiFab plotdata(ba, dmap, 4, 0);
      amrex::MultiFab::Copy(plotdata, Te_mf, 0, 0, 1, 0);
      amrex::MultiFab::Copy(plotdata, E_over_N_mf, 0, 1, 1, 0);
      amrex::MultiFab::Copy(plotdata, ne_mf, 0, 2, 1, 0);
      amrex::MultiFab::Copy(plotdata, E_field_plot_mf, 0, 3, 1, 0);
      amrex::Vector<std::string> varnames = {"electron_temp", "E_over_N", "electron_dens", "E_field"};

      // Write to disc
      std::string filename = amrex::Concatenate(prob_parm_d->plot_harps_file.data(), step_counter, 5);

      amrex::RealBox rb(geomdata.ProbLo(), geomdata.ProbHi());
      int is_per[AMREX_SPACEDIM];
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
          is_per[dir] = geomdata.isPeriodic(dir);
      }
      amrex::Geometry geom(geomdata.Domain(), &rb, geomdata.Coord(), is_per);

      amrex::WriteSingleLevelPlotfile(filename, plotdata, varnames, geom, time, 20);
      amrex::Print() << "Save Electron Related Quantities:  " << filename << "\n";
    }


    // Allocate global flat arrays to hold the full 3D grid data
    amrex::Box domain = geomdata.Domain();
    int Nx = domain.length(0);
    int Ny = domain.length(1);
    int Nz = domain.length(2);
    std::vector<double> amrex_n_e(Nx*Ny*Nz, 0.0);
    std::vector<double> amrex_mu_re(Nx*Ny*Nz, 0.0);
    std::vector<double> amrex_mu_im(Nx*Ny*Nz, 0.0);

    // Harvest data: i, j, k are automatically in global domain coordinates
    for (amrex::MFIter mfi(n_e_mf); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      auto const& n_e_fab = n_e_mf.array(mfi);
      auto const& mu_re_fab = mu_re_mf.array(mfi);
      auto const& mu_im_fab = mu_im_mf.array(mfi);

      amrex::Loop(bx, [=, &amrex_n_e, &amrex_mu_re, &amrex_mu_im](int i, int j, int k) {
        int linear_idx = i + j*Nx + k*Nx*Ny;
        amrex_n_e[linear_idx]   = n_e_fab(i, j, k);
        amrex_mu_re[linear_idx] = mu_re_fab(i, j, k);
        amrex_mu_im[linear_idx] = mu_im_fab(i, j, k);
      });
    }

    // This replaces all 0.0 placeholders with real data from neighbor ranks.
    amrex::ParallelDescriptor::ReduceRealSum(amrex_n_e.data(), amrex_n_e.size());
    amrex::ParallelDescriptor::ReduceRealSum(amrex_mu_re.data(), amrex_mu_re.size());
    amrex::ParallelDescriptor::ReduceRealSum(amrex_mu_im.data(), amrex_mu_im.size());


    // Check if PETSc is already initialized
    PetscBool petsc_is_initialized;
    PetscInitialized(&petsc_is_initialized);
    if (!petsc_is_initialized) {
      std::string prog_name = "harps_subroutine"; 
      char* fake_argv[] = {const_cast<char*>(prog_name.c_str()), const_cast<char*>(prog_name.c_str()), nullptr};
      int fake_argc = 2;
      char** p_fake_argv = fake_argv;
      PetscInitialize(&fake_argc, &p_fake_argv, NULL, NULL);
    }

    static std::vector<double> y, z;
    static bool grid_initialized = false;
    if (!grid_initialized) {
      create_grid("input/2D_YZ_3D_CFD.in", y, z);
      grid_initialized = true;
    }

    std::pair<double, double> plasma_center = interpolate_xyz_to_yz(y, z, plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im,
                                                amrex_n_e, amrex_mu_re, amrex_mu_im, Nx, Ny, Nz, prob_lo, dx, y_c, R_in, z_0, a);   

    if (should_optimize_reflector) y_reflector = optimizeReflectorPosition("input/2D_YZ_3D_CFD.in", plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im, 0.08, 0.16, plasma_pabs, plasma_efield, 12, 5e-4, prob_parm_d->harps_verbose);

    run_harps("input/2D_YZ_3D_CFD.in", plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im, y_reflector, plasma_pabs, plasma_efield, prob_parm_d->harps_verbose);
    
    if(prob_parm_d->harps_verbose > 0){
      amrex::Print() << "Plasma center = (" << plasma_center.first << ", " << plasma_center.second << ")\n";
    }

    // Allocate DeviceVectors with the explicit sizes needed
    static amrex::Gpu::DeviceVector<double> d_y;
    static amrex::Gpu::DeviceVector<double> d_z;
    static amrex::Gpu::DeviceVector<double> d_pabs;
    static amrex::Gpu::DeviceVector<double> d_efield;

    d_y.resize(y.size());
    d_z.resize(z.size());
    d_pabs.resize(plasma_pabs.size());
    d_efield.resize(plasma_efield.size());

    amrex::Gpu::copy(amrex::Gpu::hostToDevice, y.begin(), y.end(), d_y.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z.begin(), z.end(), d_z.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, plasma_pabs.begin(), plasma_pabs.end(), d_pabs.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, plasma_efield.begin(), plasma_efield.end(), d_efield.begin());

    // 4. Extract raw pointers for the GPU kernel (same as before)
    const double* y_ptr    = d_y.data();
    const double* z_ptr    = d_z.data();
    const double* pabs_ptr = d_pabs.data();
    const double* field_ptr = d_efield.data();
    
    E_field_new_a = E_field_new_mf->arrays();

    int Ny_harps = static_cast<int>(y.size());
    int Nz_harps = static_cast<int>(z.size());
    
    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
      amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
      amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5)*dx[2];
      amrex::Real r = std::sqrt(x*x + y*y);

      amrex::Real x_plasma = (static_cast<amrex::Real>(i) - ic[0]) * dx[0] - plasma_center.first;
      amrex::Real y_plasma = (static_cast<amrex::Real>(j) - ic[1]) * dx[1] - plasma_center.second;
      amrex::Real r_plasma = std::sqrt(x_plasma*x_plasma + y_plasma*y_plasma);

      bool inside_r = (r <= R_in);
      bool inside_z = (z < z_0 && z > z_0 - L_z);

      if (inside_r && inside_z) {
        double theta = std::atan2(y_plasma, x_plasma);

        double w_right = 0.5*(1.0 + std::sin(theta));
        double w_left  = 1.0 - w_right;

        double y_left   = y_c + plasma_center.second - r_plasma;
        double y_right  = y_c + plasma_center.second + r_plasma;
        double z_target = z_0 - z;

        auto sample_2d = [=] AMREX_GPU_DEVICE (double y_t, double z_t, const double* ptr) {
          int m0 = amrex::Clamp(find_nonuniform_index(y_ptr, Ny_harps, y_t), 0, Ny_harps - 2);
          int n0 = amrex::Clamp(find_nonuniform_index(z_ptr, Nz_harps, z_t), 0, Nz_harps - 2);
          double y0 = y_ptr[m0], y1 = y_ptr[m0 + 1];
          double z0 = z_ptr[n0], z1 = z_ptr[n0 + 1];
          double dy = amrex::Clamp((y_t - y0) / (y1 - y0), 0.0, 1.0);
          double dz = amrex::Clamp((z_t - z0) / (z1 - z0), 0.0, 1.0);
          double v00 = ptr[m0*Nz_harps + n0];
          double v10 = ptr[(m0 + 1)*Nz_harps + n0];
          double v01 = ptr[m0*Nz_harps + (n0 + 1)];
          double v11 = ptr[(m0 + 1)*Nz_harps + (n0 + 1)];
          return (1.0 - dy)*(1.0 - dz)*v00 + dy*(1.0 - dz)*v10 + (1.0 - dy)*dz*v01 + dy*dz*v11;
        };
        double pabs_left   = sample_2d(y_left,  z_target, pabs_ptr);
        double pabs_right  = sample_2d(y_right, z_target, pabs_ptr);
        double field_left  = sample_2d(y_left,  z_target, field_ptr);
        double field_right = sample_2d(y_right, z_target, field_ptr);

        constexpr double e_charge = 1.60217663e-19;
        double sigma = e_charge * n_e_arr[box_no](i, j, k) * mu_re_arr[box_no](i, j, k);

        if (harps_counter > stabilizing_runs) {
          // ext_src_rhoh_a[box_no](i, j, k, RHOH) = sigma * (w_right*field_right*field_right + w_left*field_left*field_left);
          ext_src_rhoh_a[box_no](i, j, k, RHOH) = w_right*pabs_right + w_left*pabs_left;
        }
        E_field_raw_a[box_no](i, j, k) = w_right*field_right + w_left*field_left;
      } else {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0;
      }
    });

    if(prob_parm_d->harps_verbose > 0){
      std::pair<double, double> pabs_center = calculate_plasma_center(*ext_src, RHOH, Nx, Ny, ic, dx);
      amrex::Print() << "p_abs center = (" << pabs_center.first << ", " << pabs_center.second << ")\n";
    }

    if (prob_parm_d->normalize_power){
      amrex::Real cell_vol = dx[0]*dx[1]*dx[2];
      amrex::Real deposited_power = ext_src->sum(RHOH, false)*cell_vol;

      if (deposited_power > 1e-8) { // Avoid division by zero
        normalization_factor = total_power/deposited_power;
      }
      if(prob_parm_d->harps_verbose > 0){
        amrex::Print() << "[Normalize Pabs] Normalization factor = " << normalization_factor << "\n";
      }

      amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = std::min(ext_src_rhoh_a[box_no](i, j, k, RHOH)*normalization_factor, max_power);
      });
    }

    amrex::Real alpha_E = 0.8; // Under-relaxation for electric field
    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real x = (static_cast<amrex::Real>(i) - ic[0]) * dx[0];
      amrex::Real y = (static_cast<amrex::Real>(j) - ic[1]) * dx[1];
      amrex::Real z = prob_lo[2] + (static_cast<amrex::Real>(k) + 0.5)*dx[2];
      amrex::Real r = std::sqrt(x*x + y*y);

      bool inside_r = (r <= R_in);
      bool inside_z = (z < z_0 && z > z_0 - L_z);

      if (inside_r && inside_z) {
        if(sdcIter < nSDCmax - 1){
          amrex::Real E_old_val = E_field_new_a[box_no](i, j, k)*sqrt(normalization_factor);
          E_field_new_a[box_no](i, j, k) = E_old_val*pow(E_field_raw_a[box_no](i, j, k) / E_old_val, alpha_E);
        }else{
          amrex::Real E_old_val = E_field_a[box_no](i, j, k)*sqrt(normalization_factor);
          E_field_a[box_no](i, j, k) = E_old_val*pow(E_field_raw_a[box_no](i, j, k) / E_old_val, alpha_E);
        }
      }
    });
  }

  #endif

  amrex::Gpu::streamSynchronize();

  // Diagnostic check: total deposited power = sum(src * dV)
  amrex::Real cell_vol = dx[0] * dx[1] * dx[2];
  amrex::Real deposited_power = ext_src->sum(RHOH, false) * cell_vol;

  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Print() << "Deposited power = " << deposited_power << " ; target = " << total_power
      << " ; rel error = " << std::abs(deposited_power - total_power) / (std::abs(total_power) + 1.0e-50) << "\n";
  }

  amrex::Real max_T = state_new.max(TEMP, 0);
  amrex::Print() << "=== Global Maximum Temperature: " << max_T << " K ===\n";
}





void
PeleLM::freeProbParm()
{
}