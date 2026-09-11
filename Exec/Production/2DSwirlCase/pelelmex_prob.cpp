#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>
#include "lut_plasma_props.h"

#include <AMReX_PlotFileUtil.H>

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  std::string type;
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("standoff", PeleLM::prob_parm->standoff);
  pp.query("pertmag", PeleLM::prob_parm->pertmag);
  pp.query("solidBody", PeleLM::prob_parm->solidBody);

  pp.query("uz", PeleLM::prob_parm->uz);
  pp.query("inlet_center", PeleLM::prob_parm->inletcenter);
  pp.query("inlet_delta", PeleLM::prob_parm->inletdelta);

  pp.query("ur", PeleLM::prob_parm->ur);
  pp.query("utheta", PeleLM::prob_parm->utheta);
  pp.query("main_center", PeleLM::prob_parm->main_center);
  pp.query("main_delta", PeleLM::prob_parm->main_delta);

  pp.query("x_O2_main", PeleLM::prob_parm->x_O2_main);
  pp.query("x_O2_quench", PeleLM::prob_parm->x_O2_quench);

  pp.query("quench_flag", PeleLM::prob_parm->quench_flag);
  pp.query("ur_quench", PeleLM::prob_parm->ur_quench);
  pp.query("quench_center", PeleLM::prob_parm->quench_center);
  pp.query("quench_delta", PeleLM::prob_parm->quench_delta);

  pp.query("total_power", PeleLM::prob_parm->total_power);

  pp.query("EM_t_start", PeleLM::prob_parm->EM_t_start);
  pp.query("flag_solve_EM", PeleLM::prob_parm->flag_solve_EM);
  pp.query("plot_harps_int", PeleLM::prob_parm->plot_harps_int);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);
  pp.query("normalize_power", PeleLM::prob_parm->normalize_power);
  pp.query("harps_verbose", PeleLM::prob_parm->harps_verbose);

  std::string plot_harps_file_str; 
  pp.query("plot_harps_file", plot_harps_file_str); std::strncpy(prob_parm->plot_harps_file.data(), plot_harps_file_str.c_str(), 64); prob_parm->plot_harps_file[63] = '\0'; // Ensure null-termination

  pp.query("sponge_gamma", PeleLM::prob_parm->sponge_gamma);
  pp.query("sponge_zstart", PeleLM::prob_parm->sponge_zstart);
  pp.query("sponge_zend", PeleLM::prob_parm->sponge_zend);
}

void
PeleLM::freeProbParm()
{
}


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

  amrex::Real total_power = prob_parm_d->total_power;

  amrex::Real z_center     = 0.140;  // 7.5 cm 
  amrex::Real z_half_width = 0.025;  // 3.0 cm (spans 5.0 cm to 10.0 cm)
  amrex::Real r_max        = 0.008;  // 0.8 cm

  amrex::Real power_time   = 0.2;  //

  static amrex::Real normalization_factor = 1;

  bool do_harps = 0; 

  total_power *= amrex::min(1.0, time/power_time);

  static int step_counter = 0;
  if (sdcIter == 0) step_counter++;

  if (total_power < 1e-6) return;
  
  if (prob_parm_d->flag_solve_EM && time > prob_parm_d->EM_t_start){
    do_harps = 1;
  } else{
    do_harps = 0;
  }

  if (do_harps == 0) {
    amrex::Real P_0 = (3.0*total_power)/(2.0*pi*r_max*r_max*z_half_width);

    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
      amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5)*dx[0];
      amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5)*dx[1];

      bool inside_r = (r <= r_max);
      bool inside_z = (amrex::Math::abs(z - z_center) <= z_half_width);

      if (inside_r && inside_z) {
        amrex::Real r_norm = r / r_max;
        amrex::Real z_norm = (z - z_center) / z_half_width;

        ext_src_rhoh_a[box_no](i, j, k, RHOH) = P_0*(1.0 - r_norm*r_norm)*(1.0 - z_norm*z_norm)*0.99963489868;
      } else {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0; 
      }
    });
  } else if (do_harps == 1) {
    double y_c = 0.056;
    double R_in = 0.0135;
    double L_z = 0.120;

    double max_power = 10e8;

    double z_0 = z_center - (L_z/2 - 0.01);

    std::vector<std::tuple<int, int, int>> plasma_locations;
    std::vector<double> plasma_ne;
    std::vector<double> plasma_mu_re;
    std::vector<double> plasma_mu_im;
    std::vector<double> plasma_pabs;
    std::vector<double> plasma_efield;

    std::vector<double> y;
    std::vector<double> z;

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
    E_field_raw_mf.setVal(0.0);
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
    Te_mf.setVal(0.0); E_over_N_mf.setVal(0.0); ne_mf.setVal(0.0);

    auto Te_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? Te_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    auto E_over_N_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? E_over_N_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    auto ne_arr = ((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)) ? ne_mf.arrays() : amrex::MultiArray4<amrex::Real>();
    
    
    // Compute conductivity related quantities (n_e, mu)
    amrex::Real E_ion_O2 = 12.06*1.60218e-19; // J
    amrex::Real E_ion_N2 = 15.58*1.60218e-19; // J
    amrex::Real E_ion_NO = 9.26*1.60218e-19;  // J
    amrex::Real h_planck = 6.62607015e-34;      // J*s
    amrex::Real k_B = 1.380649e-23;             // J/K
    amrex::Real m_e = 9.10938356e-31;           // kg
    amrex::Real atomic_mass_unit = 1.66053906660e-27; // kg
    
    PlasmaLookupTable lut;
    amrex::Real alpha_under_relaxation = 0.5;

    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5)*dx[1];
      if (z > z_0 && z < z_0 + L_z){
        amrex::Real Tg_mid;
        amrex::Real n_O2_mid; amrex::Real n_N2_mid; amrex::Real n_NO_mid = 0; amrex::Real n_O_mid = 0; amrex::Real n_N_mid = 0;
        amrex::Real n_e_O2; amrex::Real n_e_N2; amrex::Real n_e_NO = 0;
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
        n_e_O2 = sqrt(n_O2_mid*pow(2*3.14159*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_O2/(k_B*Te_mid)));
        n_e_N2 = sqrt(n_N2_mid*pow(2*3.14159*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_N2/(k_B*Te_mid)));
        n_e_NO = sqrt(n_NO_mid*pow(2*3.14159*m_e*k_B*Te_mid/h_planck/h_planck,1.5)*exp(-E_ion_NO/(k_B*Te_mid)));

        amrex::Real ne_saha = sqrt(n_e_O2*n_e_O2 + n_e_N2*n_e_N2 + n_e_NO*n_e_NO);    //*(1/(1 + pow((Te_mid/Tg_mid - 1)/(1.8 - 1), 4)));  // Non equilibrium correction factor
        amrex::Real ne_old = n_e_old_arr[box_no](i,j,k);

        amrex::Real ne_new = ne_old*pow(ne_saha / ne_old, alpha_under_relaxation); // Under relaxation
        n_e_arr[box_no](i,j,k) = ne_new;
        n_e_old_arr[box_no](i,j,k) = ne_new;

        if(i == 1 && j == 350 && prob_parm_d->harps_verbose > 1) {
          std::cout << "(1,350) Tg: " << Tg_mid << ", E: " << E_field_mid << "; Te: " << Te_mid << "; ne: " << n_e_arr[box_no](i,j,k) << std::endl;
        }

        if((step_counter % prob_parm_d->plot_harps_int == 0) && (sdcIter == 0)){
          Te_arr[box_no](i, j, k) = Te_mid;
          E_over_N_arr[box_no](i, j, k) = E_over_N;
          ne_arr[box_no](i, j, k) = n_e_arr[box_no](i,j,k);
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
      amrex::MultiFab plotdata(ba, dmap, 3, 0);
      amrex::MultiFab::Copy(plotdata, Te_mf, 0, 0, 1, 0);
      amrex::MultiFab::Copy(plotdata, E_over_N_mf, 0, 1, 1, 0);
      amrex::MultiFab::Copy(plotdata, ne_mf, 0, 2, 1, 0); 
      amrex::Vector<std::string> varnames = {"electron_temp", "E_over_N", "electron_dens"};

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


    // Allocate global flat arrays to hold the full 2D grid data
    amrex::Box domain = geomdata.Domain();
    int Nr = domain.length(0);
    int Nz = domain.length(1);
    std::vector<double> amrex_n_e(Nr*Nz, 0.0);
    std::vector<double> amrex_mu_re(Nr*Nz, 0.0);
    std::vector<double> amrex_mu_im(Nr*Nz, 0.0);

    // Harvest data: i and j are automatically in global domain coordinates
    for (amrex::MFIter mfi(n_e_mf); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      auto const& n_e_fab = n_e_mf.array(mfi);
      auto const& mu_re_fab = mu_re_mf.array(mfi);
      auto const& mu_im_fab = mu_im_mf.array(mfi);

      amrex::Loop(bx, [=, &amrex_n_e, &amrex_mu_re, &amrex_mu_im](int i, int j, int k) {
        int linear_idx = i + j*Nr;
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

    create_grid("input/2D_RZ.in", y, z);

    interpolate_rz_to_yz(y, z, plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im,
                        amrex_n_e, amrex_mu_re, amrex_mu_im, Nr, Nz, prob_lo, dx, y_c, R_in, z_0);   

    run_harps("input/2D_RZ.in", plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im, plasma_pabs, plasma_efield);

    // Allocate DeviceVectors with the explicit sizes needed
    amrex::Gpu::DeviceVector<double> d_y(y.size());
    amrex::Gpu::DeviceVector<double> d_z(z.size());
    amrex::Gpu::DeviceVector<double> d_pabs(plasma_pabs.size());
    amrex::Gpu::DeviceVector<double> d_efield(plasma_efield.size());

    // Explicitly copy data from Host to Device
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, y.begin(), y.end(), d_y.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z.begin(), z.end(), d_z.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, plasma_pabs.begin(), plasma_pabs.end(), d_pabs.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, plasma_efield.begin(), plasma_efield.end(), d_efield.begin());

    // 4. Extract raw pointers for the GPU kernel (same as before)
    const double* y_ptr    = d_y.data();
    const double* z_ptr    = d_z.data();
    const double* pabs_ptr = d_pabs.data();
    const double* field_ptr = d_efield.data();

    int Ny = static_cast<int>(y.size());
    Nz = z.size();
    
    E_field_new_a = E_field_new_mf->arrays();

    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5)*dx[0];
      amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5)*dx[1];

      bool inside_r = (r <= R_in);
      bool inside_z = (z > z_0 && z < z_0 + L_z);

      if (inside_r && inside_z) {
        double y_left  = y_c - r;
        double y_right = y_c + r;
        double z_target = z - z_0;

        auto sample_2d = [=] AMREX_GPU_DEVICE (double y_t, double z_t, const double* ptr) {
          int m0 = amrex::Clamp(find_nonuniform_index(y_ptr, Ny, y_t), 0, Ny - 2);
          int n0 = amrex::Clamp(find_nonuniform_index(z_ptr, Nz, z_t), 0, Nz - 2);

          double y0 = y_ptr[m0],     y1 = y_ptr[m0 + 1];
          double z0 = z_ptr[n0],     z1 = z_ptr[n0 + 1];

          double dy = amrex::Clamp((y_t - y0) / (y1 - y0), 0.0, 1.0);
          double dz = amrex::Clamp((z_t - z0) / (z1 - z0), 0.0, 1.0);

          double v00 = ptr[m0*Nz + n0];
          double v10 = ptr[(m0 + 1)*Nz + n0];
          double v01 = ptr[m0*Nz + (n0 + 1)];
          double v11 = ptr[(m0 + 1)*Nz + (n0 + 1)];

          return (1.0 - dy)*(1.0 - dz)*v00 + dy*(1.0 - dz)*v10 + (1.0 - dy)*dz*v01 + dy*dz*v11;
        };

        // Bilinear interpolation for both sides across the center axis
        double pabs_left   = sample_2d(y_left,  z_target, pabs_ptr);
        double pabs_right  = sample_2d(y_right, z_target, pabs_ptr);

        double field_left  = sample_2d(y_left,  z_target, field_ptr);
        double field_right = sample_2d(y_right, z_target, field_ptr);

        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.5*(pabs_left + pabs_right);
        E_field_raw_a[box_no](i, j, k)        = 0.5*(field_left + field_right);
      } else {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0; 
      }
    });

    if (prob_parm_d->normalize_power){
      amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
      amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename amrex::ReduceData<amrex::Real>::Type;

      for (amrex::MFIter mfi(*ext_src); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& ext_src_a = ext_src->array(mfi);

        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple{
          amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5)*dx[0];
          amrex::Real cell_vol = 2.0*pi*r*dx[0]*dx[1];
            
          return ext_src_a(i, j, k, RHOH)*cell_vol;
        });
      }

      // Gather the sum across all local GPU threads
      amrex::Real deposited_power = amrex::get<0>(reduce_data.value());
      amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

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
      amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5)*dx[0];
      amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5)*dx[1];
  
      if (r <= R_in && z > z_0 && z < z_0 + L_z) {
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

  if(prob_parm_d->harps_verbose > 0) {
    amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename amrex::ReduceData<amrex::Real>::Type;

    for (amrex::MFIter mfi(*ext_src); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& ext_src_a = ext_src->array(mfi);

        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
        {
            amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5)*dx[0];
            
            // Calculate the volume of this specific cylindrical ring shell
            amrex::Real cell_vol = 2.0*pi*r*dx[0]*dx[1];
            
            return ext_src_a(i, j, k, RHOH)*cell_vol;
        });
    }

    // Gather the sum across all local GPU threads
    amrex::Real deposited_power = amrex::get<0>(reduce_data.value());
    amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

    if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
        << "[Print Pabs] Deposited power = " << deposited_power
        << " ; target = " << total_power
        << "\n";
    }
  }

};