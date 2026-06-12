#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  std::string type;
  pp.query("P_mean", prob_parm->P_mean);
  pp.query("standoff", PeleLM::prob_parm->standoff);
  pp.query("pertmag", PeleLM::prob_parm->pertmag);
  pp.query("solidBody", PeleLM::prob_parm->solidBody);

  pp.query("utheta", PeleLM::prob_parm->utheta);
  pp.query("uz", PeleLM::prob_parm->uz);
  pp.query("ur", PeleLM::prob_parm->ur);

  pp.query("inlet_center", PeleLM::prob_parm->inletcenter);
  pp.query("inlet_delta", PeleLM::prob_parm->inletdelta);

  pp.query("total_power", PeleLM::prob_parm->total_power);
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
  const int sdcIter)
{
  const auto prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();
  auto ext_src_rhoh_a = ext_src->arrays();
  auto ext_src_a = ext_src->arrays();
  auto const& state_old_a = state_old.const_arrays();
  auto const& state_new_a = state_new.const_arrays();

  const amrex::Real pi = 3.141592653589793;

  amrex::Real total_power = prob_parm_d->total_power;

  amrex::Real z_center     = 0.075;  // 7.5 cm 
  amrex::Real z_half_width = 0.025;  // 2.5 cm (spans 5.0 cm to 10.0 cm)
  amrex::Real r_max        = 0.008;  // 0.8 cm

  bool print_P_in = 1;
  bool do_harps = 0;
  bool normalize_power = 1;
  
  if (time < 0.05){
    do_harps = 0;
  } else {
    do_harps = 1;
  }

  if (do_harps == 0) {
    const amrex::Real P_0 = (3.0 * total_power) / (2.0 * pi * r_max * r_max * z_half_width);

    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
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
    });
  } else if (do_harps == 1) {
    total_power *= 1.5;
    double y_c = 0.146;
    double R_in = 0.0135;
    double z_0 = 0.015;
    double L_z = 0.120;

    std::vector<std::tuple<int, int, int>> plasma_locations;
    std::vector<double> plasma_ne;
    std::vector<double> plasma_mu_re;
    std::vector<double> plasma_mu_im;
    std::vector<double> plasma_pabs;

    std::vector<double> y;
    std::vector<double> z;


    amrex::MultiFab n_e_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab mu_re_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);
    amrex::MultiFab mu_im_mf(ext_src->boxArray(), ext_src->DistributionMap(), 1, 0);

    auto n_e_arr = n_e_mf.arrays();
    auto mu_re_arr = mu_re_mf.arrays();
    auto mu_im_arr = mu_im_mf.arrays();

    // Compute conductivity related quantities (n_e, mu)
    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept {
      amrex::Real Tg_mid;
      if(sdcIter > 0) {
        Tg_mid = (state_old_a[box_no](i, j, k, TEMP) + state_new_a[box_no](i, j, k, TEMP)) * 0.5;
      } else {
        Tg_mid = state_old_a[box_no](i, j, k, TEMP);
      }

      n_e_arr[box_no](i, j, k)   = 3e19/(1 + std::exp((7000 - Tg_mid)/600));
      mu_re_arr[box_no](i, j, k) = 50/std::sqrt(Tg_mid);
      mu_im_arr[box_no](i, j, k) = -100/std::sqrt(Tg_mid);
    });

    // Allocate global flat arrays to hold the full 2D grid data
    amrex::Box domain = geomdata.Domain();
    int Nr = domain.length(0);
    int Nz = domain.length(1);
    std::vector<double> amrex_n_e(Nr * Nz, 0.0);
    std::vector<double> amrex_mu_re(Nr * Nz, 0.0);
    std::vector<double> amrex_mu_im(Nr * Nz, 0.0);

    // Harvest data: i and j are automatically in global domain coordinates
    for (amrex::MFIter mfi(n_e_mf); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      auto const& n_e_fab = n_e_mf.array(mfi);
      auto const& mu_re_fab = mu_re_mf.array(mfi);
      auto const& mu_im_fab = mu_im_mf.array(mfi);

      amrex::Loop(bx, [=, &amrex_n_e, &amrex_mu_re, &amrex_mu_im](int i, int j, int k) {
        int linear_idx = i + j * Nr;
        amrex_n_e[linear_idx]   = n_e_fab(i, j, k);
        amrex_mu_re[linear_idx] = mu_re_fab(i, j, k);
        amrex_mu_im[linear_idx] = mu_im_fab(i, j, k);
      });
    }

    // This replaces all 0.0 placeholders with real data from neighbor ranks.
    amrex::ParallelDescriptor::ReduceRealSum(amrex_n_e.data(), amrex_n_e.size());
    amrex::ParallelDescriptor::ReduceRealSum(amrex_mu_re.data(), amrex_mu_re.size());
    amrex::ParallelDescriptor::ReduceRealSum(amrex_mu_im.data(), amrex_mu_im.size());


    create_grid("input/2D_RZ.in", y, z);

    interpolate_rz_to_yz(y, z, plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im,
                        amrex_n_e, amrex_mu_re, amrex_mu_im, Nr, Nz, prob_lo, dx, y_c, R_in);    
    
    run_harps("input/2D_RZ.in", plasma_locations, plasma_ne, plasma_mu_re, plasma_mu_im, plasma_pabs);


    // Allocate DeviceVectors with the explicit sizes needed
    amrex::Gpu::DeviceVector<double> d_y(y.size());
    amrex::Gpu::DeviceVector<double> d_z(z.size());
    amrex::Gpu::DeviceVector<double> d_pabs(plasma_pabs.size());

    // Explicitly copy data from Host to Device
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, y.begin(), y.end(), d_y.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, z.begin(), z.end(), d_z.begin());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, plasma_pabs.begin(), plasma_pabs.end(), d_pabs.begin());

    // 4. Extract raw pointers for the GPU kernel (same as before)
    const double* y_ptr    = d_y.data();
    const double* z_ptr    = d_z.data();
    const double* pabs_ptr = d_pabs.data();

    int Ny = static_cast<int>(y.size());
    Nz = z.size();
    
    // Interpolate p_abs from Harps grid into PeleLMeX grid
    amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
      amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
      amrex::Real z = prob_lo[1] + (static_cast<amrex::Real>(j) + 0.5) * dx[1];

      bool inside_r = (r <= R_in);
      bool inside_z = (z > z_0 && z < z_0 + L_z);

      if (inside_r && inside_z) {
        double y_target = y_c - r;
        double z_target = z;

        // Find surrounding indices on the non-uniform grid
        int m0 = find_nonuniform_index(y_ptr, Ny, y_target);
        m0 = amrex::Clamp(m0, 0, Ny - 2); // 
        int m1 = m0 + 1;

        int n0 = find_nonuniform_index(z_ptr, Nz, z_target);
        n0 = amrex::Clamp(n0, 0, Nz - 2); // 
        int n1 = n0 + 1;

        // Get exact coordinates of the bounding box corners
        double y0 = y_ptr[m0];
        double y1 = y_ptr[m1];
        double z0 = z_ptr[n0];
        double z1 = z_ptr[n1];

        double dy = amrex::Clamp((y_target - y0) / (y1 - y0), 0.0, 1.0);
        double dz = amrex::Clamp((z_target - z0) / (z1 - z0), 0.0, 1.0);

        // Fetch values from flat 2D layout layout: [m * Nz + n])
        double v00 = pabs_ptr[m0*Nz + n0]; // Bottom-Left  (y0, z0)
        double v10 = pabs_ptr[m1*Nz + n0]; // Bottom-Right (y1, z0)
        double v01 = pabs_ptr[m0*Nz + n1]; // Top-Left     (y0, z1)
        double v11 = pabs_ptr[m1*Nz + n1]; // Top-Right    (y1, z1)

        // Execute non-uniform bilinear interpolation
        amrex::Real interpolated_plasma_pabs = (1.0 - dy)*(1.0 - dz)*v00 + dy*(1.0 - dz)*v10 + (1.0 - dy)*dz*v01 + dy*dz*v11;

        //std::cout << interpolated_plasma_pabs << std::endl;
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = interpolated_plasma_pabs;
      } else {
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = 0.0; 
      }
    });

    if (normalize_power){
      amrex::ReduceOps<amrex::ReduceOpSum> reduce_op;
      amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
      using ReduceTuple = typename amrex::ReduceData<amrex::Real>::Type;

      for (amrex::MFIter mfi(*ext_src); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& ext_src_a = ext_src->array(mfi);

        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple{
          amrex::Real r = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
            
          amrex::Real cell_vol = 2.0 * pi * r * dx[0] * dx[1];
            
          return ext_src_a(i, j, k, RHOH) * cell_vol;
        });
      }

      // Gather the sum across all local GPU threads
      amrex::Real deposited_power = amrex::get<0>(reduce_data.value());
      amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

      amrex::Real normalization_factor = 0;
      if (deposited_power > 1e-8) { // Avoid division by zero
        normalization_factor = total_power/deposited_power;
      }

      amrex::ParallelFor(*ext_src, [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k) noexcept{
        ext_src_rhoh_a[box_no](i, j, k, RHOH) = std::min(ext_src_rhoh_a[box_no](i, j, k, RHOH)*normalization_factor, 3e8);
      });
    }
  }


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
    amrex::ParallelDescriptor::ReduceRealSum(deposited_power);

    if (amrex::ParallelDescriptor::IOProcessor()) {
      amrex::Print()
        << "Deposited power = " << deposited_power
        << " ; target = " << total_power
        << "\n";
    }
  }

};