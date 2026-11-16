#ifndef HARPS_H  // Include guard to prevent double inclusion
#define HARPS_H

#include <string>
#include <utility>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

void create_grid(const std::string& config_file_path, std::vector<double>& y, std::vector<double>& z);

double run_harps(const std::string& config_file_path, std::vector<std::tuple<int, int, int>> plasma_locations,
            std::vector<double> plasma_ne, std::vector<double> plasma_mu_re, std::vector<double> plasma_mu_im, double y_reflector,
            std::vector<double>& plasma_pabs, std::vector<double>& plasma_E_field, int harps_verbose = 1);

void interpolate_rz_to_yz(const std::vector<double>& y, const std::vector<double>& z, std::vector<std::tuple<int, int, int>>& plasma_locations,
                        std::vector<double>& plasma_ne, std::vector<double>& plasma_mu_re, std::vector<double>& plasma_mu_im,
                        const std::vector<double>& amrex_n_e, const std::vector<double>& amrex_mu_re, const std::vector<double>& amrex_mu_im,
                        int Nr, int Nz, const double* prob_lo, const double* dx, double y_c, double R_in, double z_0);

std::pair<double, double> interpolate_xyz_to_yz(const std::vector<double>& y, const std::vector<double>& z, std::vector<std::tuple<int, int, int>>& plasma_locations,
                        std::vector<double>& plasma_ne, std::vector<double>& plasma_mu_re, std::vector<double>& plasma_mu_im,
                        const std::vector<double>& amrex_n_e, const std::vector<double>& amrex_mu_re, const std::vector<double>& amrex_mu_im,
                        int Nx, int Ny, int Nz, const double* prob_lo, const double* dx, double y_c, double R_in, double z_0, double a);

std::pair<double, double> calculate_plasma_center(const amrex::MultiFab& ext_src, int comp, int Nx, int Ny, const amrex::GpuArray<double, 3>& ic, const double* dx);

double optimizeReflectorPosition(const std::string& config_file_path, const std::vector<std::tuple<int, int, int>>& plasma_locations,
    const std::vector<double>& plasma_ne, const std::vector<double>& plasma_mu_re, const std::vector<double>& plasma_mu_im,
    double y_lower, double y_upper, std::vector<double>& plasma_pabs, std::vector<double>& plasma_efield,
    int max_iterations = 10, double tolerance = 1e-4, int harps_verbose = 1);


#endif // HARPS_H