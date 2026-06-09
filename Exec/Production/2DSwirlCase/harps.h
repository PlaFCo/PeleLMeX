#ifndef HARPS_H  // Include guard to prevent double inclusion
#define HARPS_H

#include <string>

void create_grid(const std::string& config_file_path, std::vector<double>& y, std::vector<double>& z);

double run_harps(const std::string& config_file_path, std::vector<std::tuple<int, int, int>> plasma_locations,
            std::vector<double> plasma_ne, std::vector<double> plasma_mu_re, std::vector<double> plasma_mu_im,
            std::vector<double>& plasma_pabs);

void interpolate_rz_to_yz(const std::vector<double>& y, const std::vector<double>& z, std::vector<std::tuple<int, int, int>>& plasma_locations,
                        std::vector<double>& plasma_ne, std::vector<double>& plasma_mu_re, std::vector<double>& plasma_mu_im,
                        const std::vector<double>& amrex_n_e, const std::vector<double>& amrex_mu_re, const std::vector<double>& amrex_mu_im,
                        int Nr, int Nz, const double* prob_lo, const double* dx, double y_c, double R_in);

#endif // HARPS_H