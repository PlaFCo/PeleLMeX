#ifndef HARPS_H  // Include guard to prevent double inclusion
#define HARPS_H

#include <string>

void create_grid(const std::string& config_file_path, std::vector<double>& y, std::vector<double>& z);

int run_harps(const std::string& config_file_path, std::vector<std::tuple<int, int, int>> plasma_locations, std::vector<double> plasma_ne, std::vector<double> plasma_mu_re, std::vector<double> plasma_mu_im);

void convert_rz_to_2d(std::vector<std::tuple<int, int, int>>& plasma_locations, std::vector<double>& plasma_ne, std::vector<double>& plasma_mu_re, std::vector<double>& plasma_mu_im);


#endif // HARPS_H