#include "ConfigParser.h"
#include "CartesianCoordinates.h"
#include "MatrixSolver.h"
#include "Constants.h"
#include "Utils.h"

#include "harps.h"
#include <PeleLMeX.H>

std::string harps_dir = "../../../../power_coupling/harps/";    // Path to the harps directory if I'm running from harps or 1d_fluid if it's next to harps


void create_grid(const std::string& config_file_path, std::vector<double>& y, std::vector<double>& z){
        ConfigParser parser;
        HARPSConfig config;
        config = parser.parseFile(harps_dir + config_file_path);

        std::unique_ptr<CoordinateSystem> Grid;
        Grid = std::make_unique<CartesianCoordinateSystem>(config.n_x, config.n_y, config.n_z, config.lengthX, config.lengthY, config.lengthZ);
        bool non_uniform_grid = Grid->createNonUniformGrid(config.refinementFactor_x, config.refinementFactor_y, config.refinementFactor_z, 
                            config.x_BL, config.x_BR, config.x_RL, config.x_RR, config.y_BL, config.y_BR, config.y_RL, config.y_RR,
                            config.z_BL, config.z_BR, config.z_RL, config.z_RR, config.x_grid, config.y_grid, config.z_grid);

        y = Grid->y; z = Grid->z;

        return;
}


int run_harps(const std::string& config_file_path, std::vector<std::tuple<int, int, int>> plasma_locations,
            std::vector<double> plasma_ne, std::vector<double> plasma_mu_re, std::vector<double> plasma_mu_im, std::vector<double>& plasma_pabs, std::vector<double>& plasma_E_field){
    using Complex = std::complex<double>;
    const Complex zero_C(0.0, 0.0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "[HARPS] Subroutine initialized with config: " << config_file_path << std::endl;
        std::cout << "[HARPS] Total MPI ranks: " << size << std::endl;
    }

    Mat systemMaxwell = nullptr;
    Vec b_vector = nullptr;
    Vec x_seq = nullptr;
    VecScatter scatter = nullptr;

    try {
        ConfigParser parser;
        HARPSConfig config;
        
        // Start measuring time
        PetscLogDouble startTime, endTime;
        PetscTime(&startTime);
        double previous_time = startTime;

        config = parser.parseFile(harps_dir + config_file_path);

        if(config.printConfig && rank == 0) std::cout << config << std::endl;

        // debugging prints
        const bool print_ranks = 0;
        const bool print_times = 0;
        
        const bool symmetric_harps = false;

        int num_pts, num_variables, temp_idx;
        double angular_frequency, vacuum_wave_number;

        num_pts = (config.n_x * config.n_y * config.n_z);
        num_variables = num_pts * 3;

        if(num_pts%size != 0){
            throw std::runtime_error("Number of threads (" + std::to_string(size) +") needs to evenly divide the number of points (" + std::to_string(num_pts) +"): " + std::to_string(1.*num_pts/size));
            return 0;
        }
        
        angular_frequency = 2.0 * M_PI * config.frequency;
        vacuum_wave_number = angular_frequency / Constants::C_LIGHT;

        // Initialize arrays
        double* electron_density = new double[num_pts];
        double* real_permittivity = new double[num_pts];
        double* real_conductivity = new double[num_pts];
        double* real_mobility = new double[num_pts];
        double* imag_mobility = new double[num_pts];
        Complex* complex_conductivity = new Complex[num_pts];
        Complex* complex_permittivity = new Complex[num_pts];

        // Initialize default values
        std::fill_n(electron_density, num_pts, config.electronDensity);
        std::fill_n(real_permittivity, num_pts, config.realPermittivity);
        std::fill_n(real_mobility, num_pts, config.realMobility);
        std::fill_n(imag_mobility, num_pts, config.imagMobility);

        // Apply input position specific values of epsilon_r and n_e
        for (size_t i = 0; i < config.realPermittivityLocations.size(); ++i) {
            const auto& loc = config.realPermittivityLocations[i];
            double value = config.realPermittivityValues[i];
            
            size_t index = std::get<0>(loc)*config.n_z*config.n_y + std::get<1>(loc)*config.n_z + std::get<2>(loc);
            real_permittivity[index] = value;
        }

        config.electronDensityLocations = plasma_locations;
        config.electronDensityValues = plasma_ne;
        config.realMobilityLocations = plasma_locations;
        config.realMobilityValues = plasma_mu_re;
        config.imagMobilityLocations = plasma_locations;
        config.imagMobilityValues = plasma_mu_im;
        
        for (size_t i = 0; i < config.electronDensityLocations.size(); ++i) {
            const auto& loc = config.electronDensityLocations[i];
            double value = config.electronDensityValues[i];
            
            size_t index = std::get<0>(loc)*config.n_z*config.n_y + std::get<1>(loc)*config.n_z + std::get<2>(loc);
            electron_density[index] = value;
        }
        for (size_t i = 0; i < config.realMobilityLocations.size(); ++i) {
            const auto& loc = config.realMobilityLocations[i];
            double value = config.realMobilityValues[i];
            
            size_t index = std::get<0>(loc)*config.n_z*config.n_y + std::get<1>(loc)*config.n_z + std::get<2>(loc);
            real_mobility[index] = value;
        }
        for (size_t i = 0; i < config.imagMobilityLocations.size(); ++i) {
            const auto& loc = config.imagMobilityLocations[i];
            double value = config.imagMobilityValues[i];
            
            size_t index = std::get<0>(loc)*config.n_z*config.n_y + std::get<1>(loc)*config.n_z + std::get<2>(loc);
            imag_mobility[index] = value;
        }

        // Create output directory if it doesn't exist
        config.outputDirectory = harps_dir + config.outputDirectory;
        if (!config.outputDirectory.empty() && config.outputDirectory.back() != '/') config.outputDirectory += '/';
        
        if(print_times){
            PetscTime(&endTime);
            PetscPrintf(PETSC_COMM_WORLD, "Time taken on config: %.3f seconds\n", endTime - previous_time);
            previous_time = endTime;
        }

        if(config.printProgress && rank == 0) std::cout << "Creating System Matrix (" << num_variables << "x" << num_variables << ")" << std::endl;

        std::unique_ptr<CoordinateSystem> Grid;
        Grid = std::make_unique<CartesianCoordinateSystem>(config.n_x, config.n_y, config.n_z, config.lengthX, config.lengthY, config.lengthZ);
        bool non_uniform_grid = Grid->createNonUniformGrid(config.refinementFactor_x, config.refinementFactor_y, config.refinementFactor_z, 
                            config.x_BL, config.x_BR, config.x_RL, config.x_RR, config.y_BL, config.y_BR, config.y_RL, config.y_RR,
                            config.z_BL, config.z_BR, config.z_RL, config.z_RR, config.x_grid, config.y_grid, config.z_grid);

        if(config.printGrid && rank == 0 && non_uniform_grid) printGrid(config.outputDirectory + "Grid.txt", config.n_x, config.n_y, config.n_z,
                                                                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);

        // Apply PML if enabled (needs to be called before creating matrix)
        if (config.enableXLowerPML)  Grid->createPMLProfile('x', false, config.xLowerLayers, angular_frequency, config.orderPML, config.sigma_0);
        if (config.enableXUpperPML)  Grid->createPMLProfile('x', true, config.xUpperLayers, angular_frequency, config.orderPML, config.sigma_0);
        if (config.enableYLowerPML)  Grid->createPMLProfile('y', false, config.yLowerLayers, angular_frequency, config.orderPML, config.sigma_0);
        if (config.enableYUpperPML)  Grid->createPMLProfile('y', true, config.yUpperLayers, angular_frequency, config.orderPML, config.sigma_0);
        if (config.enableZLowerPML)  Grid->createPMLProfile('z', false, config.zLowerLayers, angular_frequency, config.orderPML, config.sigma_0);
        if (config.enableZUpperPML)  Grid->createPMLProfile('z', true, config.zUpperLayers, angular_frequency, config.orderPML, config.sigma_0);

        // Calculate plasma conductivity and permitivity
        for (int i = 0; i < num_pts; ++i) {
            Complex mobility(real_mobility[i], imag_mobility[i]);
            complex_conductivity[i] = Constants::CHARGE_E*electron_density[i]*mobility;
        }
        if (config.flag_cylindrical_plasma) Grid->calculatePlasmaFillingFactor(complex_conductivity, config.yCenter);     // 2D YZ Plasma Filling in X

        for (int i = 0; i < num_pts; ++i){
            real_conductivity[i] = std::real(complex_conductivity[i]);  // Used for p_abs = 0.5*cond_real*|E|^2
            complex_permittivity[i] = real_permittivity[i] - Complex(0.0, 1.0)*complex_conductivity[i]/(angular_frequency*Constants::EPSILON_0);
        }

        std::vector<Complex> f_grad_cond = Grid->calculateCondGradFunction(complex_conductivity, complex_permittivity, angular_frequency);        

        // Create system matrix
        systemMaxwell = Grid->createMaxwellEquationMatrix(f_grad_cond.data(), complex_permittivity, vacuum_wave_number, config.waveguide_number, size);
        
        VecCreate(PETSC_COMM_WORLD, &b_vector);
        VecSetSizes(b_vector, PETSC_DECIDE, num_variables);
        VecSetFromOptions(b_vector);
        VecSetBlockSize(b_vector, 3);

        // Set right-hand side (forcing function)
        VecSet(b_vector, 0.0);

        PetscInt rstart, rend;
        MatGetOwnershipRange(systemMaxwell, &rstart, &rend);
        PetscInt low_rank, high_rank;
        VecGetOwnershipRange(b_vector, &low_rank, &high_rank);
        
        if(print_ranks){
            std::cout << "Vector range: " << rank << " owns rows " << low_rank << " to " << high_rank << std::endl;
            std::cout << "Matrix range: " << rank << " owns rows " << rstart   << " to " << rend << std::endl;
        }

        MatAssemblyBegin(systemMaxwell, MAT_FLUSH_ASSEMBLY);
        MatAssemblyEnd(systemMaxwell, MAT_FLUSH_ASSEMBLY);

        if(print_times){
            PetscTime(&endTime);
            PetscPrintf(PETSC_COMM_WORLD, "Time taken on creating Matrix: %.3f seconds\n", endTime - previous_time);
            previous_time = endTime;
        }

        // Apply boundary conditions
        if(config.printProgress  && rank == 0) std::cout << "Applying Boundary Conditions" << std::endl;

        auto applyXBC = [&](const std::string& bcType, bool upper) {
            if (bcType == "PerfectConductor") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_y", 'x', upper, zero_C, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_z", 'x', upper, zero_C, low_rank, high_rank);
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "E_x", 'x', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Homogeneous") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "all", 'x', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Dirichlet") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "x", 'x', upper, config.DirichletBx_x, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "y", 'x', upper, config.DirichletBx_y, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "z", 'x', upper, config.DirichletBx_z, low_rank, high_rank);
            } else if (bcType == "Neumann") {
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "all", 'x', upper, zero_C, low_rank, high_rank);
            }  else if (bcType == "Robin") {
                for(int j = 0; j<3; j++) Grid->RobinBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'x', upper, config.injectionValues[j], vacuum_wave_number, config.waveguide_number, low_rank, high_rank);
            } else if (bcType == "Excitation") {
                for(int j = 0; j<3; j++){
                    if (std::abs(config.excitationValue[j].real() - Constants::FREE.real()) > 1e-10 && std::abs(config.excitationValue[j].imag() - Constants::FREE.imag()) > 1e-10)
                        Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'x', upper, config.excitationValue[j], low_rank, high_rank);
                }
            }
        };

        auto applyYBC = [&](const std::string& bcType, bool upper) {
            if (bcType == "Robin") {
                for(int j = 0; j<3; j++) Grid->RobinBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'y', upper, config.injectionValues[j], vacuum_wave_number, config.waveguide_number, low_rank, high_rank);
            }else if (bcType == "PerfectConductor") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_x", 'y', upper, zero_C, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_z", 'y', upper, zero_C, low_rank, high_rank);
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "E_y", 'y', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Homogeneous") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "all", 'y', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Dirichlet") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "x", 'y', upper, config.DirichletBy_x, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "y", 'y', upper, config.DirichletBy_y, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "z", 'y', upper, config.DirichletBy_z, low_rank, high_rank);
            } else if (bcType == "Neumann") {
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "all", 'y', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Excitation") {
                for(int j = 0; j<3; j++){
                    if (std::abs(config.excitationValue[j].real() - Constants::FREE.real()) > 1e-10 && std::abs(config.excitationValue[j].imag() - Constants::FREE.imag()) > 1e-10)
                        Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'y', upper, config.excitationValue[j], low_rank, high_rank);
                }
            }
        };

        auto applyZBC = [&](const std::string& bcType, bool upper) {
            if (bcType == "PerfectConductor") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_x", 'z', upper, zero_C, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "E_y", 'z', upper, zero_C, low_rank, high_rank);
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "E_z", 'z', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Homogeneous") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "all", 'z', upper, zero_C, low_rank, high_rank);
            } else if (bcType == "Dirichlet") {
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "x", 'z', upper, config.DirichletBz_x, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "y", 'z', upper, config.DirichletBz_y, low_rank, high_rank);
                Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, "z", 'z', upper, config.DirichletBz_z, low_rank, high_rank);
            }  else if (bcType == "Neumann") {
                Grid->NeumannBoundaryConditions(systemMaxwell, b_vector, "all", 'z', upper, zero_C, low_rank, high_rank);
            }  else if (bcType == "Robin") {
                for(int j = 0; j<3; j++) Grid->RobinBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'z', upper, config.injectionValues[j], vacuum_wave_number, config.waveguide_number, low_rank, high_rank);
            } else if (bcType == "Excitation") {
                for(int j = 0; j<3; j++){
                    if (std::abs(config.excitationValue[j].real() - Constants::FREE.real()) > 1e-10 && std::abs(config.excitationValue[j].imag() - Constants::FREE.imag()) > 1e-10)
                        Grid->DirichletBoundaryConditions(systemMaxwell, b_vector, Grid->fields_str[j], 'z', upper, config.excitationValue[j], low_rank, high_rank);
                }
            }
        };
        
        applyXBC(config.xLowerBC, false); applyXBC(config.xUpperBC, true);
        applyYBC(config.yLowerBC, false); applyYBC(config.yUpperBC, true);
        applyZBC(config.zLowerBC, false); applyZBC(config.zUpperBC, true);

        if(config.xUpperBC == "Periodic") Grid->PeriodicBoundaryConditions(systemMaxwell, b_vector,'x', low_rank, high_rank);
        if(config.yUpperBC == "Periodic") Grid->PeriodicBoundaryConditions(systemMaxwell, b_vector,'y', low_rank, high_rank);
        if(config.zUpperBC == "Periodic") Grid->PeriodicBoundaryConditions(systemMaxwell, b_vector,'z', low_rank, high_rank);

        // Apply source
        for (size_t i = 0; i < config.sourceLocations.size(); ++i) {
            const auto& loc = config.sourceLocations[i];
            std::vector<Complex> value = config.sourceValues[i];
    
            for(int j = 0; j<3; j++){
                if (std::abs(value[j].real() - Constants::FREE.real()) > 1e-10 && std::abs(value[j].imag() - Constants::FREE.imag()) > 1e-10){
                    temp_idx = Grid->point_index(std::get<0>(loc), std::get<1>(loc), std::get<2>(loc)) + j;
                    if(temp_idx >= low_rank && temp_idx < high_rank) VecSetValue(b_vector, temp_idx , value[j], INSERT_VALUES);
                }
            }
        }

        // Apply excitations
        for (size_t i = 0; i < config.excitationLocations.size(); ++i) {
            const auto& loc = config.excitationLocations[i];
            temp_idx = Grid->point_index(std::get<0>(loc), std::get<1>(loc), std::get<2>(loc));
            std::vector<Complex> value = config.excitationValues[i];
                    
            for (int j = 0; j<3; j++){
                if (std::abs(value[j].real() - Constants::FREE.real()) > 1e-10 && std::abs(value[j].imag() - Constants::FREE.imag()) > 1e-10){
                    Grid->DirichletBoundaryConditionsPoint(systemMaxwell, b_vector, value[j], temp_idx + j, low_rank, high_rank);
                } 
            }
        }

        Grid->AddMetalPointsInVolume(systemMaxwell, b_vector, config.metalLocations, vacuum_wave_number, low_rank, high_rank);

        MatAssemblyBegin(systemMaxwell, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(systemMaxwell, MAT_FINAL_ASSEMBLY);

        VecAssemblyBegin(b_vector);
        VecAssemblyEnd(b_vector);


        if(config.printMatrix) {
            if(config.printProgress  && rank == 0) std::cout << "Printing Matrix" << std::endl;
            printMatrixToFile(systemMaxwell, config.outputDirectory+"matrix.txt");
            printVecToFile(b_vector, config.outputDirectory+"b_vector.txt");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(print_times){
            PetscTime(&endTime);
            PetscPrintf(PETSC_COMM_WORLD, "Time taken on Applying Boundary Conditions: %.3f seconds\n", endTime - previous_time);
            previous_time = endTime;
        }

        if(config.printProgress  && rank == 0) std::cout << "Solving Equations" << std::endl;

        MatrixSolver solver(num_variables, config.tolerance, config.maxIterations, config.scalar);
        if(config.solverMethod != "direct_LU" && config.initial_guess_file != "") solver.setInitialGuessFromFileBinary(harps_dir + config.initial_guess_file);
        
        solver.solve(systemMaxwell, b_vector, size, config.solverMethod, config.PC); // "sor", "amg"
        solver.printConvergenceInfo();

        Vec solution = solver.getSolution();

        if(print_times){
            PetscTime(&endTime);
            PetscPrintf(PETSC_COMM_WORLD, "Time taken on Solving System: %.3f seconds\n", endTime - previous_time);
            previous_time = endTime;
        }

        if(config.printSolution){
            printVecToFileBinary(solution, config.outputDirectory + "fields_solution.bin");
            //printVecToFile(solution, config.outputDirectory + "fields_solution.txt");
            
            if(print_times){
                PetscTime(&endTime);
                PetscPrintf(PETSC_COMM_WORLD, "Time taken on Storing Solution: %.3f seconds\n", endTime - previous_time);
                previous_time = endTime;
            }
        }

        // Create a sequential vector to hold the complete solution on each process
        MPI_Barrier(MPI_COMM_WORLD);
        if(config.printProgress  && rank == 0) std::cout << "Creating Sequential Vector" << std::endl;

                Vec x_seq = NULL; 
        VecScatter scatter;

        VecScatterCreateToAll(solution, &scatter, &x_seq);

        VecScatterBegin(scatter, solution, x_seq, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatter, solution, x_seq, INSERT_VALUES, SCATTER_FORWARD);

        std::vector<Complex> fields(num_variables);
        
        
        // If scalar system was solved put that field in the correct place on the full vector 
        if(config.scalar != -1){
            Complex* fields_scalar;
            VecGetArray(x_seq, &fields_scalar);
            for(int i = num_pts-1; i > -1; i--) 
                fields[i*3+config.scalar] = fields_scalar[i];

            for(int i = 0; i < num_pts; i++) {
                fields[i*3 + (config.scalar+1)%3] = zero_C;
                fields[i*3 + (config.scalar+2)%3] = zero_C;
            }
            VecRestoreArray(x_seq, &fields_scalar);
        } else {
            Complex* fields_petsc;
            VecGetArray(x_seq, &fields_petsc);
            std::copy(fields_petsc, fields_petsc + num_variables, fields.data());
            VecRestoreArray(x_seq, &fields_petsc);
        }
        VecScatterDestroy(&scatter);
        VecDestroy(&x_seq);

        if(config.printProgress  && rank == 0) std::cout << "Calculating Absorbed Power" << std::endl;
        
        double* absorbedPowerDensity;
        absorbedPowerDensity = Grid->computeAbsorbedPowerDens(fields.data(), real_conductivity, num_pts);  // Using Joule Heating
        plasma_pabs.resize(num_pts);
        std::copy(absorbedPowerDensity, absorbedPowerDensity + plasma_pabs.size(), plasma_pabs.begin());

        double* E_amplitude = new double[num_pts];
        Grid->calculateFieldAmplitudes(fields.data(), E_amplitude, num_pts);
        plasma_E_field.resize(num_pts);
        std::copy(E_amplitude, E_amplitude + num_pts, plasma_E_field.begin());
        delete[] E_amplitude;
        
        if(rank == 0){
            if(config.storeResults){
                if(config.printProgress) std::cout << "Storing results" << std::endl;

                // Helper function to check if a field should be exported
                auto shouldExport = [](const std::string& fieldName, const std::vector<std::string>& outputFields) {
                    return std::find(outputFields.begin(), outputFields.end(), fieldName) != outputFields.end();
                };            


                if(shouldExport("PowerFlux",config.outputFields)){
                    double* P_flux = new double[3*num_pts];
                    double* P_poynting = new double[num_pts];
                    double* P_flux_amplitude = new double[num_pts];
                    double* P_flux_x = new double[num_pts];
                    double* P_flux_y = new double[num_pts];
                    double* P_flux_z = new double[num_pts];

                    Grid->calculateFlux(fields.data(), P_flux, angular_frequency, num_pts);
                    Grid->calculatePoynting(P_flux, plasma_pabs.data(), P_poynting, num_pts);
                    
                    Grid->calculateFieldAmplitudes(P_flux, P_flux_amplitude, num_pts);
                    Grid->getFieldComponents(P_flux, P_flux_x, P_flux_y, P_flux_z, num_pts);

                    exportField3D(P_flux_amplitude, config.outputDirectory + "power_flux_amp.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);    
                    exportField3D(P_poynting, config.outputDirectory + "power_poynting.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ , non_uniform_grid, Grid->x, Grid->y, Grid->z);

                    exportField3D(P_flux_x, config.outputDirectory + "power_flux_x.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z); 
                    exportField3D(P_flux_y, config.outputDirectory + "power_flux_y.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);  
                    exportField3D(P_flux_z, config.outputDirectory + "power_flux_z.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    
                    delete[] P_flux_amplitude; delete[] P_flux_x; delete[] P_flux_y; delete[] P_flux_z; delete[] P_flux; delete[] P_poynting;
                }
                if(shouldExport("AbsorbedPower",config.outputFields)){
                    exportField3D(plasma_pabs.data(), config.outputDirectory + "absorbed_power_density.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);    
                }
                if(shouldExport("Inputs",config.outputFields)){
                    exportField3D(electron_density, config.outputDirectory + "electron_density.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ , non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(real_mobility, config.outputDirectory + "real_mobility.txt", config.n_x, config.n_y, config.n_z,
                            config.lengthX, config.lengthY, config.lengthZ , non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(imag_mobility, config.outputDirectory + "imag_mobility.txt", config.n_x, config.n_y, config.n_z,
                            config.lengthX, config.lengthY, config.lengthZ , non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    if((int) config.sourceLocations.size() == num_pts) {
                        std::vector<double> source_abs = complexToAbsArray(config.sourceValues, 0);
                        exportField3D(source_abs.data(), config.outputDirectory + "source_values.txt", config.n_x, config.n_y, config.n_z,
                            config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    }
                }
                if(shouldExport("FieldAmplitudes",config.outputFields)){
                    double* E_amplitude = new double[num_pts];
                    Grid->calculateFieldAmplitudes(fields.data(), E_amplitude, num_pts);

                    exportField3D(E_amplitude, config.outputDirectory + "E_amplitude.txt", config.n_x, config.n_y, config.n_z, 
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    
                    delete[] E_amplitude;
                }
                if(shouldExport("FieldRealPart",config.outputFields)){
                    double* E_real = new double[num_pts];
                    double* E_imag = new double[num_pts];
                    Grid->calculateFieldRealPart(fields.data(), E_real, num_pts);
                    Grid->calculateFieldImagPart(fields.data(), E_imag, num_pts);

                    exportField3D(E_real, config.outputDirectory + "E_real.txt", config.n_x, config.n_y, config.n_z, 
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(E_imag, config.outputDirectory + "E_imag.txt", config.n_x, config.n_y, config.n_z, 
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                
                    delete[] E_real;    delete[] E_imag;
                }
                if(shouldExport("FieldComponents",config.outputFields)){
                    double* Ex_amp = new double[num_pts];
                    double* Ey_amp = new double[num_pts];
                    double* Ez_amp = new double[num_pts];
                    Grid->getFieldComponents(fields.data(), Ex_amp, Ey_amp, Ez_amp, num_pts);

                    exportField3D(Ex_amp, config.outputDirectory + "Ex_amp.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(Ey_amp, config.outputDirectory + "Ey_amp.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(Ez_amp, config.outputDirectory + "Ez_amp.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);

                    delete[] Ex_amp; delete[] Ey_amp; delete[] Ez_amp;
                }
                if(shouldExport("F_Grad_Components",config.outputFields)){
                    double* f_grad_cond_x = new double[num_pts];
                    double* f_grad_cond_y = new double[num_pts];
                    double* f_grad_cond_z = new double[num_pts];
                    Grid->getFieldComponents(f_grad_cond.data(),f_grad_cond_x, f_grad_cond_y, f_grad_cond_z, num_pts);

                    exportField3D(f_grad_cond_x, config.outputDirectory + "f_grad_cond_x.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(f_grad_cond_y, config.outputDirectory + "f_grad_cond_y.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                    exportField3D(f_grad_cond_z, config.outputDirectory + "f_grad_cond_z.txt", config.n_x, config.n_y, config.n_z,
                        config.lengthX, config.lengthY, config.lengthZ, non_uniform_grid, Grid->x, Grid->y, Grid->z);
                
                    delete[] f_grad_cond_x; delete[] f_grad_cond_y; delete[] f_grad_cond_z;
                }
            }

            if(print_times){
                PetscTime(&endTime);
                PetscPrintf(PETSC_COMM_WORLD, "Time taken on Storing Results and Calculating Pabs: %.3f seconds\n", endTime - previous_time);
                previous_time = endTime;
            }
            if(config.printProgress) std::cout << "End of Program" << std::endl;

        }
        PetscTime(&endTime);
        PetscPrintf(PETSC_COMM_WORLD, "Time taken: %.3f seconds\n", endTime - startTime);


        delete[] electron_density; delete[] real_mobility; delete[] imag_mobility;
        delete[] real_conductivity; delete[] complex_conductivity; delete[] complex_permittivity; delete[] real_permittivity;

        VecDestroy(&b_vector);
        MatDestroy(&systemMaxwell);
        
        return  0;
    } catch (const std::exception& error_config) {
        std::cerr << "Error: " << error_config.what() << std::endl;

        if (b_vector) VecDestroy(&b_vector);
        if (x_seq) VecDestroy(&x_seq);
        if (scatter) VecScatterDestroy(&scatter);
        if (systemMaxwell) MatDestroy(&systemMaxwell);

        return 1;
    }

    return -1;
}


void interpolate_rz_to_yz(const std::vector<double>& y,const std::vector<double>& z, std::vector<std::tuple<int, int, int>>& plasma_locations,
                        std::vector<double>& plasma_ne, std::vector<double>& plasma_mu_re, std::vector<double>& plasma_mu_im,
                        const std::vector<double>& amrex_n_e, const std::vector<double>& amrex_mu_re, const std::vector<double>& amrex_mu_im,
                        int Nr, int Nz, const double* prob_lo, const double* dx, double y_c, double R_in, double z_0) {
    plasma_locations.clear();
    plasma_ne.clear();
    plasma_mu_re.clear();
    plasma_mu_im.clear();

    for (size_t m = 0; m < y.size(); ++m) {
        double r_target = std::abs(y[m] - y_c);

        if (r_target > R_in) continue;

        for (size_t n = 0; n < z.size(); ++n) {
            double z_target = z[n] + z_0;

            // AMReX grid
            double f_i = (r_target - prob_lo[0]) / dx[0] - 0.5;
            double f_j = (z_target - prob_lo[1]) / dx[1] - 0.5;

            // Identify the 4 surrounding bounding cells
            int i0 = std::floor(f_i);
            int j0 = std::floor(f_j);

            // Ensure indices stay within safely interpolatable limits [0, N-2]
            i0 = std::clamp(i0, 0, Nr - 2);
            j0 = std::clamp(j0, 0, Nz - 2);

            int i1 = i0 + 1;
            int j1 = j0 + 1;

            double dr = std::clamp(f_i - i0, 0.0, 1.0);
            double dz = std::clamp(f_j - j0, 0.0, 1.0);

            // Standard Bilinear Interpolation
            auto bilinear_interp = [&](const std::vector<double>& field) {
                double v00 = field[i0 + j0 * Nr]; // Bottom-Left
                double v10 = field[i1 + j0 * Nr]; // Bottom-Right
                double v01 = field[i0 + j1 * Nr]; // Top-Left
                double v11 = field[i1 + j1 * Nr]; // Top-Right

                return (1.0 - dr) * (1.0 - dz) * v00 + dr * (1.0 - dz) * v10 + (1.0 - dr) * dz * v01 + dr * dz * v11;
            };

            plasma_locations.push_back(std::make_tuple(0, static_cast<int>(m), static_cast<int>(n)));
            plasma_ne.push_back(bilinear_interp(amrex_n_e));
            plasma_mu_re.push_back(bilinear_interp(amrex_mu_re));
            plasma_mu_im.push_back(bilinear_interp(amrex_mu_im));
        }
    }
}
