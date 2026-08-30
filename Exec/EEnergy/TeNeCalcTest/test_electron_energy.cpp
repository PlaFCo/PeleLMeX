#include <cmath>
#include <iomanip>
#include <iostream>

#include <PhysicsConstants.H>
#include <PeleLMeX_ElectronEnergy.H>

int
main()
{
  using amrex::Real;

  // Electron molar mass:
  // g/mol, kg/kmol.
  constexpr Real We = 5.48579909065e-4;

  constexpr Real Th = 300.0;       // K
  constexpr Real Te_exact = 1.0e4; // K
  constexpr Real ne_exact = 1.0e18; // 1/m^3

  // PelePhysics RU is erg/mol/K.
  constexpr Real kB =
    1.0e-7 * pele::physics::Constants::RU /
    pele::physics::Constants::Avna;

  // ne = rho_e * 1e3 / We * NA
  // rho_e = ne * We / (1e3 * NA)
  const Real rho_e =
    ne_exact * We /
    (1.0e3 * pele::physics::Constants::Avna);

  // Ee from Te_exact
  const Real Ee =
    1.5 * ne_exact * kB * Te_exact;

  const Real ne =
    pele::electron_energy::numberDensity(rho_e, We);

  const Real Te =
    pele::electron_energy::temperature(
      Ee, rho_e, We, Th);

  const Real ne_rel_err =
    std::abs(ne - ne_exact) / ne_exact;

  const Real Te_rel_err =
    std::abs(Te - Te_exact) / Te_exact;

  constexpr Real rho_e_zero = 0.0;
  constexpr Real Ee_zero = 0.0;

  const Real Te_zero =
    pele::electron_energy::temperature(
      Ee_zero, rho_e_zero, We, Th);

  const Real fallback_err =
    std::abs(Te_zero - Th);

  std::cout << std::scientific
            << std::setprecision(16);

  std::cout
    << "ELECTRON ENERGY CONVERSION TEST\n";

  std::cout
    << "Constants\n"
    << "---------\n"
    << "We             = " << We << " g/mol\n"
    << "kB             = " << kB << " J/K\n\n";

  std::cout
    << "Nonzero-electron test\n"
    << "---------------------\n"
    << "rho_e          = " << rho_e << " kg/m^3\n"
    << "Ee             = " << Ee << " J/m^3\n"
    << "ne expected    = " << ne_exact << " 1/m^3\n"
    << "ne recovered   = " << ne << " 1/m^3\n"
    << "ne rel error   = " << ne_rel_err << "\n"
    << "Te expected    = " << Te_exact << " K\n"
    << "Te recovered   = " << Te << " K\n"
    << "Te rel error   = " << Te_rel_err << "\n\n";

  std::cout
    << "Zero-electron test\n"
    << "------------------\n"
    << "rho_e          = " << rho_e_zero << "\n"
    << "Ee             = " << Ee_zero << "\n"
    << "Th             = " << Th << " K\n"
    << "Te returned    = " << Te_zero << " K\n"
    << "fallback error = " << fallback_err << "\n\n";

  // Tight tol
  constexpr Real tol = 1.0e-12;

  const bool pass_ne = ne_rel_err < tol;
  const bool pass_Te = Te_rel_err < tol;
  const bool pass_zero = fallback_err < tol;

  std::cout
    << "Results\n"
    << "ne conversion : "
    << pass_ne << "\n"
    << "Te conversion : "
    << pass_Te << "\n"
    << "ne=0 fallback : "
    << pass_zero << "\n";

  return 1;
}