#ifndef PELELMEX_ELECTRONENERGY_H
#define PELELMEX_ELECTRONENERGY_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>

#include <PhysicsConstants.H>

namespace pele::electron_energy
{

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
numberDensity(
  const amrex::Real rho_e,
  const amrex::Real W_e) noexcept
{
  // rho_e : kg/m^3
  // W_e   : g/mol = kg/kmol numerically
  // Avna  : particles/mol
  //
  // rho_e * 1000 / W_e -> mol/m^3
  return rho_e * 1.0e3 / W_e *
         pele::physics::Constants::Avna;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
temperature(
  const amrex::Real Ee,
  const amrex::Real rho_e,
  const amrex::Real W_e,
  const amrex::Real T_h) noexcept
{
  // No electrons: Te is undefined physically.
  // Use the heavy-particle temperature as a benign convention.
  if (rho_e <= 0.0) {
    return T_h;
  }

  const amrex::Real ne = numberDensity(rho_e, W_e);

  // PelePhysics RU is in erg/mol/K.
  // 1 erg = 1e-7 J.
  constexpr amrex::Real kB =
    1.0e-7 * pele::physics::Constants::RU /
    pele::physics::Constants::Avna;

  return 2.0 * Ee / (3.0 * ne * kB);
}

} // namespace pele::electron_energy

#endif