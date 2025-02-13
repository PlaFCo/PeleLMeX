#include <PeleLMeX.H>
#include <PeleLMeX_EF_Constants.H>

using namespace amrex;

Real
PeleLM::estEFIonsDt(const TimeStamp& a_time)
{
  Real estdt = 1.0e200;

  if (m_ef_model == EFModel::EFglobal) { 
    estdt = estEFglobIonsDt(a_time);
  } else if (m_ef_model == EFModel::EFlocal) {
    //estdt = estEFlocIonsDt(a_time);
    estdt = estEFglobIonsDt(a_time);
  }

  return estdt;
}

Real 
PeleLM::estEFlocIonsDt(const TimeStamp& a_time)
{
  Real estdt = 1.0e200;

  // Need the mobility of the ions
  calcDiffusivity(a_time);

  for (int lev = 0; lev <= finest_level; ++lev) {

    Real estdt_lev = 1.0e200;

    //---------------------------------------------------------------
    // Compute CC charge distribution
    //---------------------------------------------------------------
    int nGhost = 1;
    MultiFab Charge_CC = MultiFab(grids[lev], dmap[lev],
                            1, nGhost, MFInfo(), *m_factory[lev]);
    MultiFab E_CC = MultiFab(grids[lev], dmap[lev],
                         AMREX_SPACEDIM, 0, MFInfo(), *m_factory[lev]);

    MultiFab driftVelMax_cc(grids[lev], dmap[lev], 1, 0,
                        MFInfo(), *m_factory[lev]);

    // Get level data
    auto ldata_p = getLevelDataPtr(lev, a_time);
    const auto dx = Geom(lev).CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Charge_CC, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.growntilebox();
      auto const& rhoY =  ldata_p->state.const_array(mfi, FIRSTSPEC);
      auto const& ChO = Charge_CC.array(mfi);
      amrex::ParallelFor(
        bx, [ChO, rhoY,
             zk = zk] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          ChO(i, j, k) = 0.0;
          for (int n = 0; n < NUM_SPECIES; n++) {
            ChO(i, j, k) += zk[n] * rhoY(i, j, k, n);
          }
        });
    }

    //---------------------------------------------------------------
    // Compute EField
    //---------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(E_CC, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        const Box bx = mfi.tilebox();
        const auto& ChO = Charge_CC.const_array(mfi);
        const auto& ef_old = E_CC.array(mfi,idim);
        Real factor = 1.0 / (eps0*epsr);
        if (idim == 0) {
          amrex::ParallelFor(
            bx, 
            [ef_old, ChO, dx, factor] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              ef_old(i, j, k) = 0.5 * factor * ( ( ChO(i  , j, k) - ChO(i-1, j, k) ) +
                                                 ( ChO(i+1, j, k) - ChO(i  , j, k) ) );///dx[0];
            });
        } else if (idim == 1) {
          amrex::ParallelFor(
            bx, 
            [ef_old, ChO, dx, factor] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              ef_old(i, j, k) = 0.5 * factor * ( ( ChO(i, j  , k) - ChO(i, j-1, k) ) +
                                                 ( ChO(i, j+1, k) - ChO(i, j  , k) ) );///dx[1];
            });
        } else if (idim == 2) {
          amrex::ParallelFor(
            bx, 
            [ef_old, ChO, dx, factor] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
              ef_old(i, j, k) = 0.5 * factor * ( ( ChO(i, j, k  ) - ChO(i, j, k-1) ) +
                                                 ( ChO(i, j, k+1) - ChO(i, j, k  ) ) );///dx[2];
            });
        }
      }
    }

    // Get cell centered max effective velocities across
    // all dimension/ions
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& vel = ldata_p->state.const_array(mfi, VELX);
      auto const& efield = E_CC.const_array(mfi);
      auto const& mob_cc = ldata_p->mob_cc.const_array(mfi);
      auto const& uDrMax = driftVelMax_cc.array(mfi);
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          Real maxVel = 0.0;
          for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            for (int n = 0; n < NUM_IONS; n++) {
              Real ueff =
                vel(i, j, k, idim) + mob_cc(i, j, k, n) * efield(i, j, k, idim);
              maxVel = amrex::max(maxVel, std::abs(ueff));
            }
          }
          uDrMax(i, j, k) = maxVel;
        });
    }

    amrex::Real cfl_lcl = m_cfl;
    estdt_lev = amrex::ReduceMin(
      driftVelMax_cc, 0,
      [dx, cfl_lcl] AMREX_GPU_HOST_DEVICE(
        Box const& bx, Array4<Real const> const& ueffm) noexcept -> Real {
        using namespace amrex::literals;
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || \
  (__CUDACC_VER_MINOR__ != 2)
        amrex::Real velmax = std::numeric_limits<amrex::Real>::min();
#else
        amrex::Real velmax = -1.e37_rt;
#endif
        for (int k = lo.z; k <= hi.z; ++k) {
          for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
              velmax = amrex::max(velmax, ueffm(i, j, k));
            }
          }
        }
        return dx[0] / velmax * cfl_lcl;
      });

    // Min across levels
    estdt = std::min(estdt, estdt_lev);
  }

  // Min across processors
  ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}

Real 
PeleLM::estEFglobIonsDt(const TimeStamp& a_time)
{
  Real estdt = 1.0e200;
  constexpr Real small = 1.0e-8;

  auto bcRecPhiV = fetchBCRecArray(PHIV, 1);

  // Need the mobility of the ions
  calcDiffusivity(a_time);

  for (int lev = 0; lev <= finest_level; ++lev) {

    Real estdt_lev = 1.0e200;

    // Get cell centered gradient of phiV
    auto ldata_p = getLevelDataPtr(lev, a_time);
    MultiFab efield_cc(grids[lev], dmap[lev], AMREX_SPACEDIM, 0);
    MultiFab driftVelMax_cc(grids[lev], dmap[lev], 1, 0);

    const auto dxinv = Geom(lev).InvCellSizeArray();
    const auto domain = Geom(lev).Domain();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& phiV = ldata_p->state.const_array(mfi, PHIV);
      auto const& efield = efield_cc.array(mfi, 0);

      // X
      auto bc_lo = bcRecPhiV[0].lo(0);
      auto bc_hi = bcRecPhiV[0].hi(0);
      amrex::Real factor = -0.5 * dxinv[0];
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) && i <= domain.smallEnd(0));
          bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) && i >= domain.bigEnd(0));
          efield(i, j, k, 0) = factor * (phiV(i + 1, j, k) - phiV(i - 1, j, k));
          if (on_lo)
            efield(i, j, k, 0) = factor * (phiV(i + 1, j, k) + phiV(i, j, k) -
                                           2.0 * phiV(i - 1, j, k));
          if (on_hi)
            efield(i, j, k, 0) = factor * (2.0 * phiV(i + 1, j, k) -
                                           phiV(i, j, k) - phiV(i - 1, j, k));
        });

#if (AMREX_SPACEDIM > 1)
      // Y
      bc_lo = bcRecPhiV[0].lo(1);
      bc_hi = bcRecPhiV[0].hi(1);
      factor = -0.5 * dxinv[1];
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) && j <= domain.smallEnd(1));
          bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) && j >= domain.bigEnd(1));
          efield(i, j, k, 1) = factor * (phiV(i, j + 1, k) - phiV(i, j - 1, k));
          if (on_lo)
            efield(i, j, k, 1) = factor * (phiV(i, j + 1, k) + phiV(i, j, k) -
                                           2.0 * phiV(i, j - 1, k));
          if (on_hi)
            efield(i, j, k, 1) = factor * (2.0 * phiV(i, j + 1, k) -
                                           phiV(i, j, k) - phiV(i, j - 1, k));
        });

#if (AMREX_SPACEDIM > 2)
      // Z
      bc_lo = bcRecPhiV[0].lo(2);
      bc_hi = bcRecPhiV[0].hi(2);
      factor = -0.5 * dxinv[2];
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          bool on_lo =
            ((bc_lo == amrex::BCType::ext_dir) && k <= domain.smallEnd(2));
          bool on_hi =
            ((bc_hi == amrex::BCType::ext_dir) && k >= domain.bigEnd(2));
          efield(i, j, k, 2) = factor * (phiV(i, j, k + 1) - phiV(i, j, k - 1));
          if (on_lo)
            efield(i, j, k, 2) = factor * (phiV(i, j, k + 1) + phiV(i, j, k) -
                                           2.0 * phiV(i, j, k - 1));
          if (on_hi)
            efield(i, j, k, 2) = factor * (2.0 * phiV(i, j, k + 1) -
                                           phiV(i, j, k) - phiV(i, j, k - 1));
        });
#endif
#endif
    }

    // Get cell centered max effective velocities across
    // all dimension/ions
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& vel = ldata_p->state.const_array(mfi, VELX);
      auto const& efield = efield_cc.const_array(mfi);
      auto const& mob_cc = ldata_p->mob_cc.const_array(mfi);
      auto const& uDrMax = driftVelMax_cc.array(mfi);
      amrex::ParallelFor(
        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          Real maxVel = 0.0;
          for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
            for (int n = 0; n < NUM_IONS; n++) {
              Real ueff =
                vel(i, j, k, idim) + mob_cc(i, j, k, n) * efield(i, j, k, idim);
              maxVel = amrex::max(maxVel, std::abs(ueff));
            }
          }
          uDrMax(i, j, k) = maxVel;
        });
    }

    const auto dx = Geom(lev).CellSizeArray();
    amrex::Real cfl_lcl = m_cfl;
    estdt_lev = amrex::ReduceMin(
      driftVelMax_cc, 0,
      [dx, cfl_lcl] AMREX_GPU_HOST_DEVICE(
        Box const& bx, Array4<Real const> const& ueffm) noexcept -> Real {
        using namespace amrex::literals;
        const auto lo = amrex::lbound(bx);
        const auto hi = amrex::ubound(bx);
#if !defined(__CUDACC__) || (__CUDACC_VER_MAJOR__ != 9) || \
  (__CUDACC_VER_MINOR__ != 2)
        amrex::Real velmax = std::numeric_limits<amrex::Real>::min();
#else
        amrex::Real velmax = -1.e37_rt;
#endif
        for (int k = lo.z; k <= hi.z; ++k) {
          for (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
              velmax = amrex::max(velmax, ueffm(i, j, k));
            }
          }
        }
        return dx[0] / velmax * cfl_lcl;
      });

    // Min across levels
    estdt = std::min(estdt, estdt_lev);
  }

  // Min across processors
  ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}
