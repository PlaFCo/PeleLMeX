#include <PeleLMeX.H>
#include <PeleLMeX_K.H>


amrex::Real
PeleLM::computeDt(const int is_init, const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::computeDt()");

  amrex::Real estdt = 1.0e200;

  //----------------------------------------------------------------
  // Store prev dt(s)
  m_prev_dt = m_dt;

  //----------------------------------------------------------------
  // Compute dt estimate from level data
  if (m_fixed_dt > 0.0) {
    estdt = m_fixed_dt;
  } else {
    if (((is_init != 0) || m_nstep == 0) && m_init_dt > 0.0) {
      estdt = m_init_dt;
    } else {
      amrex::Real dtconv = estConvectiveDt(a_time);
      estdt = amrex::min<amrex::Real>(estdt, dtconv);
      amrex::Real dtdivU = 1.0e200;
      if ((m_incompressible == 0) && (m_has_divu != 0)) {
        dtdivU = estDivUDt(a_time);
        estdt = amrex::min<amrex::Real>(estdt, dtdivU);
      }

#ifdef PELE_USE_AXISWIRL
      amrex::Real dtswirl = estSwirlViscousDt(a_time);
      estdt = amrex::min<amrex::Real>(estdt, dtswirl);
#endif
#ifdef PELE_USE_PLASMA
      amrex::Real dtions = estEFIonsDt(a_time);
      estdt = amrex::min<amrex::Real>(estdt, dtions);
#endif
#ifdef PELE_USE_SPRAY
      amrex::Real dtspray = SprayEstDt();
      estdt = amrex::min<amrex::Real>(estdt, dtspray);
#endif
      if (m_verbose != 0) {
        amrex::Print() << " Est. time step - Conv: " << dtconv
                       << ", divu: " << dtdivU
#ifdef PELE_USE_AXISWIRL
                       << ", swirl_visc: " << dtswirl
#endif
#ifdef PELE_USE_PLASMA
                       << ", ions: " << dtions
#endif
#ifdef PELE_USE_SPRAY
                       << ", sprays: " << dtspray
#endif
                       << "\n";
      }
    }
  }

  //----------------------------------------------------------------
  // Limit dt
  if ((is_init != 0) || m_nstep == 0) {
    estdt *= m_dtshrink;
  } else {
    estdt = amrex::min<amrex::Real>(estdt, m_prev_dt * m_dtChangeMax);
    estdt = amrex::min<amrex::Real>(estdt, m_max_dt);
    // Shorten the dt to output plt file at exact req. time
    if (m_plot_per_exact > 0.0) {
      amrex::Real timeToNextPlot =
        (std::floor(m_cur_time / m_plot_per_exact) + 1) * m_plot_per_exact -
        m_cur_time;
      if (2.0 * estdt > timeToNextPlot && timeToNextPlot > estdt) {
        estdt = amrex::Real(0.5) * timeToNextPlot;
      } else {
        if (timeToNextPlot > 1.e-12) {
          estdt = amrex::min<amrex::Real>(estdt, timeToNextPlot);
        }
      }
    }
    // If we are getting close to the end of the simulation, shorten the dt
    if (m_stop_time >= 0.0) {
      amrex::Real timeLeft = (m_stop_time - m_cur_time);
      if (2.0 * estdt > timeLeft && timeLeft > estdt) {
        estdt = 0.5 * timeLeft;
      } else {
        estdt = amrex::min<amrex::Real>(estdt, timeLeft);
      }
    }
  }

  if (estdt < m_min_dt) {
    amrex::Print() << "\n";
    amrex::Print() << " ###################################### \n";
    amrex::Print() << " Estimated dt " << estdt << " is below allowed dt_min "
                   << m_min_dt << ": the simulation will stop ! \n";
    amrex::Print() << " ###################################### \n";
    amrex::Print() << "\n";
  }

  return estdt;
}

amrex::Real
PeleLM::estConvectiveDt(const TimeStamp a_time)
{

  amrex::Real estdt = 1.0e200;
  constexpr amrex::Real small = 1.0e-8;

  for (int lev = 0; lev <= finest_level; ++lev) {

    amrex::Real estdt_lev = 1.0e200;

    //----------------------------------------------------------------
    // Get level data ptr
    auto* ldata_p = getLevelDataPtr(lev, a_time);

    auto const dx = geom[lev].CellSizeArray();

    //----------------------------------------------------------------
    // Get velocity forces
    constexpr int nGrow_force = 0;
    amrex::MultiFab velForces(
      grids[lev], dmap[lev], AMREX_SPACEDIM, nGrow_force, amrex::MFInfo(),
      Factory(lev));

    constexpr int add_gradP = 1;
    getVelForces(a_time, lev, nullptr, &velForces, add_gradP);

    //----------------------------------------------------------------
    // Get max forces
    amrex::Vector<amrex::Real> f_max(AMREX_SPACEDIM);
    f_max = velForces.norm0({AMREX_D_DECL(0, 1, 2)}, 0, true, true);

    // Get max velocity
    amrex::Vector<amrex::Real> u_max(AMREX_SPACEDIM);
    u_max =
      ldata_p->state.norm0({AMREX_D_DECL(VELX, VELY, VELZ)}, 0, true, true);

    //----------------------------------------------------------------
    // Est. min time step on lev
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
      if (u_max[idim] > small) {
        estdt_lev = amrex::min<amrex::Real>(estdt_lev, dx[idim] / u_max[idim]);
      }
      if (f_max[idim] > small) {
        estdt_lev = amrex::min<amrex::Real>(
          estdt_lev, std::sqrt(2.0 * dx[idim] / f_max[idim]));
      }
    }

    //----------------------------------------------------------------
    // Set overall convective dt
    estdt = amrex::min<amrex::Real>(estdt, estdt_lev * m_cfl);
  }

  amrex::ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}

amrex::Real
PeleLM::estDivUDt(const TimeStamp a_time)
{

  amrex::Real estdt = 1.0e200;

  // Note: only methods 1 & 2 of PeleLM are available here
  AMREX_ASSERT(m_divu_checkFlag >= 0 && m_divu_checkFlag <= 2);

  for (int lev = 0; lev <= finest_level; ++lev) {

    auto* ldata_p = getLevelDataPtr(lev, a_time);
    std::unique_ptr<amrex::MultiFab> density =
      std::make_unique<amrex::MultiFab>(
        ldata_p->state, amrex::make_alias, DENSITY, 1);

    auto dtfac = m_divu_dtFactor;
    auto rhoMin = m_divu_rhoMin;
    if (m_divu_checkFlag == 1) {
      amrex::Real divu_dt = amrex::ReduceMin(
        *density, ldata_p->divu, 0,
        [dtfac, rhoMin] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx, amrex::Array4<amrex::Real const> const& rho,
          amrex::Array4<amrex::Real const> const& divu) -> amrex::Real {
          const auto lo = amrex::lbound(bx);
          const auto hi = amrex::ubound(bx);
          amrex::Real dt = 1.e37;
          for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
              for (int i = lo.x; i <= hi.x; ++i) {
                amrex::Real dtcell =
                  est_divu_dt_1(i, j, k, dtfac, rhoMin, rho, divu);
                dt = amrex::min<amrex::Real>(dt, dtcell);
              }
            }
          }
          return dt;
        });
      estdt = amrex::min<amrex::Real>(divu_dt, estdt);
    } else if (m_divu_checkFlag == 2) {
      const auto& dxinv = geom[lev].InvCellSizeArray();
      std::unique_ptr<amrex::MultiFab> velo = std::make_unique<amrex::MultiFab>(
        ldata_p->state, amrex::make_alias, VELX, AMREX_SPACEDIM);
      amrex::Real divu_dt = amrex::ReduceMin(
        *density, *velo, ldata_p->divu, 0,
        [dtfac, rhoMin, dxinv] AMREX_GPU_HOST_DEVICE(
          amrex::Box const& bx, amrex::Array4<amrex::Real const> const& rho,
          amrex::Array4<amrex::Real const> const& vel,
          amrex::Array4<amrex::Real const> const& divu) -> amrex::Real {
          const auto lo = amrex::lbound(bx);
          const auto hi = amrex::ubound(bx);
          amrex::Real dt = 1.e37;
          for (int k = lo.z; k <= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
              for (int i = lo.x; i <= hi.x; ++i) {
                amrex::Real dtcell =
                  est_divu_dt_2(i, j, k, dtfac, rhoMin, dxinv, rho, vel, divu);
                dt = amrex::min<amrex::Real>(dt, dtcell);
              }
            }
          }
          return dt;
        });
      estdt = amrex::min<amrex::Real>(divu_dt, estdt);
    }
  }

  amrex::ParallelDescriptor::ReduceRealMin(estdt);

  return estdt;
}

void
PeleLM::checkDt(const TimeStamp a_time, const amrex::Real a_dt)
{
  BL_PROFILE("PeleLMeX::checkDt()");

  if (m_fixed_dt > 0.0 || (m_divu_checkFlag == 0)) {
    return;
  }

  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata_p = getLevelDataPtr(lev, a_time);
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxinv =
      geom[lev].InvCellSizeArray();

    auto const& state_ma = ldata_p->state.const_arrays();
    auto const& divu_ma = ldata_p->divu.const_arrays();

    amrex::ParallelFor(
      ldata_p->state,
      [state_ma, divu_ma, dxinv, a_dt, divu_checkFlag = m_divu_checkFlag,
       dtfac = m_divu_dtFactor,
       rhoMin =
         m_divu_rhoMin] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        amrex::Array4<amrex::Real const> rho(state_ma[box_no], DENSITY);
        amrex::Array4<amrex::Real const> vel(state_ma[box_no], VELX);
        amrex::Array4<amrex::Real const> divu = divu_ma[box_no];
        check_divu_dt(
          i, j, k, divu_checkFlag, dtfac, rhoMin, dxinv, rho, vel, divu, a_dt);
      });
  }
  amrex::Gpu::streamSynchronize();
}



#ifdef PELE_USE_AXISWIRL
amrex::Real
PeleLM::estSwirlViscousDt(const TimeStamp a_time)
{
  BL_PROFILE("PeleLMeX::estSwirlViscousDt()");

  if (m_nAux <= 0 || m_angmom_aux < 0) {
    return 1.0e200;
  }

  amrex::Real dtswirl = 1.0e200;
  const amrex::Real cfl_visc = 0.5; // Viscous CFL safety factor (<= 0.5 for 2D explicit)

  for (int lev = 0; lev <= finest_level; ++lev) {
    auto* ldata = getLevelDataPtr(lev, a_time);
    const auto dx = geom[lev].CellSizeArray();
    const amrex::Real inv_dx2 = (1.0 / (dx[0] * dx[0])) + (1.0 / (dx[1] * dx[1]));

    amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
    amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInParallelRegion())
#endif
    for (amrex::MFIter mfi(ldata->state, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.tilebox();
      auto const& state_arr = ldata->state.const_array(mfi);
      auto const& visc_arr  = ldata->visc_cc.const_array(mfi);

      reduce_op.eval(bx, reduce_data,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept -> ReduceTuple
      {
        const amrex::Real rho = state_arr(i, j, k, DENSITY);
        const amrex::Real mu  = visc_arr(i, j, k, 0);
        const amrex::Real nu  = mu / rho;

        if (nu > 1.0e-12) {
          return cfl_visc / (2.0 * nu * inv_dx2);
        }
        return 1.0e200;
      });
    }

    amrex::Real dt_lev = amrex::get<0>(reduce_data.value(reduce_op));
    dtswirl = std::min(dtswirl, dt_lev);
  }

  amrex::ParallelDescriptor::ReduceRealMin(dtswirl);
  return dtswirl;
}
#endif