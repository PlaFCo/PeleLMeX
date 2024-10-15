#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#include <PeleLMeX_EF_Constants.H>
#include <AMReX_FillPatchUtil.H>
#include <PeleLMeX_BCfill.H>

using namespace amrex;

void
PeleLM::ionDriftVelocity(std::unique_ptr<AdvanceAdvData>& advData)
{
  //----------------------------------------------------------------
  // set udrift boundaries to zero
  if (advData->uDrift[0][0].nGrow() > 0) {
    for (int lev = 0; lev <= finest_level; ++lev) {
      AMREX_D_TERM(advData->uDrift[lev][0].setBndry(0.0);
                   , advData->uDrift[lev][1].setBndry(0.0);
                   , advData->uDrift[lev][2].setBndry(0.0););
    }
  }

  //----------------------------------------------------------------
  // Get the gradient of Old and New phiV
  Vector<Array<MultiFab, AMREX_SPACEDIM>> EOld(finest_level + 1);
  Vector<Array<MultiFab, AMREX_SPACEDIM>> ENew(finest_level + 1);
  int nGrow = 0; // No need for ghost face on gphiV
  for (int lev = 0; lev <= finest_level; ++lev) {
    const auto& ba = grids[lev];
    const auto& factory = Factory(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
      EOld[lev][idim].define(
        amrex::convert(ba, IntVect::TheDimensionVector(idim)), dmap[lev], 1,
        nGrow, MFInfo(), factory);
      ENew[lev][idim].define(
        amrex::convert(ba, IntVect::TheDimensionVector(idim)), dmap[lev], 1,
        nGrow, MFInfo(), factory);
    }
  }

  if (m_ef_model == EFModel::EFglobal) { // E is grad PhiV
    int do_avgDown = 0;                  // TODO or should I ?
    auto bcRecPhiV = fetchBCRecArray(PHIV, 1);
    getDiffusionOp()->computeGradient(
      GetVecOfArrOfPtrs(EOld), {}, // don't need the laplacian out
      GetVecOfConstPtrs(getPhiVVect(AmrOldTime)), bcRecPhiV[0], do_avgDown);
    getDiffusionOp()->computeGradient(
      GetVecOfArrOfPtrs(ENew), {}, // don't need the laplacian out
      GetVecOfConstPtrs(getPhiVVect(AmrNewTime)), bcRecPhiV[0], do_avgDown);
  } else if (m_ef_model == EFModel::EFlocal) { // Eamb
    for (int lev = 0; lev <= finest_level; ++lev) {
      //---------------------------------------------------------------
      // Compute the old and new charge distribution
      //---------------------------------------------------------------
      const auto geomdata = geom[lev].data();
      int nGhost = 1;
      auto ChargeOld_CC = MultiFab(grids[lev], dmap[lev],
                              1, nGhost, MFInfo(), *m_factory[lev]);
      auto ChargeNew_CC = MultiFab(grids[lev], dmap[lev],
                              1, nGhost, MFInfo(), *m_factory[lev]);

      // Get level data
      auto ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
      auto ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(ChargeOld_CC, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        auto const& rhoYOld =  ldataOld_p->state.const_array(mfi, FIRSTSPEC);
        auto const& rhoYNew =  ldataNew_p->state.const_array(mfi, FIRSTSPEC);
        //auto const& nEOld = ldataOld_p->state.const_array(mfi, NE);
        //auto const& nENew = ldataNew_p->state.const_array(mfi, NE);
        auto const& ChO = ChargeOld_CC.array(mfi);
        auto const& ChN = ChargeNew_CC.array(mfi);
        Real factor = 1.0 / (eps0*epsr);
        amrex::ParallelFor(
          bx, [ChO, ChN, rhoYOld, rhoYNew, factor,
               zk = zk] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            ChO(i, j, k) = 0.0; //-nEOld(i, j, k) * elemCharge * factor;
            ChN(i, j, k) = 0.0; //-nEOld(i, j, k) * elemCharge * factor;
            for (int n = 0; n < NUM_SPECIES; n++) {
              ChO(i, j, k) += zk[n] * rhoYOld(i, j, k, n) * factor;
              ChN(i, j, k) += zk[n] * rhoYNew(i, j, k, n) * factor;
            }
          });
      }

      //---------------------------------------------------------------
      // Compute EField
      //---------------------------------------------------------------
      const amrex::Real* dx = geomdata.CellSize();
      for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        EOld[lev][idim].setVal(0.0);
        ENew[lev][idim].setVal(0.0);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(EOld[lev][idim], TilingIfNotGPU()); mfi.isValid();
             ++mfi) {
          const Box bx = mfi.tilebox();
          const auto& ef_old = EOld[lev][idim].array(mfi);
          const auto& ef_new = ENew[lev][idim].array(mfi);
          const auto& ChO = ChargeOld_CC.const_array(mfi);
          const auto& ChN = ChargeNew_CC.const_array(mfi);
          if (idim == 0) {
            amrex::ParallelFor(
              bx, 
              [ef_old, ef_new, ChO, ChN, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                ef_old(i, j, k) = (ChO(i, j, k) - ChO(i-1, j, k));///dx[0];
                ef_new(i, j, k) = (ChN(i, j, k) - ChN(i-1, j, k));///dx[0];
              });
          } else if (idim == 1) {
            amrex::ParallelFor(
              bx, 
              [ef_old, ef_new, ChO, ChN, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                ef_old(i, j, k) = (ChO(i, j, k) - ChO(i, j-1, k));///dx[1];
                ef_new(i, j, k) = (ChN(i, j, k) - ChN(i, j-1, k));///dx[1];
              });
          } else if (idim == 2) {
            amrex::ParallelFor(
              bx, 
              [ef_old, ef_new, ChO, ChN, dx] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                ef_old(i, j, k) = (ChO(i, j, k) - ChO(i, j, k-1));///dx[2];
                ef_new(i, j, k) = (ChN(i, j, k) - ChN(i, j, k-1));///dx[2];
              });
          }
        }
        //EOld[lev][idim].setVal(0.0);
        //ENew[lev][idim].setVal(0.0);
      }
    }
  
    //---------------------------------------------------------------
    // Average down efield faces to make consistent across levels
    //---------------------------------------------------------------
    // Average down the fluxes
    for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
      EB_average_down_faces(
        GetArrOfConstPtrs(EOld[lev]), GetArrOfPtrs(EOld[lev - 1]), 
        refRatio(lev - 1), geom[lev - 1]); 
      EB_average_down_faces(
        GetArrOfConstPtrs(ENew[lev]), GetArrOfPtrs(ENew[lev - 1]), 
        refRatio(lev - 1), geom[lev - 1]); 
#else
      average_down_faces(
        GetArrOfConstPtrs(EOld[lev]), GetArrOfPtrs(EOld[lev - 1]), 
        refRatio(lev - 1), geom[lev - 1]); 
      average_down_faces(
        GetArrOfConstPtrs(ENew[lev]), GetArrOfPtrs(ENew[lev - 1]), 
        refRatio(lev - 1), geom[lev - 1]); 
#endif
    }

    //Vector<std::unique_ptr<MultiFab>> EF_CC(finest_level + 1);
    //for (int lev = 0; lev <= finest_level; ++lev) {
    //  EF_CC[lev].reset(new MultiFab(
    //    grids[lev], dmap[lev], AMREX_SPACEDIM, 0, MFInfo(), *m_factory[lev]));
    //  EF_CC[lev]->setVal(0.0);
    //  average_face_to_cellcenter(
    //    *EF_CC[lev], 0, GetArrOfConstPtrs(ENew[lev]));
    //}
    //WriteDebugPlotFile(GetVecOfConstPtrs(EF_CC),"plt_EF_CCNew_test_"+std::to_string(m_nstep));
    //amrex::Abort();
  } else {
    for (int lev = 0; lev <= finest_level; ++lev) {
      for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        EOld[lev][idim].setVal(0.0);
        ENew[lev][idim].setVal(0.0);
      }
    }
  }

  //----------------------------------------------------------------
  // TODO : this assumes that all the ions are grouped together at th end ...
  auto bcRecIons =
    fetchBCRecArray(FIRSTSPEC + NUM_SPECIES - NUM_IONS, NUM_IONS);

  for (int lev = 0; lev <= finest_level; ++lev) {
    // Get CC ions t^{n+1/2} mobilities
    // TODO In the old version, there's a switch to only use an instant. value
    auto ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto ldataNew_p = getLevelDataPtr(lev, AmrNewTime);

    MultiFab mobH_cc(grids[lev], dmap[lev], NUM_IONS, 1);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mobH_cc, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& gbx = mfi.growntilebox();
      const auto& mob_o = ldataOld_p->mob_cc.const_array(mfi);
      const auto& mob_n = ldataNew_p->mob_cc.const_array(mfi);
      const auto& mob_h = mobH_cc.array(mfi);
      amrex::ParallelFor(
        gbx, NUM_IONS,
        [mob_o, mob_n,
         mob_h] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          mob_h(i, j, k, n) = 0.5 * (mob_o(i, j, k, n) + mob_n(i, j, k, n));
        });
    }

    // Get the face centered ions mobility
    int doZeroVisc = 0;
    Array<MultiFab, AMREX_SPACEDIM> mobH_ec =
      getDiffusivity(lev, 0, NUM_IONS, doZeroVisc, bcRecIons, mobH_cc);

    // Assemble the ions drift velocity
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
      for (MFIter mfi(mobH_ec[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box bx = mfi.tilebox();
        const auto& mob_h = mobH_ec[idim].const_array(mfi);
        const auto& gp_o = EOld[lev][idim].const_array(mfi);
        const auto& gp_n = ENew[lev][idim].const_array(mfi);
        const auto& Ud_Sp = advData->uDrift[lev][idim].array(mfi);
        amrex::ParallelFor(
          bx, NUM_IONS,
          [mob_h, gp_o, gp_n,
           Ud_Sp] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
            Ud_Sp(i, j, k, n) =
              mob_h(i, j, k, n) * -0.5 * (gp_o(i, j, k) + gp_n(i, j, k));
          });
      }
    }
  }

  //----------------------------------------------------------------
  // Average down faces
  for (int lev = finest_level; lev > 0; --lev) {
#ifdef AMREX_USE_EB
    EB_average_down_faces(
      GetArrOfConstPtrs(advData->uDrift[lev]),
      GetArrOfPtrs(advData->uDrift[lev - 1]), refRatio(lev - 1), geom[lev - 1]);
#else
    average_down_faces(
      GetArrOfConstPtrs(advData->uDrift[lev]),
      GetArrOfPtrs(advData->uDrift[lev - 1]), refRatio(lev - 1), geom[lev - 1]);
#endif
  }

  // FillPatch Udrift on levels > 0
  for (int lev = 0; lev <= finest_level; ++lev) {
    if (lev > 0) {
      IntVect rr = geom[lev].Domain().size() / geom[lev - 1].Domain().size();
      Interpolater* mapper = &face_linear_interp;

      // Set BCRec for Umac
      Vector<BCRec> bcrec(NUM_IONS);
      for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
        for (int ion = 0; ion < NUM_IONS; ion++) {
          if (geom[lev - 1].isPeriodic(idim)) {
            bcrec[ion].setLo(idim, BCType::int_dir);
            bcrec[ion].setHi(idim, BCType::int_dir);
          } else {
            bcrec[ion].setLo(idim, BCType::foextrap);
            bcrec[ion].setHi(idim, BCType::foextrap);
          }
        }
      }
      Array<Vector<BCRec>, AMREX_SPACEDIM> bcrecArr = {
        AMREX_D_DECL(bcrec, bcrec, bcrec)};

      PhysBCFunct<GpuBndryFuncFab<umacFill>> crse_bndry_func(
        geom[lev - 1], bcrec, umacFill{});
      Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
        cbndyFuncArr = {
          AMREX_D_DECL(crse_bndry_func, crse_bndry_func, crse_bndry_func)};

      PhysBCFunct<GpuBndryFuncFab<umacFill>> fine_bndry_func(
        geom[lev], bcrec, umacFill{});
      Array<PhysBCFunct<GpuBndryFuncFab<umacFill>>, AMREX_SPACEDIM>
        fbndyFuncArr = {
          AMREX_D_DECL(fine_bndry_func, fine_bndry_func, fine_bndry_func)};

      Real dummy = 0.;
      FillPatchTwoLevels(
        GetArrOfPtrs(advData->uDrift[lev]), IntVect(1), dummy,
        {GetArrOfPtrs(advData->uDrift[lev - 1])}, {dummy},
        {GetArrOfPtrs(advData->uDrift[lev])}, {dummy}, 0, 0, NUM_IONS,
        geom[lev - 1], geom[lev - 1], cbndyFuncArr, 0, fbndyFuncArr, 0, rr,
        mapper, bcrecArr, 0);
    } else {
      AMREX_D_TERM(
        advData->uDrift[lev][0].FillBoundary(geom[lev].periodicity());
        , advData->uDrift[lev][1].FillBoundary(geom[lev].periodicity());
        , advData->uDrift[lev][2].FillBoundary(geom[lev].periodicity()));
    }
  }
}

void
PeleLM::ionDriftAddUmac(int lev, std::unique_ptr<AdvanceAdvData>& advData)
{
  // Add umac to the ions drift velocity to get the effective velocity
  for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(advData->umac[lev][idim], TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const Box gbx = mfi.growntilebox();
      const auto& umac = advData->umac[lev][idim].const_array(mfi);
      const auto& Ud_Sp = advData->uDrift[lev][idim].array(mfi);
      amrex::ParallelFor(
        gbx, NUM_IONS,
        [umac, Ud_Sp] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          Ud_Sp(i, j, k, n) += umac(i, j, k);
        });
    }
    advData->uDrift[lev][idim].FillBoundary(geom[lev].periodicity());
  }
}
