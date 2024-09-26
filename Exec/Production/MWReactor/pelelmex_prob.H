#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <pelelmex_prob_parm.H>
#include <PMFData.H>
#include <PelePhysics.H>


AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pelelmex_initdata(
  int i,
  int j,
  int k,
  int /*is_incompressible*/,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& /*aux*/,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real molafrac[NUM_SPECIES] = {0.0};
  amrex::Real vmax_l = 0;
  amrex::Real rz = 0;
  amrex::Real dist_to_center = 0.0;

  amrex::Real x[3] = {
    prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
    prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
    prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};


  vmax_l = prob_parm.vmax;

  dist_to_center += (x[0] - prob_parm.spark_locationx)
	    *(x[0] - prob_parm.spark_locationx)
	    /prob_parm.spark_radiusx/prob_parm.spark_radiusx;
  dist_to_center += (x[1] - prob_parm.spark_locationx)
	    *(x[1] - prob_parm.spark_locationx)
	    /prob_parm.spark_radiusx/prob_parm.spark_radiusx;
  dist_to_center += (x[2] - prob_parm.spark_locationz)
	    *(x[2] - prob_parm.spark_locationz)
	    /prob_parm.spark_radiusz/prob_parm.spark_radiusz;
  
  rz = sqrt((x[0]*x[0]+x[1]*x[1]));

  state(i, j, k, TEMP) = prob_parm.T_mean;

// case heat wave initialization (Temp profile calculated elsewhere)
// only with airthermal chemistry
//
//  if (rz < 0.013 and false) {
//    state(i, j, k, TEMP) = prob_parm.T_mean + 4800*(1.0-(x[0]*x[0]+x[1]*x[1])/0.013/0.013)*
//	    exp(-(x[2]-0.12)*(x[2]-0.12)/0.025/0.025);
//  }
//  else{
//    state(i, j, k, TEMP) = prob_parm.T_mean;
//  }
//  if ( rz < prob_parm.cyl_R+0.0005 or x[2] > 0.12 ) {
//    molafrac[O2_ID] = 1.0e-19;
//    molafrac[N_ID]  = 0.0; //std::max(0.0,-4.63e-3+3.65e-5*exp(1.565e-3*state(i, j, k, TEMP)));
//    molafrac[N2_ID] = 1.0-molafrac[O2_ID] - molafrac[N_ID];
//    for (int n = 0; n < NUM_SPECIES; n++) {
//      massfrac[n] = molafrac[n];
//    }
////    eos.X2Y(molafrac, massfrac);
//    
//  
//    if (rz<prob_parm.cyl_R) {
//      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
//                  , state(i, j, k, VELZ) =0.0);
//    }
//    else if (x[0]>0.0){
//      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
//                  , state(i, j, k, VELZ) =-1.0);
//    }
//    else {
//      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
//                  , state(i, j, k, VELZ) =-1.0);
//    }
//    //std::cout << "VMAX_L " << vmax_l;
//  
//    amrex::Real P_cgs = prob_parm.P_mean * 10.0;
//  
//    // Density
//    amrex::Real rho_cgs = 0.0;
//    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
//    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
//  
//    // Enthalpy
//    amrex::Real h_cgs = 0.0;
//    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
//    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);
//  
//    // Species mass
//    for (int n = 0; n < NUM_SPECIES; n++) {
//      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
//    }
//  }
//  else{
    massfrac[O2_ID] = 1.0;
    massfrac[N2_ID]  = 1.0-massfrac[O2_ID];
  
    if (rz<prob_parm.cyl_R) {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else if (x[0]>0.0){
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    else {
      AMREX_D_TERM(state(i, j, k, VELX) = 0.0;, state(i, j, k, VELY) =0.0; 
                  , state(i, j, k, VELZ) =0.0);
    }
    //std::cout << "VMAX_L " << vmax_l;
  
    amrex::Real P_cgs = prob_parm.P_mean * 10.0;
  
    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
  
    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
    state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);
  
    // Species mass
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
    }
// }

}

/*
AMRE_GPU_DEVICE
AMREX_FORCE_INLINE
*/

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const int /*m_nAux*/,
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real X_main[NUM_SPECIES] = {0.0};
  amrex::Real Y_main[NUM_SPECIES] = {0.0};
  amrex::Real X_rad[NUM_SPECIES] = {0.0};
  amrex::Real Y_rad[NUM_SPECIES] = {0.0};
  amrex::Real T_tan = 300.0;
  amrex::Real T_rad = 300.0;
  amrex::Real eta = 0.0;
  amrex::Real theta = 0.0; // angle of injection
  amrex::Real vmax_l = 10.0;
  amrex::Real tanvmax_l = 10.0;
  amrex::Real R_l = 0.45;
  amrex::Real rtrans_l = 0.2;
  amrex::Real tvelocity = 0.0;
  amrex::Real rad_pos = 0.0;
  amrex::Real zradeff = 0.0;
  bool in_zrange = false;
  bool in_xaxis = false;
  bool in_yaxis = false;
  bool is_O2;
  bool is_N2;

  const amrex::Real xi  = 0.013;
  const amrex::Real dxi = 0.001;
  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;
  amrex::Real ztan; 
  amrex::Real zrad; 
  amrex::Real cylt_R; 
  amrex::Real zhi; 
  amrex::Real cyl_R; 
  amrex::Real rtan; 


  // Simple, O2 into air injection
  Y_main[O2_ID] = 0.0;
  Y_main[N2_ID] = 1.0 - Y_main[O2_ID];
//Y_main[CH4_ID]=Y_main[CH4_ID]/(Y_main[CH4_ID]+Y_main[H2_ID]);
//  Y_main[H2_ID] =Y_main[H2_ID]/(Y_main[CH4_ID]+Y_main[H2_ID]);

  Y_rad[O2_ID]= 1.0;
  Y_rad[N2_ID] = 1.0 - Y_rad[O2_ID];
//  Y_rad[CH4_ID]=Y_rad[CH4_ID]/(Y_rad[CH4_ID]+Y_rad[H2_ID]);
//  Y_rad[H2_ID] =Y_rad[H2_ID] /(Y_rad[CH4_ID]+Y_rad[H2_ID]);


  //eos.X2Y(X_main, Y_main);
  //eos.X2Y(X_rad, Y_rad);

  if (idir == 0) {

    // Fill defaults: no slip wall, sort of.
    amrex::Real p_cgs = prob_parm.P_mean * 10.0;
    amrex::Real rho_cgs = 0.0;
    amrex::Real RhoH_temp = 0.0;
    amrex::Real massfrac_ch[NUM_SPECIES] = {0.0};
    // for (int n = 0; n < NUM_SPECIES; n++) {
    //   massfrac_ch[n] = s_ext[FIRSTSPEC + n]/s_ext[DENSITY];
    // }

    ztan = prob_parm.ztan;
    zrad = prob_parm.zrad;
    cylt_R = prob_parm.cylt_R;
    zhi  = geomdata.ProbHi(2);
    cyl_R = prob_parm.cyl_R;
    rtan = prob_parm.rtan;

    is_N2 = false;
    is_O2 = false;
    if (x[2] < 0.11){
     // is_O2 = true;
      zrad = zhi+prob_parm.ztan+zrad;
      rad_pos = sqrt((x[1]-0.0)*(x[1]-0.0) + (x[2]-zrad)*(x[2]-zrad));
      if (rad_pos < cylt_R){
        is_O2 = true;
      }
    }
    else {
      //is_N2 = true;
      if(x[1]>cyl_R-rtan-0.0035 and x[1]<cyl_R-0.0035 ){
        is_N2 = true;
      }
      else if (x[1]>-cyl_R+0.0035 and x[1]< -cyl_R+rtan+0.0035){
        is_N2 = true;
      }
    }


    if (is_O2){
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_rad, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_rad, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_rad[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = sgn*prob_parm.vmean_rad;
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;
    }
    else if (is_N2){
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }

      s_ext[VELX] = sgn*prob_parm.vmean_tan;
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;
    }
    else{
      s_ext[TEMP] = prob_parm.T_mean;
      eos.PYT2R(p_cgs, Y_main, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
      eos.TY2H(s_ext[TEMP], Y_main, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
      for (int n = 0; n < NUM_SPECIES; n++) {
        s_ext[FIRSTSPEC + n] = Y_main[n] * s_ext[DENSITY];
      }
      s_ext[VELX] = 0.0;
      s_ext[VELY] = 0.0;
      s_ext[VELZ] = 0.0;
    }

  }
  else {
      s_ext[TEMP] = prob_parm.T_mean;
  }


  // if (idir !=2){
  //   // Fill defaults: no slip wall, sort of.
  //   amrex::Real p_cgs = prob_parm.P_mean * 10.0;
  //   amrex::Real rho_cgs = 0.0;
  //   amrex::Real RhoH_temp = 0.0;
  //   amrex::Real massfrac_ch[NUM_SPECIES] = {0.0};
  //   for (int n = 0; n < NUM_SPECIES; n++) {
  //     massfrac_ch[n] = s_ext[FIRSTSPEC + n]/s_ext[DENSITY];
  //   }

  //   s_ext[TEMP] = prob_parm.T_mean;
  //   eos.PYT2R(p_cgs, massfrac_ch, s_ext[TEMP], rho_cgs);
  //   s_ext[DENSITY] = rho_cgs * 1.0e3;
  //   eos.TY2H(s_ext[TEMP], massfrac_ch, RhoH_temp);
  //   s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
  //   for (int n = 0; n < NUM_SPECIES; n++) {
  //     s_ext[FIRSTSPEC + n] = massfrac_ch[n] * s_ext[DENSITY];
  //   }

  //   in_zrange = (x[2]>prob_parm.ztan - 2*dxi) and (x[2] < prob_parm.ztan);
  //   in_xaxis  = (x[1]<dxi) and (x[1]>-dxi);
  //   in_yaxis  = (x[0]<dxi) and (x[0]>dxi);
  //   if (in_zrange and (in_xaxis or in_yaxis)){ // if upper injection
  //     // set other data like mixture
  //     s_ext[TEMP] = prob_parm.T_mean;
  //     eos.PYT2R(p_cgs, Y_rad, s_ext[TEMP], rho_cgs);
  //     s_ext[DENSITY] = rho_cgs * 1.0e3;
  //     eos.TY2H(s_ext[TEMP], Y_rad, RhoH_temp);
  //     s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion
  //     for (int n = 0; n < NUM_SPECIES; n++) {
  //       s_ext[FIRSTSPEC + n] = Y_rad[n] * s_ext[DENSITY];
  //     }
  //     theta = (sgn+idir)*0.5*Pi + 0.25*Pi;
  //     s_ext[VELX] = vmax_l*cos(theta);
  //     s_ext[VELY] = vmax_l*cos(theta);
  //     s_ext[VELZ] = 0.0;
  //   }
  //   else{
  //     s_ext[VELX] = 0.0;
  //     s_ext[VELY] = 0.0;
  //     s_ext[VELZ] = 0.0;
  //   }

  // }

  // if (idir == 2 and sgn == -1) {
  //     for (int n = 0; n < NUM_SPECIES; n++) {
  //       massfrac[n] = Y_main[n] ;
  //     }
  //     vmax_l = prob_parm.vmax;
  //     tanvmax_l = prob_parm.tanvmax;
  //     R_l = prob_parm.R;
  //     rtrans_l = prob_parm.rtrans;

  //     amrex::Real rho_cgs, P_cgs, RhoH_temp, radius, angle;
  //     P_cgs = prob_parm.P_mean * 10.0;

  //     s_ext[TEMP] = (1.0 ) * prob_parm.T_mean;

  //     eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
  //     s_ext[DENSITY] = rho_cgs * 1.0e3;

  //     eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
  //     s_ext[RHOH] =
  //       RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

  //     for (int n = 0; n < NUM_SPECIES; n++) {
  //       s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
  //     }

  //     radius = sqrt(x[0]*x[0]+x[1]*x[1]);
  //     angle = atan2(x[1],x[0]);
  //     // Velocity
  //     //
  //     s_ext[VELZ] = vmax_l*(1.0 - (radius/R_l)*(radius/R_l));
  //     s_ext[VELX] = 0.0;
  //     s_ext[VELY] = 0.0;
  //     if (tanvmax_l > 0) {

  //       if (radius < rtrans_l) {
  //         tvelocity = tanvmax_l*radius/rtrans_l;
  //       }
  //       else {
  //         tvelocity = tanvmax_l*rtrans_l/radius*(R_l-radius)/(R_l-rtrans_l);
  //       }
  //     }

  //     s_ext[VELX] = -tvelocity*sin(angle);
  //     s_ext[VELY] = tvelocity*cos(angle);

  //     //std::cout << "vx->" << s_ext[VELX] << "\n";
  //     //std::cout << "vy->" << s_ext[VELY] << "\n";
  //     //std::cout << "vz->" << s_ext[VELZ] << "\n";
  //     //std::cout << "\n";

  // }
  // else if ( idir == 1 ) {

  //   for (int n = 0; n < NUM_SPECIES; n++) {
  //     massfrac[n] = Y_rad[n];
      
  //   }

  //   amrex::Real rho_cgs, P_cgs, RhoH_temp;
  //   P_cgs = prob_parm.P_mean * 10.0;

  //   s_ext[TEMP] = prob_parm.T_rad;

  //   eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
  //   s_ext[DENSITY] = rho_cgs * 1.0e3;

  //   eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
  //   s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

  //   for (int n = 0; n < NUM_SPECIES; n++) {
  //     s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
  //   }

  //   // Velocity
  //   s_ext[VELZ] = 0.0;
  //   s_ext[VELY] = 0.0;
  //   s_ext[VELX] = sgn * prob_parm.V_rad;
    
  // }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& beta,
  amrex::GeometryData const& geomdata,
  amrex::Box const& domainBox,
  const int dir,
  const int beta_comp,
  const int nComp)
{
   amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
  // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
  // otherwise this routine could be called for other face diffusivity (Temp,
  // velocity, ...)
 // if (beta_comp == 0 && nComp == NUM_SPECIES) {
 //    if (dir == 2 && k <= domainBox.smallEnd(dir)) {
 //
 //      // Just over whole inlet
 //      for (int n = 0; n < nComp; n++) {
 //        beta(i, j, k, n) = 0.0;
 //      }
 //    }
 //  }
}

#endif
