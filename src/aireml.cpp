#include "rbp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <memory>

namespace rbp {

template<typename T>
arma::Col<T> MoM_RHE_cpp(
  arma::Col<T> y,
  const arma::Mat<T>& x,
  const std::vector<arma::umat>& Z, 
  const std::vector<ObjectK<T>>& K, 
  const arma::Mat<T>& RanVec, 
  int ncpus
) {
  arma::Mat<T> xxix = sym_safe_inverse(arma::Mat<T>(x.t() * x)) * x.t();
  y -= x * (xxix * y); 
  
  arma::uword n = y.n_rows;
  arma::uword nk = Z.size();
  
  arma::Mat<T> trGG(nk, nk, arma::fill::zeros);
  arma::Col<T> yGy(nk, arma::fill::zeros);
    
  arma::Mat<T> S_RanVec = RanVec - x * (xxix * RanVec);
  arma::uword ncu = RanVec.n_cols;
  std::vector<arma::Mat<T>> Gu(nk);
     
  for (arma::uword i = 0; i < nk; ++i) {
    const arma::umat& Zi = Z[i];
    const ObjectK<T>& Ki = K[i];
        
    if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
      yGy(i) = arma::as_scalar(y.t() * y);
      Gu[i] = S_RanVec;
    } else if (Zi.n_elem == 1) {
      yGy(i) = arma::as_scalar(y.t() * block_Kx(Ki, y, ncpus));
      Gu[i] = block_Kx(Ki, S_RanVec, ncpus);
    } else if (Ki.type == KType::IDENTITY) {
      arma::uword ncol_z = arma::max(Zi.col(1)) + 1;
      arma::Mat<T> Zu(ncol_z, ncu, arma::fill::zeros);
      arma::Mat<T> Zy(ncol_z, 1, arma::fill::zeros); 
      for (arma::uword j = 0; j < Zi.n_rows; ++j) { 
        Zy.row(Zi(j, 1)) += y.row(Zi(j, 0));
        Zu.row(Zi(j, 1)) += S_RanVec.row(Zi(j, 0));
      }
      yGy(i) = arma::as_scalar(Zy.t() * Zy);
      arma::Mat<T> Gui(n, ncu, arma::fill::zeros);
      Gui.rows(Zi.col(0)) = Zu.rows(Zi.col(1));
      Gu[i] = Gui;
    } else {
      arma::uword ncol_z = (Ki.type == KType::KINSHIP) ? Ki.kinship.nrow() : Ki.compress.nrow();
      arma::Mat<T> Zu(ncol_z, ncu, arma::fill::zeros);
      arma::Mat<T> Zy(ncol_z, 1, arma::fill::zeros); 
      for (arma::uword j = 0; j < Zi.n_rows; ++j) { 
        Zy.row(Zi(j, 1)) += y.row(Zi(j, 0));
        Zu.row(Zi(j, 1)) += S_RanVec.row(Zi(j, 0));
      }
      yGy(i) = arma::as_scalar(Zy.t() * block_Kx(Ki, Zy, ncpus));
      arma::Mat<T> KZu = block_Kx(Ki, Zu, ncpus);
      arma::Mat<T> Gui(n, ncu, arma::fill::zeros);
      Gui.rows(Zi.col(0)) = KZu.rows(Zi.col(1));
      Gu[i] = Gui;
    }
  }
    
  for (arma::uword row = 0; row < nk; ++row) {
    for (arma::uword col = row; col < nk; ++col) {
      arma::Mat<T>& ki = Gu[row];
      arma::Mat<T>& kj = Gu[col];
      trGG(row, col) = arma::sum(arma::sum(ki % (kj - x * (xxix * kj)), 0)) / ncu;
      if (row != col) trGG(col, row) = trGG(row, col);
    }
  }
   
  arma::Col<T> VCs;
  if (!arma::solve(VCs, trGG, yGy)) {
    VCs = arma::pinv(trGG) * yGy;
    std::cerr << "Warning: traceMatrix is singular in MoM_RHE, using pseudo-inverse.\n";
  }
  return VCs;
}

template<typename T>
AIResult<T> ai_operator_cpp(
  const std::vector<arma::umat>& Z, 
  const std::vector<ObjectK<T>>& K, 
  const arma::Col<T>& scale,  
  const arma::Mat<T>& X, 
  const arma::Col<T>& y,
  const arma::Mat<T>& RanVec,
  const arma::Mat<T>& cg_initial_0,
  const arma::Mat<T>& cg_initial_1,
  const std::string& solver,
  arma::Mat<T>& P,  
  arma::Mat<T>& R,  
  int ncpus, 
  double tol, 
  int maxit
) {
  arma::uword ncx = X.n_cols;
  arma::uword ncy = 1;
  arma::uword n = y.n_elem;
  arma::uword nk = Z.size();

  AIResult<T> result;
  
  if (solver == "Cholesky") {
    P.zeros();
    bool ok = compute_V(Z, K, scale, P, ncpus);
    if (!ok) throw std::runtime_error("Failed to calculate V matrix");
    if (!arma::chol(R, P)) {
      std::cout << "Cholesky failed, using ridge inversion insteaded" << std::endl; 
      P.diag() += 1e-4; 
      if (!arma::chol(R, P)) throw std::runtime_error("ridge inversion failed");
    }
    arma::inv(P, arma::trimatu(R));
    P = arma::symmatu(P * P.t()); 
        
    arma::Mat<T> VX = P * X; 
    arma::Mat<T> Vy = P * y;
    arma::Mat<T> XVX = X.t() * VX;
    arma::Mat<T> XVXi = sym_safe_inverse(XVX);
    arma::Col<T> XVy = VX.t() * y;
    arma::Col<T> Beta = XVXi * XVy;
    arma::Col<T> Varbeta = XVXi.diag();
    arma::Col<T> Py = Vy - VX * Beta;
     
    P -= VX * XVXi * VX.t(); 
    VX.reset(); Vy.reset(); XVX.reset(); XVXi.reset(); XVy.reset();
        
    arma::Mat<T> GPy(n, nk, arma::fill::zeros);
    arma::Mat<T> PGPy(n, nk, arma::fill::zeros);
    arma::Col<T> trPG(nk, arma::fill::zeros);
    arma::Col<T> yPGPy(nk, arma::fill::zeros);

    for (arma::uword i = 0; i < nk; ++i) {
      const arma::umat& Zi = Z[i];
      const ObjectK<T>& Ki = K[i];
      if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
        GPy.col(i) = Py;
        PGPy.col(i) = P * Py; 
        trPG(i) = sum(P.diag());
        yPGPy(i) = arma::as_scalar(Py.t() * Py);
      } else if (Zi.n_elem == 1) {
        GPy.col(i) = Ki.kinship.to_arma_mat() * Py;
        PGPy.col(i) = P * GPy.col(i); 
        trPG(i) = arma::sum(arma::sum(P % Ki.kinship.to_arma_mat(), 0)); 
        yPGPy(i) = arma::as_scalar(Py.t() * GPy.col(i));
      } else if (Ki.type == KType::IDENTITY) {
        arma::Col<T> ZPy(arma::max(Zi.col(1))+1, arma::fill::zeros); 
        std::vector<std::vector<arma::uword>> col_groups(arma::max(Zi.col(1))+1);
        for (arma::uword ii = 0; ii < Zi.n_rows; ++ii) {
          ZPy.at(Zi(ii, 1)) += Py.at(Zi(ii, 0)); 
          col_groups[Zi(ii, 1)].push_back(Zi(ii, 0));
        }
        arma::Col<T> GPyi = GPy.col(i);
        GPyi.elem(Zi.col(0))= ZPy.elem(Zi.col(1));
        GPy.col(i) = GPyi;
        T trPZZ = 0.0;
        #pragma omp parallel for num_threads(ncpus) schedule(static) reduction(+:trPZZ)
        for (size_t k = 0; k < col_groups.size(); ++k) { 
          const auto& group = col_groups[k]; 
          if (!group.empty()) { 
            arma::uvec rows = arma::conv_to<arma::uvec>::from(group);
            trPZZ += arma::sum(arma::sum(P.submat(rows, rows), 0));
          }
        }
        PGPy.col(i) = P * GPy.col(i); 
        trPG.at(i) = trPZZ;
        yPGPy.at(i) = arma::as_scalar(Py.t() * GPy.col(i));
      } else {
        arma::uvec z0 = Zi.col(0);
        arma::uvec z1 = Zi.col(1); 
        arma::Mat<T>& Gi = R;
        Gi.submat(z0, z0) = Ki.kinship.to_arma_mat().submat(z1, z1);
        GPy.col(i) = Gi * Py;
        PGPy.col(i) = P * GPy.col(i); 
        trPG(i) = arma::sum(arma::sum(P % Gi, 0));
        yPGPy(i) = arma::as_scalar(Py.t() * GPy.col(i));
      }
    }

    arma::Mat<T> AI(nk, nk, arma::fill::zeros);
    arma::Col<T> RHS = 0.5 * (trPG - yPGPy);

    for (arma::uword i = 0; i < nk; ++i) {
      for (arma::uword j = i; j < nk; ++j) {
        AI(i, j) = 0.5 * arma::as_scalar(GPy.col(i).t() * PGPy.col(j)); 
        if (i != j) AI(j, i) = AI(i, j);
      }
    }
        
    result.AI = AI;
    result.RHS = RHS;
    result.Py = Py;
    result.Beta = Beta;
    result.Varbeta = Varbeta;
    result.trPG = trPG;
    return result;
        
  } else if (solver == "CG") {
        
    arma::Mat<T> Vsol = cg_solve_cpp(Z, K, scale, arma::join_rows(y, X, RanVec), cg_initial_0, ncpus, tol, maxit, true);
        
    arma::uword ncu = RanVec.n_cols;
    arma::Mat<T> Py = Vsol.col(0);
    arma::Mat<T> VX = Vsol.cols(1, ncy + ncx - 1);
    arma::Mat<T> Pu = Vsol.cols(ncy + ncx, ncy + ncx + ncu - 1); 

    arma::Mat<T> XVX = X.t() * VX;
    arma::Mat<T> XVXi = sym_safe_inverse(XVX);
    arma::Col<T> XVy = VX.t() * y;
    arma::Col<T> Beta = XVXi * XVy;
    arma::Col<T> Varbeta = XVXi.diag();
    Py -= VX * Beta;
    Pu -= VX * XVXi * (VX.t() * RanVec); 
    XVX.reset(); XVy.reset();

    arma::Mat<T> GPy(n, nk, arma::fill::zeros);
    arma::Col<T> trPG(nk, arma::fill::zeros);

    for (arma::uword i = 0; i < nk; ++i) {
      const arma::umat& Zi = Z[i];
      const ObjectK<T>& Ki = K[i];
            
      if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
        GPy.col(i) = Py; 
        trPG(i) = arma::sum(arma::sum(RanVec % Pu, 0)) / ncu;
      } else if (Zi.n_elem == 1) {
        arma::Mat<T> KX = block_Kx(Ki, arma::Mat<T>(arma::join_rows(Py, RanVec)), ncpus);
        GPy.col(i) = KX.col(0);
        trPG(i) = arma::sum(arma::sum(KX.cols(1, ncy + ncu - 1) % Pu, 0)) / ncu;
      } else if (Ki.type == KType::IDENTITY) {
        arma::uword ncol_z = arma::max(Zi.col(1)) + 1;
        arma::Mat<T> Zu(ncol_z, ncu, arma::fill::zeros);
        arma::Col<T> ZPy(ncol_z, arma::fill::zeros); 
        for (arma::uword ii = 0; ii < Zi.n_rows; ++ii) { 
          ZPy(Zi(ii, 1)) += Py(Zi(ii, 0));
          Zu.row(Zi(ii, 1)) += RanVec.row(Zi(ii, 0));
        }
        arma::Col<T> GPyi = GPy.col(i);
        GPyi.elem(Zi.col(0))= ZPy.elem(Zi.col(1));
        GPy.col(i) = GPyi;
        arma::Mat<T> Gu (n, ncu, arma::fill::zeros);
        Gu.rows(Zi.col(0)) = Zu.rows(Zi.col(1));
        trPG(i) = arma::sum(arma::sum(Gu % Pu, 0)) / ncu;
      } else {
        arma::uword ncol_z = (Ki.type == KType::KINSHIP) ? Ki.kinship.nrow() : Ki.compress.nrow();
        arma::Mat<T> Zu(ncol_z, ncu, arma::fill::zeros);
        arma::Col<T> ZPy(ncol_z, arma::fill::zeros); 
        for (arma::uword ii = 0; ii < Zi.n_rows; ++ii) { 
          ZPy(Zi(ii, 1)) += Py(Zi(ii, 0));
          Zu.row(Zi(ii, 1)) += RanVec.row(Zi(ii, 0));
        }
        arma::Mat<T> Ky = block_Kx(Ki, ZPy, ncpus);
        arma::Mat<T> Ku = block_Kx(Ki, Zu, ncpus);
        arma::Col<T> GPyi = GPy.col(i);
        GPyi.elem(Zi.col(0))= Ky.elem(Zi.col(1));
        GPy.col(i) = GPyi;
        arma::Mat<T> Gu(n, ncu, arma::fill::zeros);
        Gu.rows(Zi.col(0)) = Ku.rows(Zi.col(1));
        trPG(i) = arma::sum(arma::sum(Gu % Pu, 0)) / ncu;
      }
    }
        
    arma::Col<T> yPGPy = (Py.t() * GPy).t();
    arma::Mat<T> PGPy = cg_solve_cpp(Z, K, scale, GPy, cg_initial_1, ncpus, tol, maxit, true);
    result.VGPy = PGPy;
    PGPy -= VX * XVXi * (VX.t() * GPy);

    arma::Mat<T> AI(nk, nk, arma::fill::zeros);
    arma::Col<T> RHS = 0.5 * (trPG - yPGPy);

    for (arma::uword i = 0; i < nk; ++i) {
      for (arma::uword j = i; j < nk; ++j) {
        AI(i, j) = 0.5 * arma::as_scalar(GPy.col(i).t() * PGPy.col(j)); 
        if (i != j) AI(j, i) = AI(i, j);
      }
    }

    result.AI = AI;
    result.RHS = RHS;
    result.Py = Py;
    result.Beta = Beta;
    result.Varbeta = Varbeta;
    result.trPG = trPG;
    result.VyXu = Vsol;
    result.VX = VX;
    result.XVXi = XVXi;
    result.Pu = Pu;

    return result;
  } else {
    throw std::runtime_error("Unknown 'solver' type");
  }
}

template<typename T>
single_result single_model_cpp(
  const arma::Col<T>& y,
  const arma::Mat<T>& X,
  const std::vector<arma::umat>& Z, 
  const std::vector<ObjectK<T>>& K,
  const std::string& solver,
  const std::string& test_method, 
  const int criterion, 
  const std::string out_file,
  const arma::uword num_RanVec,
  const std::string& bed_file,
  const arma::uvec& ind_row,
  const arma::uvec& ind_col,
  const arma::umat& Z_geno, 
  bool is_chunk,
  arma::Col<T> pars0,
  bool init_by_mom, 
  int random_seed,
  int re_gamma_size, 
  bool re_estimate_gamma,
  int maxit_par,
  int maxit_cg,
  double tol_iter,
  double tol_cg,
  int ncpus, 
  bool is_repeat,
  bool get_PEV,
  bool do_test 
) {
  const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
  std::cout << "=====================[ AIREML Running ]=====================\n";
  
  std::string current_solver = solver;
  if (is_repeat) current_solver = "CG";

  bool get_P = false;
  if (get_PEV) get_P = true;
  if (do_test && test_method == "standard" && !is_repeat) get_P = true;

  arma::uword n = y.n_elem;
  arma::arma_rng::set_seed(random_seed);
  //arma::Mat<T> RanVec = arma::randn<arma::Mat<T>>(n, num_RanVec);
  arma::Mat<T> RanVec(n, num_RanVec, arma::fill::none);
  for (arma::uword i = 0; i < num_RanVec; ++i) {
    RanVec.col(i) = arma::randn<arma::Col<T>>(n);
  }
  
  if (init_by_mom) {
    std::cout << "Initializing variance components via MoM-RHE...\n";
    pars0 = MoM_RHE_cpp(y, X, Z, K, RanVec, ncpus);
    std::cout << "MoM-RHE estimated thetas: ";
    if (pars0.n_elem <= 5) {
      for (arma::uword i = 0; i < pars0.n_elem; ++i) {
        std::cout << std::fixed << std::setprecision(4) << pars0[i];
        if (i < pars0.n_elem - 1) std::cout << ", ";
      }
    } else {
      for (arma::uword i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(4) << pars0[i] << ", ";
      }
      std::cout << "... "  << std::fixed << std::setprecision(4) << pars0[pars0.n_elem - 1] ;
    }
    std::cout << std::endl;
  }
  if (pars0.is_empty()) {
    pars0 =  arma::Col<T>(Z.size(), arma::fill::value(arma::var(y) / static_cast<T>(Z.size())));
  }

  arma::Col<T> pars = pars0;
  arma::Mat<T> initial_sol_0(1, 1, arma::fill::zeros);
  arma::Mat<T> initial_sol_1(1, 1, arma::fill::zeros);
  arma::Col<T> se, Beta, Varbeta, Py, trPG;
  arma::Mat<T> VX, XVXi, Pu, P, R;
  std::unique_ptr<MappedMatrix<T>> mapp_P_ptr;
  std::unique_ptr<MappedMatrix<T>> mapp_R_ptr;

  if (current_solver == "Cholesky") {
    std::string P_file = out_file + ".P.bin";
    std::string R_file = out_file + ".R.bin";
    mapp_P_ptr = std::make_unique<MappedMatrix<T>>(P_file, n, n, true, false, true);
    mapp_R_ptr = std::make_unique<MappedMatrix<T>>(R_file, n, n, true, false, true);
    P = mapp_P_ptr->to_arma_mat();
    R = mapp_R_ptr->to_arma_mat();
  }

  int it = 0;
  bool converged = false;

  while (it < maxit_par) {
    const std::chrono::steady_clock::time_point start_time_i = std::chrono::steady_clock::now();
    std::cout << ">>>-------------------[ Iteration " << it+1 << " ]-------------------<<<\n";
    AIResult<T> AIstep = ai_operator_cpp(
      Z, K, pars, X, y, RanVec, initial_sol_0, 
      initial_sol_1, current_solver, P, R, ncpus, tol_cg, maxit_cg
    );

    arma::Mat<T> AI = AIstep.AI;
    arma::Col<T> RHS = AIstep.RHS;
    Py = AIstep.Py;
    Beta = AIstep.Beta;
    Varbeta = AIstep.Varbeta;
    trPG = AIstep.trPG;

    if (current_solver == "CG") {
      initial_sol_0 = AIstep.VyXu;
      initial_sol_1 = AIstep.VGPy;
      VX = AIstep.VX;
      XVXi = AIstep.XVXi;
      Pu = AIstep.Pu;
    }

    arma::Mat<T> covM;
    if (!arma::inv(covM, AI)) {
      std::cout << "Generated singular AI matrix, using pseudo-inverse\n";
      covM = arma::pinv(AI, static_cast<T>(1e-6));
    } 
    se = arma::sqrt(covM.diag());
    arma::Col<T> pars_new = pars - covM * RHS;
    T norm_grad = arma::norm(RHS);
    T rel_error = arma::norm(pars_new - pars) / arma::norm(pars_new);
        
    auto now_i = std::chrono::steady_clock::now();
    auto time_elapsed_i = std::chrono::duration_cast<std::chrono::milliseconds>(now_i - start_time_i).count();
    std::cout << "gradient: " << std::fixed << std::setprecision(7) << norm_grad 
              << " | relERR: " << std::fixed << std::setprecision(7) << rel_error 
              << " | IterTime: " << string_time(double(time_elapsed_i) / 1000.0 ) << std::endl;
    std::cout << "updated theta: ";
    if (pars_new.n_elem <= 5) {
      for (arma::uword i = 0; i < pars_new.n_elem; ++i) {
        std::cout << std::fixed << std::setprecision(4) << pars_new(i);
        if (i < pars_new.n_elem - 1) std::cout << ", ";
      }
    } else {
      for (arma::uword i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(4) << pars_new(i) << ", ";
      }
      std::cout << "... " << std::fixed << std::setprecision(4) << pars_new(pars_new.n_elem - 1);
    }
    std::cout << std::endl;
        
    bool has_issue = false;
    for (arma::uword vi = 0; vi < pars_new.n_elem; vi++) {
      T v = pars_new(vi);
      if (!std::isfinite(v) || v < 0.0) {
        pars_new(vi) = static_cast<T>(tol_iter);
        has_issue = true;
      }
    }
    if (has_issue) {
      std::cout << "Iteration stop as generating invalid thetas, which were corrected to tolerence!" << std::endl;
      converged = false;
      break;
    }

    if (maxit_par == 1) {
      pars_new = pars0;
      std::cout << "****Iterated only once, using initial values instead of updating values!****" << std::endl;
      break;
    } else {
      pars = pars_new; 
    }
        
    if ((criterion == 0 && (rel_error <= static_cast<T>(tol_iter) || norm_grad <= static_cast<T>(tol_iter))) ||
        (criterion == 1 && rel_error <= static_cast<T>(tol_iter)) ||
        (criterion == 2 && norm_grad <= static_cast<T>(tol_iter))) {
      converged = true;
      break;
    }
    ++it;
  }
  auto now = std::chrono::steady_clock::now();
  auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
  if (maxit_par != 1) {
    if (converged) {
      std::cout << "[ ^_^ Converged successfully in " << string_time(double(time_elapsed) / 1000.0 )  <<" ]" << std::endl;
    } else {
      std::cout << "[ -_-! Failed to converge, used time " << string_time(double(time_elapsed) / 1000.0 )  <<" ]" << std::endl;
    }
  }

  bool cal_P = false;
  if (get_P && current_solver == "CG") {
    try {
      std::chrono::steady_clock::time_point pt1 = std::chrono::steady_clock::now();
      std::cout << "Generating projection matrix..." << std::flush;
      std::string P_file = out_file + ".P.bin";
      MappedMatrix<T> mapp_P(P_file, n, n, true, false, true);
      std::string R_file = out_file + ".R.bin";
      MappedMatrix<T> mapp_R(R_file, n, n, true, false, true);
      P = mapp_P.to_arma_mat();
      R = mapp_R.to_arma_mat();
      P.zeros();
      bool ok = compute_V(Z, K, pars, P, ncpus);
      if (!ok) throw std::runtime_error("Failed to calculate V matrix");
              
      if (arma::chol(R, P)) {
        arma::inv(P, arma::trimatu(R));
        P = arma::symmatu(P * P.t()); 
      } else {
        std::cout << "Cholesky failed, using Lanczos general inverse..." << std::endl;
        P.diag() += 1e-4; // ridge V
        if (arma::chol(R, P)) {
          arma::inv(P, arma::trimatu(R));
          P = arma::symmatu(P * P.t()); //ridge V^-1
          bool success = false;
          arma::vec val;
          arma::mat vec;
          for (int i = 0; i < 100; ++i) {
            int m = 5 * (i + 1);
            if (m > static_cast<int>(n / 10)) break;
            LanczosResult res = lanczos_eigen_mat(P, m, -1, tol_cg, ncpus);
            val = res.eigenvalues;
            vec = res.eigenvectors;
            arma::uvec index = arma::find(val >= (1.0 / (2.0 * 1e-6 + 1e-4)));
            if (index.n_elem < val.n_elem) {
              val = val.elem(index);
              vec = vec.cols(index);
              success = true;
              break;
            }
          }
          if (!success) {
            std::cout << "Lanczos failed, using ridge inverse..." << std::endl;
          } else {
            arma::Mat<T> penalty = arma::conv_to<arma::Mat<T>>::from(vec * arma::diagmat(val) * vec.t());
            P -= penalty;
          }
        } else {
          throw std::runtime_error("Cholesky for Lanczos also failed, this should not happen");
        }
      }
      P -= VX * XVXi * VX.t(); 
      std::chrono::steady_clock::time_point pt2 = std::chrono::steady_clock::now();
      auto time_elapsed2 = std::chrono::duration_cast<std::chrono::milliseconds>(pt2 - pt1).count();
      std::cout << "P matrix has been calculated in " << string_time(double(time_elapsed2) / 1000.0) << std::endl;
      cal_P = true;
    } catch (const std::exception& e) {
      std::cerr << "[Error] Failed to generate projection matrix: " << e.what() << std::endl;
    }
  }

  std::cout << "Calculating random effects";
  if (get_PEV) std::cout << " and PEV";
  std::cout << "..." << std::endl;
  std::vector<arma::Mat<T>> rand_T = random_effect_cpp(Z, K, pars, Py, P, ncpus, get_PEV);

  arma::mat gwas_result;
  if (do_test) {
    std::cout << "=====================[ GWAS Scanning ]=====================\n";
    std::string tm = test_method;
    if (is_repeat && tm == "standard") tm = "repeat";
    
    T gamma = -1.0;
    if (K[0].type != KType::KINSHIP) re_estimate_gamma = true;
    if (!re_estimate_gamma && trPG.n_elem > 0) {
      gamma = trPG(0) / (ind_row.n_elem - 1.0);
    }
    if (Pu.is_empty() && !P.is_empty() && !RanVec.is_empty()) {
      Pu = P * RanVec;
    }

    gwas_result = gwas_scan_cpp(
      bed_file, ind_row, ind_col, is_chunk, tm,
      Py, P, Z_geno, Pu, RanVec, X, Z, K,
      pars, VX, XVXi, re_gamma_size, random_seed, gamma,
      8, tol_cg, maxit_cg, ncpus
    );
  }
  
  single_result res;
  res.vc = arma::conv_to<arma::vec>::from(pars);
  res.se_vc = arma::conv_to<arma::vec>::from(se);
  res.beta = arma::conv_to<arma::vec>::from(Beta);
  res.se_beta = arma::conv_to<arma::vec>::from(arma::sqrt(Varbeta)); 
  
  for (const auto& r : rand_T) {
    res.rand_eff.push_back(arma::conv_to<arma::mat>::from(r));
  }
  if (do_test) {
    res.gwas_res = gwas_result; 
  }

  if (!R.is_empty()) {
    std::string fileR = out_file + ".R.bin";
    if (std::remove(fileR.c_str()) != 0) std::cout << "Failed to remove file: " << fileR << std::endl;
  }
  if (!P.is_empty() && (!get_P || !cal_P)) {
    std::string fileP = out_file + ".P.bin";
    if (std::remove(fileP.c_str()) != 0) std::cout << "Failed to remove file: " << fileP << std::endl;
  }

  return res;
}

template arma::Col<double> MoM_RHE_cpp<double>(arma::Col<double>, const arma::Mat<double>&, const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Mat<double>&, int);
template arma::Col<float> MoM_RHE_cpp<float>(arma::Col<float>, const arma::Mat<float>&, const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Mat<float>&, int);

template AIResult<double> ai_operator_cpp<double>(const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, const arma::Mat<double>&, const arma::Col<double>&, const arma::Mat<double>&, const arma::Mat<double>&, const arma::Mat<double>&, const std::string&, arma::Mat<double>&, arma::Mat<double>&, int, double, int);
template AIResult<float> ai_operator_cpp<float>(const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, const arma::Mat<float>&, const arma::Col<float>&, const arma::Mat<float>&, const arma::Mat<float>&, const arma::Mat<float>&, const std::string&, arma::Mat<float>&, arma::Mat<float>&, int, double, int);

template single_result single_model_cpp<double>(const arma::Col<double>&, const arma::Mat<double>&, const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const std::string&, const std::string&, const int, const std::string, const arma::uword, const std::string&, const arma::uvec&, const arma::uvec&, const arma::umat&, bool, arma::Col<double>, bool, int, int, bool, int, int, double, double, int, bool, bool, bool);
template single_result single_model_cpp<float>(const arma::Col<float>&, const arma::Mat<float>&, const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const std::string&, const std::string&, const int, const std::string, const arma::uword, const std::string&, const arma::uvec&, const arma::uvec&, const arma::umat&, bool, arma::Col<float>, bool, int, int, bool, int, int, double, double, int, bool, bool, bool);

} // namespace rbp