#include "rbp.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>

namespace rbp {

template<typename T>
std::vector<arma::Mat<T>> random_effect_cpp(
    const std::vector<arma::umat>& Z,
    const std::vector<ObjectK<T>>& K,
    const arma::Col<T>& scale,
    const arma::Mat<T>& Py,
    const arma::Mat<T>& P, 
    int ncpus,
    bool get_pev
) {
    arma::uword nk = Z.size();
    arma::uword n = Py.n_rows;
    arma::uword np = Py.n_cols;

    if (np > 1 && get_pev) {
        std::cout << "PEV can only be calculated for single trait" << std::endl;
        get_pev = false;
    }

    std::vector<arma::Mat<T>> revl(nk);
    for (arma::uword i = 0; i < nk; ++i) {
            
        arma::Mat<T> revli;
        arma::uword start = i * np;
        arma::uword end = ((start + np) <= scale.n_elem) ? (start + np - 1) : (scale.n_elem - 1);
        const arma::umat& Zi = Z[i];
        const ObjectK<T>& Ki = K[i];
        const arma::Row<T> g_row = scale.subvec(start, end).t();

        if (Zi.n_elem == 1 && Ki.type == KType::IDENTITY) {
            revli = Py;
            revli.each_row() %= g_row;
            if (get_pev && !P.is_empty()) {
                T gi = g_row[0];
                arma::Col<T> pev = gi - P.diag() * (gi * gi);
                arma::Col<T> r2 = 1.0 - pev / gi;
                revli = arma::join_rows(revli, pev, r2);
            } 
            revl[i] = revli;

        } else if (Zi.n_elem == 1) {
            revli = block_Kx(Ki, arma::Mat<T>(Py.each_row() % g_row), ncpus);
            if (get_pev && !P.is_empty()) {
                T gi = g_row[0];
                arma::Mat<T> K_mat = (Ki.type == KType::KINSHIP) ? Ki.kinship.to_arma_mat() : Ki.compress.to_arma_mat();
                arma::Col<T> term1 = (Ki.type == KType::KINSHIP) ? arma::Col<T>(K_mat.diag() * gi) : arma::Col<T>(arma::sum(arma::square(K_mat), 1) * gi);
                arma::Col<T> term2;
                if (Ki.type == KType::KINSHIP) {
                    term2 = arma::sum((K_mat % (P * K_mat)), 0).t() * (gi * gi);
                }else {
                    arma::Mat<T> KcPKc = K_mat.t() * P * K_mat;
                    term2 = arma::sum(K_mat % (K_mat * KcPKc), 1) * (gi * gi);
                } 
                arma::Col<T> pev = term1 - term2;
                arma::Col<T> r2 = 1.0 - pev / gi;
                revli = arma::join_rows(revli, pev, r2);
            } 
            revl[i] = revli;

        } else if (Ki.type == KType::IDENTITY) {
            arma::uword ncz = arma::max(Zi.col(1)) + 1; 
            arma::Mat<T> omegai(ncz, np, arma::fill::zeros); 
            T gi = g_row[0];
            arma::Col<T> ZPZ_diag(ncz, arma::fill::zeros);
            std::vector<std::vector<arma::uword>> col_groups(ncz);
            for (arma::uword j = 0; j < Zi.n_rows; ++j) {
                col_groups[Zi(j, 1)].push_back(Zi(j, 0));
            }
            #pragma omp parallel for num_threads(ncpus) schedule(static)
            for (size_t ii = 0; ii < col_groups.size(); ++ii) { 
                const auto& group_i = col_groups[ii]; 
                if (!group_i.empty()) { 
                    arma::uvec P_rows = arma::conv_to<arma::uvec>::from(group_i);
                    omegai.row(ii) = arma::sum(Py.rows(P_rows), 0);
                    if (get_pev && !P.is_empty()) ZPZ_diag[ii] = arma::sum(arma::sum(P.submat(P_rows, P_rows), 0)); 
                }
            }
            omegai.each_row() %= g_row; 
            if (get_pev && !P.is_empty()) {
                arma::Col<T> pev = gi - ZPZ_diag * (gi * gi);
                arma::Col<T> r2 = 1.0 - pev / gi;
                revli = arma::join_rows(omegai, pev, r2);
            } else {
                revli = omegai;
            }
            revl[i] = revli;
            
        } else {
            arma::uword ncz = (Ki.type == KType::KINSHIP) ? Ki.kinship.nrow() : Ki.compress.nrow();
            arma::Mat<T> omegai(ncz, np, arma::fill::zeros); 
            std::vector<std::vector<arma::uword>> col_groups(ncz);
            for (arma::uword j = 0; j < Zi.n_rows; ++j) {
                omegai.row(Zi(j, 1)) += Py.row(Zi(j, 0));
                col_groups[Zi(j, 1)].push_back(Zi(j, 0));
            }
            revli = block_Kx(Ki, arma::Mat<T>(omegai.each_row() % g_row), ncpus); 
            
            if (get_pev && !P.is_empty()) {
                T gi = g_row[0];
                arma::Mat<T> K_mat = (Ki.type == KType::KINSHIP) ? Ki.kinship.to_arma_mat() : Ki.compress.to_arma_mat();
                arma::Col<T> term1 = (Ki.type == KType::KINSHIP) ? arma::Col<T>(K_mat.diag() * gi) : arma::Col<T>(arma::sum(arma::square(K_mat), 1) * gi);
                arma::Mat<T> zPz(ncz, ncz, arma::fill::zeros);
                #pragma omp parallel for num_threads(ncpus) schedule(guided)
                for (size_t ii = 0; ii < col_groups.size(); ++ii) { 
                    if (!col_groups[ii].empty()) { 
                        arma::uvec P_rows = arma::conv_to<arma::uvec>::from(col_groups[ii]);
                        zPz(ii, ii) = arma::sum(arma::sum(P.submat(P_rows, P_rows), 0)); 
                        for (size_t j = ii+1; j < col_groups.size(); ++j) {
                            if (!col_groups[j].empty()) { 
                                arma::uvec P_cols = arma::conv_to<arma::uvec>::from(col_groups[j]);
                                zPz(j, ii) = arma::sum(arma::sum(P.submat(P_cols, P_rows), 0)); 
                                zPz(ii, j) = zPz(j, ii);
                            }
                        }
                    }
                }
                arma::Col<T> term2;
                if (Ki.type == KType::KINSHIP) {
                    term2 = arma::sum((K_mat % (zPz * K_mat)), 0).t() * (gi * gi);
                }else {
                    arma::Mat<T> KcPKc = K_mat.t() * zPz * K_mat;
                    term2 = arma::sum(K_mat % (K_mat * KcPKc), 1) * (gi * gi);
                } 
                arma::Col<T> pev = term1 - term2;
                arma::Col<T> r2 = 1.0 - pev / gi;
                revli = arma::join_rows(revli, pev, r2);
            } 
            revl[i] = revli;
        }
    }

    return revl;
}

template std::vector<arma::Mat<double>> random_effect_cpp<double>(const std::vector<arma::umat>&, const std::vector<ObjectK<double>>&, const arma::Col<double>&, const arma::Mat<double>&, const arma::Mat<double>&, int, bool);
template std::vector<arma::Mat<float>> random_effect_cpp<float>(const std::vector<arma::umat>&, const std::vector<ObjectK<float>>&, const arma::Col<float>&, const arma::Mat<float>&, const arma::Mat<float>&, int, bool);

} // namespace rbp