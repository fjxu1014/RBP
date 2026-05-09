#include "rbp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <iomanip>
#include <unordered_set>

namespace rbp {

// 辅助函数：判断字符串是否为缺失值
bool is_na(const std::string& s) {
    if (s.empty()) return true;
    std::string lower_s = s;
    std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
    return (lower_s == "na" || lower_s == "nan" || lower_s == "null");
}

// 辅助函数：分割字符串 (支持逗号、制表符或空格)
std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : line) {
        if (c == ',' || c == '\t' || c == ' ') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) tokens.push_back(token);
    return tokens;
}

/**
 * 数据读取与模型矩阵构建 (支持多表型，完整保留元数据)
 */
template<typename T>
DataInput<T> load_data_cpp(
    const std::string& phe_file,
    const std::vector<int>& phe_pos,                
    const std::vector<int>& fix_pos,
    const std::vector<int>& qcovar_pos,
    const std::vector<int>& rand_pos,
    const std::vector<std::string>& rand_cov_files, 
    const std::vector<std::string>& grm_files,      
    const std::string& geno_file,                   
    const std::vector<size_t>& geno_row_selected,   
    bool is_chunk
) {
    DataInput<T> res;
    std::cout << "Loading and extracting data...\n";

    // ========================================================================
    // 1. 读取表型文件并解析表头与文本数据
    // ========================================================================
    std::ifstream file_in(phe_file);
    if (!file_in.is_open()) throw std::runtime_error("Cannot open phenotype file: " + phe_file);

    std::string line;
    std::getline(file_in, line); 
    std::vector<std::string> header = split_line(line); 
    
    std::vector<std::vector<std::string>> raw_data;
    while (std::getline(file_in, line)) {
        if (line.empty()) continue;
        raw_data.push_back(split_line(line));
    }
    file_in.close();

    for (int p : phe_pos) {
        std::string name = (p < header.size()) ? header[p] : "Phe_" + std::to_string(p);
        res.phe_names.push_back(name);
    }

    // ========================================================================
    // 2. 加载外部 ID 字典并保留 GRM 的文件名及原始 ID 列表
    // ========================================================================
    for (const std::string& full_path : grm_files) {
        size_t pos = full_path.find_last_of("/\\");
        std::string pure_name = (pos == std::string::npos) ? full_path : full_path.substr(pos + 1);
        res.grm_names.push_back(pure_name);
    }
    
    std::vector<std::unordered_map<std::string, size_t>> grm_id_maps(grm_files.size());
    res.grm_ids.resize(grm_files.size()); 

    for (size_t i = 0; i < grm_files.size(); ++i) {
        std::ifstream id_in(grm_files[i] + ".id");
        if (!id_in) throw std::runtime_error("Cannot open GRM ID file: " + grm_files[i] + ".id");
        std::string gid; size_t idx = 0;
        while (id_in >> gid) { 
            grm_id_maps[i][gid] = idx;
            idx++;
            res.grm_ids[i].push_back(gid); 
            std::getline(id_in, line);     
        }
    }

    std::unordered_map<std::string, size_t> geno_id_map;
    std::ifstream fam_in(geno_file + ".fam");
    if (!fam_in) throw std::runtime_error("Cannot open FAM file: " + geno_file + ".fam");
    
    std::vector<std::string> all_geno_ids;
    std::string fid, iid;
    while (fam_in >> fid >> iid) {
        all_geno_ids.push_back(iid);
        std::getline(fam_in, line);
    }
    fam_in.close();

    for (size_t subset_idx = 0; subset_idx < geno_row_selected.size(); ++subset_idx) {
        size_t orig_idx = geno_row_selected[subset_idx];
        if (orig_idx < all_geno_ids.size()) {
            geno_id_map[all_geno_ids[orig_idx]] = subset_idx;
        }
    }

    // ========================================================================
    // 3. 严格的缺失值 (NA) 过滤与 ID 交集计算 (Listwise Deletion)
    // ========================================================================
    std::vector<size_t> valid_rows;
    for (size_t i = 0; i < raw_data.size(); ++i) {
        const auto& row = raw_data[i];
        if (row.empty()) continue;
        std::string id = row[0]; 
        bool is_valid = true;
        for (int p : phe_pos) if (row.size() <= p || is_na(row[p])) { is_valid = false; break; }
        if (is_valid) {
            for (int p : fix_pos)    if (row.size() <= p || is_na(row[p])) { is_valid = false; break; }
            for (int p : qcovar_pos) if (row.size() <= p || is_na(row[p])) { is_valid = false; break; }
            for (int p : rand_pos)   if (row.size() <= p || is_na(row[p])) { is_valid = false; break; }
        }
        if (!is_valid) continue;
        for (const auto& gmap : grm_id_maps) if (gmap.find(id) == gmap.end()) { is_valid = false; break; }
        if (geno_id_map.find(id) == geno_id_map.end()) { is_valid = false; }
        if (is_valid) {
            valid_rows.push_back(i);
            res.valid_ids.push_back(id); 
        }
    }

    size_t n_valid = valid_rows.size();
    if (n_valid == 0) throw std::runtime_error("No valid individuals remain after NA and ID intersection filtering.");
    std::cout << n_valid << " effective individuals after filtering in phenotype file\n";

    // ========================================================================
    // 4. 构建 y 矩阵 和 模型矩阵 X，并生成对应列名
    // ========================================================================
    size_t num_phe = phe_pos.size();
    res.y.set_size(n_valid, num_phe);
    for (size_t i = 0; i < n_valid; ++i) {
        for (size_t j = 0; j < num_phe; ++j) {
            res.y(i, j) = static_cast<T>(std::stod(raw_data[valid_rows[i]][phe_pos[j]]));
        }
    }

    size_t x_cols = 1; 
    std::vector<std::unordered_map<std::string, int>> fix_levels(fix_pos.size());
    std::vector<std::vector<std::string>> fix_level_names(fix_pos.size()); 

    for (size_t f = 0; f < fix_pos.size(); ++f) {
        int p = fix_pos[f];
        int level_idx = 0;
        for (size_t i = 0; i < n_valid; ++i) {
            const std::string& val = raw_data[valid_rows[i]][p];
            if (fix_levels[f].find(val) == fix_levels[f].end()) {
                fix_levels[f][val] = level_idx;
                level_idx++;
                fix_level_names[f].push_back(val); 
            }
        }
        std::string level_to_drop = fix_level_names[f].back();
        fix_levels[f].erase(level_to_drop);
        fix_level_names[f].pop_back();
        if (fix_levels[f].size() == 0) {
            std::cout << "Notice: Fixed effect at column index " << p 
                      << " has only one level, it has been dropped from the model." << std::endl;
        }
        
        x_cols += fix_levels[f].size();
    }
    x_cols += qcovar_pos.size();

    res.X.set_size(n_valid, x_cols);
    res.X.fill(0.0);
    res.X.col(0).fill(1.0); 
    
    res.x_col_names.push_back("Intercept"); 
    for (size_t f = 0; f < fix_pos.size(); ++f) {
        std::string base_name = (fix_pos[f] < header.size()) ? header[fix_pos[f]] : "Fix" + std::to_string(f);
        if (!fix_level_names[f].empty()) {
            for (const auto& lvl : fix_level_names[f]) res.x_col_names.push_back(base_name + "_" + lvl); 
        }
    }
    for (size_t c = 0; c < qcovar_pos.size(); ++c) {
        std::string cov_name = (qcovar_pos[c] < header.size()) ? header[qcovar_pos[c]] : "Cov" + std::to_string(c);
        res.x_col_names.push_back(cov_name); 
    }

    for (size_t i = 0; i < n_valid; ++i) {
        const auto& row = raw_data[valid_rows[i]];
        size_t col_idx = 1;
        for (size_t f = 0; f < fix_pos.size(); ++f) {
            auto it = fix_levels[f].find(row[fix_pos[f]]);
            if (it != fix_levels[f].end()) {
                int level = it->second; 
                res.X(i, col_idx + level) = 1.0;
            }
            col_idx += fix_levels[f].size();
        }
        for (size_t c = 0; c < qcovar_pos.size(); ++c) {
            res.X(i, col_idx + c) = static_cast<T>(std::stod(row[qcovar_pos[c]]));
        }
    }

    // ========================================================================
    // 5. 构建 GRM 的 Z 矩阵和 K 矩阵对象
    // ========================================================================
    for (size_t g = 0; g < grm_files.size(); ++g) {
        const std::string& base_path = grm_files[g];
        std::string id_file = base_path + ".id";
        std::string col_file = base_path + ".col";
        std::string bin_file = base_path + ".bin";

        size_t N = rbp::get_file_lines(id_file);
        size_t M;
        
        std::ifstream file_check(col_file);
        bool is_compress = false;
        if (file_check.is_open()) {
            file_check.close();
            M = rbp::get_file_lines(col_file);
            if (M == 0) throw std::runtime_error("Error: The file '" + col_file + "' exists but is empty.");
            is_compress = true;
        } else {
            M = N;
        }

        ObjectK<T> obj_k;
        if (is_compress) {
            obj_k.type = KType::COMPRESS;
            obj_k.compress = MappedMatrix<T>(bin_file, N, M, false, is_chunk, true);
        } else {
            obj_k.type = KType::KINSHIP;
            obj_k.kinship = MappedMatrix<T>(bin_file, N, N, false, is_chunk, true);
        }
        res.K.push_back(std::move(obj_k));

        arma::umat Z_mat(n_valid, 2);
        for (size_t i = 0; i < n_valid; ++i) {
            Z_mat(i, 0) = i; 
            Z_mat(i, 1) = grm_id_maps[g][res.valid_ids[i]]; 
        }
        if (arma::all(Z_mat.col(0) == Z_mat.col(1))) {
            Z_mat = arma::umat(1, 1, arma::fill::ones); 
        }
        res.Z.push_back(Z_mat);
    }

    // ========================================================================
    // 6. 构建额外随机效应的 Z 和 K，并提取随机效应名称和 ID 集合
    // ========================================================================
    for (size_t r = 0; r < rand_pos.size(); ++r) {
        int p = rand_pos[r];
        std::string rand_name = (p < header.size()) ? header[p] : "Rand" + std::to_string(r);
        res.rand_names.push_back(rand_name);

        std::string cov_file = (r < rand_cov_files.size()) ? rand_cov_files[r] : "";
        arma::umat Z_mat(n_valid, 2);
        ObjectK<T> obj_k;
        std::vector<std::string> current_rand_ids; 

        if (cov_file.empty() || cov_file == "1") {
            std::unordered_map<std::string, size_t> rand_levels;
            size_t level_idx = 0;
            
            for (size_t i = 0; i < n_valid; ++i) {
                const std::string& val = raw_data[valid_rows[i]][p];
                if (rand_levels.find(val) == rand_levels.end()) {
                    rand_levels[val] = level_idx++;
                    current_rand_ids.push_back(val); // 记录动态提取的 ID
                }
                Z_mat(i, 0) = i;
                Z_mat(i, 1) = rand_levels[val];
            }
            obj_k.type = KType::IDENTITY;
            
        } else {
            std::string id_file = cov_file + ".id";
            std::string col_file = cov_file + ".col";
            std::string bin_file = cov_file + ".bin";

            std::unordered_map<std::string, size_t> rand_id_map;
            std::ifstream id_in(id_file);
            if (!id_in) throw std::runtime_error("Cannot open Random Effect ID file: " + id_file);
            
            std::string gid; size_t idx = 0; std::string tmp_line;
            while (id_in >> gid) { 
                rand_id_map[gid] = idx++;
                current_rand_ids.push_back(gid); // 记录文件的全集 ID
                std::getline(id_in, tmp_line); 
            }
            id_in.close();

            for (size_t i = 0; i < n_valid; ++i) {
                const std::string& val = raw_data[valid_rows[i]][p];
                auto it = rand_id_map.find(val);
                if (it == rand_id_map.end()) throw std::runtime_error("Error: Level '" + val + "' not in " + id_file);
                Z_mat(i, 0) = i;
                Z_mat(i, 1) = it->second;
            }

            size_t N = rand_id_map.size();
            size_t M;
            std::ifstream file_check(col_file);
            bool is_compress = false;
            if (file_check.is_open()) {
                file_check.close();
                M = rbp::get_file_lines(col_file);
                is_compress = true;
            } else {
                M = N;
            }

            if (is_compress) {
                obj_k.type = KType::COMPRESS;
                obj_k.compress = MappedMatrix<T>(bin_file, N, M, false, is_chunk, true);
            } else {
                obj_k.type = KType::KINSHIP;
                obj_k.kinship = MappedMatrix<T>(bin_file, N, N, false, is_chunk, true);
            }
        }

        if (arma::all(Z_mat.col(0) == Z_mat.col(1))) {
            Z_mat = arma::umat(1, 1, arma::fill::ones); 
        }

        res.Z.push_back(Z_mat);
        res.K.push_back(std::move(obj_k));
        res.rand_ids.push_back(current_rand_ids); // 保存以便写入 .rand
    }
    // ========================================================================
    // 7. 构建残差 (Residual) 的 Z 和 K 
    // ========================================================================
    // 追加名称和 ID 集合
    res.rand_names.push_back("Residual");
    res.rand_ids.push_back(res.valid_ids);
    
    // 残差的方差协方差矩阵为单位阵 (IDENTITY)
    ObjectK<T> obj_k_res;
    obj_k_res.type = KType::IDENTITY;
    res.K.push_back(std::move(obj_k_res));
    
    res.Z.push_back(arma::umat(1, 1, arma::fill::ones));

    res.Z_geno.set_size(n_valid, 2);
    for (size_t i = 0; i < n_valid; ++i) {
        res.Z_geno(i, 0) = i;
        res.Z_geno(i, 1) = geno_id_map[res.valid_ids[i]];
    }
    if (arma::all(res.Z_geno.col(0) == res.Z_geno.col(1))) res.Z_geno = arma::umat(1, 1, arma::fill::ones); 

    std::cout << "Data loading completed successfully.\n";
    return res;
}

/**
 * 将各类模型计算结果统一格式化写入本地磁盘文件
 */
template<typename T>
void write_out_single(
    const single_result& res,
    const DataInput<T>& data_info,
    const std::string& out_prefix,
    const std::string& bed_file,
    const arma::uvec& ind_col,
    bool get_PEV,
    bool do_test
) {
    std::cout << "Writing Results...\n";
    
    // -------------------------------------------------------------
    // 1. 写出 .var 文件 (方差组分)
    // -------------------------------------------------------------
    std::ofstream var_out(out_prefix + ".var");
    if (!var_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".var");
    
    var_out << "Item\tVariance\tSE\tProportion\n";
    double sum_vc = arma::sum(res.vc);
    if (sum_vc == 0.0) sum_vc = 1e-16; // 防除零保护

    size_t vc_idx = 0;
    for (const auto& gname : data_info.grm_names) {
        var_out << gname << "\t" << res.vc(vc_idx) << "\t" << res.se_vc(vc_idx) << "\t" << (res.vc(vc_idx) / sum_vc) << "\n";
        vc_idx++;
    }
    for (const auto& rname : data_info.rand_names) {
        var_out << rname << "\t" << res.vc(vc_idx) << "\t" << res.se_vc(vc_idx) << "\t" << (res.vc(vc_idx) / sum_vc) << "\n";
        vc_idx++;
    }
    //var_out << "Residual\t" << res.vc(vc_idx) << "\t" << res.se_vc(vc_idx) << "\t" << (res.vc(vc_idx) / sum_vc) << "\n";
    var_out.close();

    // -------------------------------------------------------------
    // 2. 写出 .beta 文件 (固定效应)
    // -------------------------------------------------------------
    std::ofstream beta_out(out_prefix + ".beta");
    if (!beta_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".beta");

    beta_out << "Level\tEffect\tSE\n";
    for (size_t i = 0; i < data_info.x_col_names.size(); ++i) {
        beta_out << data_info.x_col_names[i] << "\t" << res.beta(i) << "\t" << res.se_beta(i) << "\n";
    }
    beta_out.close();

    // -------------------------------------------------------------
    // 3. 写出 .rand 文件 (随机效应汇总，支持重复力模型的多残差输出)
    // -------------------------------------------------------------
    
    // 必定包含 Residual，且它位于 rand_names 数组的末尾
    size_t resid_effect_idx = data_info.grm_names.size() + data_info.rand_names.size() - 1;

    // 收集所有唯一的 ID (包含 GRM, 额外分类随机效应, 以及表型残差中的 ID)
    // 这样保证了任何存在于模型系统中的个体都不会被漏掉
    std::vector<std::string> ordered_unique_ids;
    std::unordered_set<std::string> seen_ids;    

    for (const auto& vec : data_info.grm_ids) {
        for (const auto& id : vec) {
            // seen_ids.insert(id).second 会返回一个布尔值：如果是新插入的，返回 true；如果已经存在，返回 false。
            if (seen_ids.insert(id).second) {
                ordered_unique_ids.push_back(id); // 只有新 ID 才追加到顺序列表中
            }
        }
    }
    for (const auto& vec : data_info.rand_ids) {
        for (const auto& id : vec) {
            if (seen_ids.insert(id).second) {
                ordered_unique_ids.push_back(id);
            }
        }
    }

    // 为每个 GRM 构建 O(1) 的快速字典映射
    std::vector<std::unordered_map<std::string, size_t>> grm_id_dicts(data_info.grm_names.size());
    for (size_t i = 0; i < data_info.grm_names.size(); ++i) {
        for (size_t j = 0; j < data_info.grm_ids[i].size(); ++j) {
            grm_id_dicts[i][data_info.grm_ids[i][j]] = j;
        }
    }
    
    // 为分类随机效应构建字典映射 (重点：扣除最后一个 Residual 项)
    size_t rand_dict_size = data_info.rand_names.size() - 1;
    std::vector<std::unordered_map<std::string, size_t>> rand_id_dicts(rand_dict_size);
    for (size_t i = 0; i < rand_dict_size; ++i) {
        for (size_t j = 0; j < data_info.rand_ids[i].size(); ++j) {
            rand_id_dicts[i][data_info.rand_ids[i][j]] = j;
        }
    }

    // 为 Residual 建立专属的“一对多”字典映射 (以适应重复力模型中的同一个体多次测定)
    std::unordered_map<std::string, std::vector<size_t>> resid_id_dict;
    const auto& resid_ids = data_info.rand_ids.back();
    for (size_t j = 0; j < resid_ids.size(); ++j) {
        resid_id_dict[resid_ids[j]].push_back(j); // 追加保存所有的残差行号
    }

    std::ofstream rand_out(out_prefix + ".rand");
    if (!rand_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".rand");

    // 写表头
    rand_out << "ID";
    for (const auto& gname : data_info.grm_names) {
        rand_out << "\t" << gname << "_Effect";
        if (get_PEV) rand_out << "\t" << gname << "_PEV\t" << gname << "_r2";
    }
    for (const auto& rname : data_info.rand_names) {
        rand_out << "\t" << rname << "_Effect";
        if (get_PEV) rand_out << "\t" << rname << "_PEV\t" << rname << "_r2";
    }
    rand_out << "\n";

    // 遍历全局去重后的所有 ID
    for (const auto& id : ordered_unique_ids) {
        // 尝试获取该 ID 对应的所有残差索引列表
        std::vector<size_t> res_indices;
        auto it_res = resid_id_dict.find(id);
        if (it_res != resid_id_dict.end()) {
            res_indices = it_res->second;
        }
        
        // 如果该个体有 N 条残差记录，打印 N 行。
        // 如果该个体仅在其他效应（如加性遗传）中有，而不在残差中（即 res_indices.size() == 0），
        // 这里的 std::max(1, 0) 会强行将其设为 1，保证该个体依然会被写回文件！
        size_t n_rows_to_print = std::max<size_t>(1, res_indices.size());

        for (size_t r = 0; r < n_rows_to_print; ++r) {
            rand_out << id;
            size_t matrix_idx = 0; 
            
            // 1. GRM 随机效应 (字典查找唯一值，如果该个体打印多行则重复写回同一育种值)
            for (size_t i = 0; i < data_info.grm_names.size(); ++i) {
                auto it = grm_id_dicts[i].find(id);
                if (it != grm_id_dicts[i].end()) {
                    size_t row_idx = it->second;
                    rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, 0);
                    if (get_PEV) {
                        rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, 1) 
                                 << "\t" << res.rand_eff[matrix_idx](row_idx, 2);
                    }
                } else {
                    rand_out << "\tNA";
                    if (get_PEV) rand_out << "\tNA\tNA";
                }
                matrix_idx++;
            }
            
            // 2. 额外分类随机效应 (如批次效应等，排除 Residual 项)
            for (size_t i = 0; i < rand_dict_size; ++i) {
                auto it = rand_id_dicts[i].find(id);
                if (it != rand_id_dicts[i].end()) {
                    size_t row_idx = it->second;
                    rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, 0);
                    if (get_PEV) {
                        rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, 1) 
                                 << "\t" << res.rand_eff[matrix_idx](row_idx, 2);
                    }
                } else {
                    rand_out << "\tNA";
                    if (get_PEV) rand_out << "\tNA\tNA";
                }
                matrix_idx++;
            }
            
            // 3. Residual 效应
            if (res_indices.empty()) {
                // 如果个体的确没有表型（不在残差字典中），在残差列填补 NA
                rand_out << "\tNA";
                if (get_PEV) rand_out << "\tNA\tNA";
            } else {
                // 存在表型残差，精确根据当前的循环指针 r 取出第 r 条记录的残差值
                size_t row_idx = res_indices[r]; 
                rand_out << "\t" << res.rand_eff[resid_effect_idx](row_idx, 0);
                if (get_PEV) {
                    rand_out << "\t" << res.rand_eff[resid_effect_idx](row_idx, 1) 
                             << "\t" << res.rand_eff[resid_effect_idx](row_idx, 2);
                }
            }
            
            rand_out << "\n";
        }
    }
    rand_out.close();
    // -------------------------------------------------------------
    // 4. 写出 .gwas 文件并计算 Lambda GC 和 Mean Chi-square
    // -------------------------------------------------------------
    if (do_test && !res.gwas_res.is_empty()) {
        std::unordered_map<size_t, size_t> bim_2_gwas;
        for (size_t i = 0; i < ind_col.n_elem; ++i) {
            bim_2_gwas[ind_col[i]] = i; 
        }
        std::string bed_path = bed_file;
        if (bed_path.size() < 4 || bed_path.substr(bed_path.size() - 4) != ".bed") {
            bed_path += ".bed";
        }
        std::string bim_file = bed_path.substr(0, bed_path.size() - 4) + ".bim";

        std::ifstream bim_in(bim_file);
        if (!bim_in.is_open()) throw std::runtime_error("Cannot open BIM file for GWAS output: " + bim_file);
        
        std::ofstream gwas_out(out_prefix + ".mlm");
        if (!gwas_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".mlm");

        gwas_out << "SNP\tCHR\tPOS\tA1\tA2\tFreq\tEffect\tSE\tPvalue\n";
        
        std::string bim_line;
        size_t current_bim_idx = 0;
        
        while (std::getline(bim_in, bim_line)) {
            if (bim_line.empty()) continue;
            
            auto it = bim_2_gwas.find(current_bim_idx);
            if (it != bim_2_gwas.end()) {
                size_t mat_row = it->second;
                std::vector<std::string> tokens = split_line(bim_line);
                if (tokens.size() >= 6) {
                    // BIM列: [0]CHR [1]SNP [2]CM [3]POS [4]A1 [5]A2
                    gwas_out << tokens[1] << "\t" << tokens[0] << "\t" << tokens[3] << "\t"
                             << tokens[4] << "\t" << tokens[5] << "\t"
                             << res.gwas_res(mat_row, 3) << "\t" // Freq
                             << res.gwas_res(mat_row, 0) << "\t" // Effect
                             << res.gwas_res(mat_row, 1) << "\t" // SE
                             << res.gwas_res(mat_row, 2) << "\n"; // Pvalue
                }
            }
            current_bim_idx++;
        }
        bim_in.close();
        gwas_out.close();

        // ---- 计算群体遗传学统计参数 (Mean Chi-square & Lambda GC) ----
        arma::vec effects = res.gwas_res.col(0);
        arma::vec ses = res.gwas_res.col(1);
        
        // 防御性剔除可能存在的零方差 (SE接近0) 引发的除0异常
        arma::uvec valid_snps = arma::find(ses > 1e-12); 
        
        if (valid_snps.n_elem > 0) {
            arma::vec chi_square = arma::square(effects.elem(valid_snps) / ses.elem(valid_snps));
            double mean_chi2 = arma::mean(chi_square);
            double median_chi2 = arma::median(chi_square);
            // qchisq(0.5, 1) = 0.45493642311
            double lambda_gc = median_chi2 / 0.45493642311; 

            std::cout << "-----------------------------------------------------------\n";
            std::cout << "GWAS scan completed for " << valid_snps.n_elem << " valid SNPs.\n";
            std::cout << "Genomic Control (Lambda GC): " << std::fixed << std::setprecision(5) << lambda_gc << "\n";
            std::cout << "Mean Chi-square            : " << std::fixed << std::setprecision(5) << mean_chi2 << "\n";
            std::cout << "-----------------------------------------------------------\n";
        }
    }
    
    std::cout << "Results successfully written to disk with prefix: " << out_prefix << "\n";
}

/**
 * 将 Batch GWAS (多表型) 模型计算结果统一格式化写入本地磁盘文件
 */
template<typename T>
void write_out_batch(
    const batch_result& res,
    const DataInput<T>& data_info,
    const std::string& out_prefix,
    const std::string& bed_file,
    const arma::uvec& ind_col,
    bool do_test
) {
    std::cout << "Writing Batch Results...\n";
    
    size_t np = data_info.phe_names.size();

    // -------------------------------------------------------------
    // 1. 写出 .var 文件 (多表型的方差组分及标准误)
    // -------------------------------------------------------------
    std::ofstream var_out(out_prefix + ".var");
    if (!var_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".var");
    
    // 表头：包含性状名、Vg、Vg的标准误、Ve、Ve的标准误，以及遗传力 h2
    var_out << "Trait\tVg\tSE_Vg\tVe\tSE_Ve\th2\n";
    for (size_t i = 0; i < np; ++i) {
        double vg = res.vg(i);
        double ve = res.ve(i);
        double se_vg = res.se_vg(i);
        double se_ve = res.se_ve(i);
        
        double sum_vc = vg + ve;
        if (sum_vc == 0.0) sum_vc = 1e-16; // 防除零保护
        double h2 = vg / sum_vc;
        
        var_out << data_info.phe_names[i] << "\t"
                << vg << "\t" << se_vg << "\t"
                << ve << "\t" << se_ve << "\t"
                << h2 << "\n";
    }
    var_out.close();

    // -------------------------------------------------------------
    // 2. 写出 .rand 文件 (多表型随机效应汇总，基于ID并集)
    // -------------------------------------------------------------
    std::vector<std::string> unique_ids;
    std::unordered_set<std::string> seen_ids;    

    for (const auto& vec : data_info.grm_ids) {
        for (const auto& id : vec) {
            // seen_ids.insert(id).second 会返回一个布尔值：如果是新插入的，返回 true；如果已经存在，返回 false。
            if (seen_ids.insert(id).second) {
                unique_ids.push_back(id); // 只有新 ID 才追加到顺序列表中
            }
        }
    }
    for (const auto& vec : data_info.rand_ids) {
        for (const auto& id : vec) {
            if (seen_ids.insert(id).second) {
                unique_ids.push_back(id);
            }
        }
    }

    std::vector<std::unordered_map<std::string, size_t>> grm_id_dicts(data_info.grm_names.size());
    for (size_t i = 0; i < data_info.grm_names.size(); ++i) {
        for (size_t j = 0; j < data_info.grm_ids[i].size(); ++j) {
            grm_id_dicts[i][data_info.grm_ids[i][j]] = j;
        }
    }
    std::vector<std::unordered_map<std::string, size_t>> rand_id_dicts(data_info.rand_names.size());
    for (size_t i = 0; i < data_info.rand_names.size(); ++i) {
        for (size_t j = 0; j < data_info.rand_ids[i].size(); ++j) {
            rand_id_dicts[i][data_info.rand_ids[i][j]] = j;
        }
    }

    std::ofstream rand_out(out_prefix + ".rand");
    if (!rand_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".rand");

    rand_out << "ID";
    for (const auto& gname : data_info.grm_names) {
        for (const auto& pname : data_info.phe_names) {
            rand_out << "\t" << gname << "_" << pname << "_Effect";
        }
    }
    for (const auto& rname : data_info.rand_names) {
        for (const auto& pname : data_info.phe_names) {
            rand_out << "\t" << rname << "_" << pname << "_Effect";
        }
    }
    rand_out << "\n";

    for (const auto& id : unique_ids) {
        rand_out << id;
        size_t matrix_idx = 0; 
        
        // GRM 随机效应
        for (size_t i = 0; i < data_info.grm_names.size(); ++i) {
            auto it = grm_id_dicts[i].find(id);
            if (it != grm_id_dicts[i].end()) {
                size_t row_idx = it->second;
                for (size_t p = 0; p < np; ++p) rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, p);
            } else {
                for (size_t p = 0; p < np; ++p) rand_out << "\tNA";
            }
            matrix_idx++;
        }
        
        // 额外分类随机效应
        for (size_t i = 0; i < data_info.rand_names.size(); ++i) {
            auto it = rand_id_dicts[i].find(id);
            if (it != rand_id_dicts[i].end()) {
                size_t row_idx = it->second;
                for (size_t p = 0; p < np; ++p) rand_out << "\t" << res.rand_eff[matrix_idx](row_idx, p);
            } else {
                for (size_t p = 0; p < np; ++p) rand_out << "\tNA";
            }
            matrix_idx++;
        }
        rand_out << "\n";
    }
    rand_out.close();

    // -------------------------------------------------------------
    // 3. 写出 .mlm 文件 (多表型 GWAS 结果)
    // -------------------------------------------------------------
    if (do_test && !res.gwas_res.is_empty()) {
        std::unordered_map<size_t, size_t> bim_2_gwas;
        for (size_t i = 0; i < ind_col.n_elem; ++i) {
            bim_2_gwas[ind_col[i]] = i; 
        }
        std::string bed_path = bed_file;
        if (bed_path.size() < 4 || bed_path.substr(bed_path.size() - 4) != ".bed") {
            bed_path += ".bed";
        }
        std::string bim_file = bed_path.substr(0, bed_path.size() - 4) + ".bim";

        std::ifstream bim_in(bim_file);
        if (!bim_in.is_open()) throw std::runtime_error("Cannot open BIM file for GWAS output: " + bim_file);
        
        std::ofstream gwas_out(out_prefix + ".batchmlm");
        if (!gwas_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".batchmlm");

        bool has_variance = (res.gwas_res.n_cols == np * 5 + 1);
        int cols_per_trait = has_variance ? 5 : 3;

        gwas_out << "SNP\tCHR\tPOS\tA1\tA2\tFreq";
        for (const auto& pname : data_info.phe_names) {
            gwas_out << "\t" << pname << "_Effect\t" << pname << "_SE\t" << pname << "_Pvalue";
            if (has_variance) {
                gwas_out << "\t" << pname << "_Vg\t" << pname << "_Ve";
            }
        }
        gwas_out << "\n";
        
        std::string bim_line;
        size_t current_bim_idx = 0;
        
        while (std::getline(bim_in, bim_line)) {
            if (bim_line.empty()) continue;
            
            auto it = bim_2_gwas.find(current_bim_idx);
            if (it != bim_2_gwas.end()) {
                size_t mat_row = it->second;
                std::vector<std::string> tokens = split_line(bim_line);
                if (tokens.size() >= 6) {
                    // BIM列: [0]CHR [1]SNP [2]CM [3]POS [4]A1 [5]A2
                    gwas_out << tokens[1] << "\t" << tokens[0] << "\t" << tokens[3] << "\t"
                             << tokens[4] << "\t" << tokens[5] << "\t"
                             << res.gwas_res(mat_row, 0); // 频率 Freq (第 0 列)
                    
                    for (size_t p = 0; p < np; ++p) {
                        gwas_out << "\t" << res.gwas_res(mat_row, 1 + p * cols_per_trait)
                                 << "\t" << res.gwas_res(mat_row, 2 + p * cols_per_trait)
                                 << "\t" << res.gwas_res(mat_row, 3 + p * cols_per_trait);
                        if (has_variance) {
                            gwas_out << "\t" << res.gwas_res(mat_row, 4 + p * cols_per_trait)
                                     << "\t" << res.gwas_res(mat_row, 5 + p * cols_per_trait);
                        }
                    }
                    gwas_out << "\n";
                }
            }
            current_bim_idx++;
        }
        bim_in.close();
        gwas_out.close();

        // ---- 计算并写回每个表型的群体遗传学统计参数 (.log 文件) ----
        std::ofstream log_out(out_prefix + ".batchSummary");
        if (!log_out) throw std::runtime_error("Cannot create file: " + out_prefix + ".batchSummary");

        // 写入表头
        log_out << "Trait\tValid_SNPs\tLambda_GC\tMean_Chi_square\n";

        for (size_t p = 0; p < np; ++p) {
            
            arma::vec effects = res.gwas_res.col(1 + p * cols_per_trait);
            arma::vec ses = res.gwas_res.col(2 + p * cols_per_trait);
            
            // 防御性过滤：剔除 SE 过小或非法的 SNP
            arma::uvec valid_idx = arma::find(ses > 1e-12 && arma::find_finite(effects)); 
            
            if (valid_idx.n_elem > 0) {
                // 计算卡方值：(beta / se)^2
                arma::vec chi_square = arma::square(effects.elem(valid_idx) / ses.elem(valid_idx));
                
                double mean_chi2 = arma::mean(chi_square);
                double median_chi2 = arma::median(chi_square);
                // 基因组控制膨胀因子 Lambda GC 计算公式：中位数 / 0.4549
                double lambda_gc = median_chi2 / 0.45493642311; 

                // 格式化写入文件：每一行一个表型
                log_out << data_info.phe_names[p] << "\t" 
                        << valid_idx.n_elem << "\t"
                        << std::fixed << std::setprecision(5) << lambda_gc << "\t"
                        << std::setprecision(5) << mean_chi2 << "\n";
            } else {
                // 若无有效数据，填入 NA
                log_out << data_info.phe_names[p] << "\t0\tNA\tNA\n";
            }
        }
        log_out.close();
    }
    
    std::cout << "Batch results successfully written to disk with prefix: " << out_prefix << "\n";
}


template void write_out_batch<double>(const batch_result&, const DataInput<double>&, const std::string&, const std::string&, const arma::uvec&, bool);
template void write_out_batch<float>(const batch_result&, const DataInput<float>&, const std::string&, const std::string&, const arma::uvec&, bool);

template DataInput<double> load_data_cpp<double>(const std::string&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<std::string>&, const std::vector<std::string>&, const std::string&, const std::vector<size_t>&, bool);
template DataInput<float> load_data_cpp<float>(const std::string&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<std::string>&, const std::vector<std::string>&, const std::string&, const std::vector<size_t>&, bool);

template void write_out_single<double>(const single_result&, const DataInput<double>&, const std::string&, const std::string&, const arma::uvec&, bool, bool);
template void write_out_single<float>(const single_result&, const DataInput<float>&, const std::string&, const std::string&, const arma::uvec&, bool, bool);

} // namespace rbp