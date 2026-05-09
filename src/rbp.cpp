#include "rbp.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <streambuf>
#include <sstream>

// ============================================================================
// 日志重定向流缓冲类 (TeeBuf)
// ============================================================================
class TeeBuf : public std::streambuf {
    std::streambuf *sb1, *sb2;
public:
    TeeBuf(std::streambuf *sb1, std::streambuf *sb2) : sb1(sb1), sb2(sb2) {}
protected:
    virtual int overflow(int c) {
        if (c == EOF) return !EOF;
        int r1 = sb1->sputc(c);
        int r2 = sb2->sputc(c);
        return (r1 == EOF || r2 == EOF) ? EOF : c;
    }
    virtual int sync() {
        int r1 = sb1->pubsync();
        int r2 = sb2->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }
};

// ============================================================================
// 软件 Logo 打印模块
// ============================================================================
void print_logo() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ██████╗ ██████╗ ██████╗     VERSION: 1.13                ║\n";
    std::cout << "║  ██╔══██╗██╔══██╗██╔══██╗    Platform: Linux x86_64       ║\n";
    std::cout << "║  ██████╔╝██████╔╝██████╔╝                                 ║\n";
    std::cout << "║  ██╔══██╗██╔══██╗██╔═══╝   * Rapid                        ║\n";
    std::cout << "║  ██║  ██║██████╔╝██║       * Batch-process-supported      ║\n";
    std::cout << "║  ╚═╝  ╚═╝╚═════╝ ╚═╝       * Powerful tool for GWAS & GS  ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "Developers: Fangjun Xu and Mengjin Zhu \n";
    std::cout << "Authorized to: HZAU \n\n";
}

// ============================================================================
// 命令行参数解析
// ============================================================================

enum class CommandMode { MAKE_GRM, KPCA, SINGLE_TRAIT, BATCH_TRAIT, HELP };

struct CommandLineArgs {
    CommandMode mode = CommandMode::HELP;
    std::vector<std::string> original_argv; 
    
    // 通用参数
    std::string geno_file;
    std::string out_prefix = "rbp";  
    int ncpus = 1;
    bool use_float = false;          
    bool is_chunk = false;           
    bool verbose = true;
    
    // 基因型过滤参数 
    std::vector<int> geno_ind;        
    std::vector<int> geno_snp;        

    // GRM 参数
    std::vector<std::string> grm_files;
    std::string weight_file;         
    int method = 1;                  
    size_t snp_chunk_size = 10000;
    size_t ind_chunk_size = 512;
    bool calc_inverse = false;       
    
    // kPCA 参数
    int npc = 10;                    
    double tol_lanczos = 1e-6; 

    // Single-Trait / Batch-Trait GWAS/REML 参数
    std::string phe_file;
    std::vector<int> phe_cols;       // 统一使用 phe-col 解析为 vector
    std::vector<int> fix_cols;
    std::vector<int> cov_cols;
    std::vector<int> rand_cols;
    std::vector<std::string> rand_covs;
    std::string test_method = "standard";
    int criterion = 0;
    std::string solver = "CG"; 
    int num_ranvec = 100;
    bool init_mom = false;
    int random_seed = 6666;
    int re_gamma_size = 500;  
    bool re_estimate_gamma = false;
    int maxit = 30;
    int cg_maxit = 500;
    double tol = 1e-5;
    double cg_tol = 1e-6;
    bool is_repeat = false;
    bool get_pev = false;
    bool do_gwas = false;
};

void print_help() {
    std::cout << "USAGE:\n  RBP <command> [options]\n\n";
    std::cout << "COMMANDS:\n";
    std::cout << "  --make-grm         Calculate genomic relationship matrix (GRM)\n";
    std::cout << "  --kpca             Calculate principal components from GRM\n";
    std::cout << "  --single-trait     Run single-trait GWAS / AI-REML analysis\n";
    std::cout << "  --batch-trait      Run multi-trait batch GWAS / AI-REML analysis\n";
    std::cout << "  --help, -h         Show this help message\n\n";
    
    std::cout << "COMMON OPTIONS:\n";
    std::cout << "  --geno FILE        Genotype file (PLINK prefix)\n";
    std::cout << "  --grm LIST         GRM prefix(es), or .txt list file(s) (space-separated)\n";
    std::cout << "  --out PREFIX       Output file prefix (default: rbp)\n";
    std::cout << "  --ncpus N          Number of CPU threads (default: 1)\n";
    std::cout << "  --float            Use single-precision (float) to save memory\n";
    std::cout << "  --quiet            Do not print message\n";
    std::cout << "  --chunk            Force chunked mode for large-scale data, NOT applicable for batch\n\n";

    std::cout << "GENOTYPE FILTERING OPTIONS:\n";
    std::cout << "  --geno-ind STR     Individual indices to include (1-based, comma-separated OR a file path)\n";
    std::cout << "  --geno-snp STR     SNP indices to include (1-based, comma-separated OR a file path)\n\n";
    
    std::cout << "GRM OPTIONS (--make-grm):\n";
    std::cout << "  --method N         1=Yang, 2=VanRaden, 3=Dominance (default: 1)\n";
    std::cout << "  --weight FILE      1-column text file for SNP weights\n";
    std::cout << "  --snp-chunk N      SNP chunk size (default: 10000)\n";
    std::cout << "  --ind-chunk N      Individual chunk size for N>50k (default: 512)\n";
    std::cout << "  --inv              Calculate the generalized inverse of GRM\n\n";

    std::cout << "kPCA OPTIONS (--kpca):\n";
    std::cout << "  --npc N            Number of top principal components (default: 10)\n";
    std::cout << "  --tol VAL          Lanczos convergence tolerance (default: 1e-6)\n\n";

    std::cout << "TRAIT OPTIONS (--single-trait / --batch-trait):\n";
    std::cout << "  --phe FILE         Phenotype file (header required)\n";
    std::cout << "  --phe-col STR      Phenotype column index (1-based, comma-separated for batch mode)\n";
    std::cout << "  --fix-col STR      Categorical fixed effect columns (1-based, comma-separated)\n";
    std::cout << "  --cov-col STR      Quantitative covariate columns (1-based, comma-separated)\n";
    std::cout << "  --rand-col STR     Additional random effect columns (single-trait only, comma-separated)\n";
    std::cout << "  --rand-cov LIST    Covariance files or .txt list for random effects (space-separated, '1' for identity)\n";
    std::cout << "  --test METHOD      GWAS test method: standard, quasi-standard(NOT applicable for batch), gamma, exact(NOT applicable for single)\n";
    std::cout << "  --gwas             Perform GWAS scan\n";
    std::cout << "  --solver METHOD    AI-REML solver: CG (default) or Cholesky\n";
    std::cout << "  --mom              Enable MoM-RHE initialization\n";
    std::cout << "  --pev              Calculate prediction error variance (PEV) for random effects\n";
    std::cout << "  --seed N           Random seed (default: 6666)\n";
    std::cout << "  --num-ranvec N     Number of random vectors for quasi-standard/gamma (default: 100)\n";
    std::cout << "  --re-gamma-size N  Number of SNPs to estimate gamma (default: 500)\n";
    std::cout << "  --re-est-gamma     Force re-estimation of gamma\n";
    std::cout << "  --maxit N          Max iterations for AI-REML (default: 30)\n";
    std::cout << "  --tol VAL          Tolerance for AI-REML (default: 1e-5)\n";
    std::cout << "  --cg-maxit N       Max iterations for CG solver (default: 500)\n";
    std::cout << "  --cg-tol VAL       Tolerance for CG solver (default: 1e-6)\n";
    std::cout << "  --repeat           Indicate repeated measures analysis for single trait\n\n";
}

/**
 * 通用解析器：判断输入是文件路径还是逗号分隔的字符串
 */
void parse_indices_or_file(const std::string& input, std::vector<int>& target) {
    std::ifstream in(input);
    if (in) { // 能成功打开，说明是文件
        int val;
        while (in >> val) {
            target.push_back(val);
        }
    } else { // 否则，按逗号分隔字符串解析
        std::istringstream iss(input);
        int val;
        while (iss >> val) {
            target.push_back(val);
            if (iss.peek() == ',') iss.ignore();
        }
    }
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;
    if (argc < 2) { print_help(); exit(1); }
    
    for (int i = 0; i < argc; ++i) {
        args.original_argv.push_back(argv[i]);
    }
    
    std::string first_arg = argv[1];
    if (first_arg == "--make-grm") args.mode = CommandMode::MAKE_GRM;
    else if (first_arg == "--kpca") args.mode = CommandMode::KPCA;
    else if (first_arg == "--single-trait") args.mode = CommandMode::SINGLE_TRAIT;
    else if (first_arg == "--batch-trait") args.mode = CommandMode::BATCH_TRAIT;
    else if (first_arg == "--help" || first_arg == "-h") { print_help(); exit(0); } 
    else { std::cerr << "Error: Unknown command '" << first_arg << "'\n\n"; print_help(); exit(1); }
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--geno" && i + 1 < argc) args.geno_file = argv[++i];
        
        else if (arg == "--grm") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string val = argv[++i];
                if (val.size() >= 4 && (val.substr(val.size() - 4) == ".txt" || val.substr(val.size() - 5) == ".list")) {
                    std::ifstream lst(val);
                    if (!lst) throw std::runtime_error("Cannot open GRM list file: " + val);
                    std::string line;
                    while (std::getline(lst, line)) {
                        if (!line.empty()) args.grm_files.push_back(line);
                    }
                    lst.close();
                } else {
                    args.grm_files.push_back(val);
                }
            }
        }
        
        else if (arg == "--out" && i + 1 < argc) args.out_prefix = argv[++i];
        else if (arg == "--ncpus" && i + 1 < argc) args.ncpus = std::stoi(argv[++i]);
        else if (arg == "--float") args.use_float = true;
        else if (arg == "--chunk") args.is_chunk = true;
        else if (arg == "--quiet") args.verbose = false;
        
        // 基因型过滤参数 (复用函数智能解析)
        else if (arg == "--geno-ind" && i + 1 < argc) parse_indices_or_file(argv[++i], args.geno_ind);
        else if (arg == "--geno-snp" && i + 1 < argc) parse_indices_or_file(argv[++i], args.geno_snp);

        // GRM specific
        else if (arg == "--method" && i + 1 < argc) args.method = std::stoi(argv[++i]);
        else if (arg == "--weight" && i + 1 < argc) args.weight_file = argv[++i];
        else if (arg == "--snp-chunk" && i + 1 < argc) args.snp_chunk_size = std::stoull(argv[++i]);
        else if (arg == "--ind-chunk" && i + 1 < argc) args.ind_chunk_size = std::stoull(argv[++i]);
        else if (arg == "--inv") args.calc_inverse = true;
        
        // kPCA specific
        else if (arg == "--npc" && i + 1 < argc) args.npc = std::stoi(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) args.tol_lanczos = std::stod(argv[++i]);

        // Trait specific
        else if (arg == "--phe" && i + 1 < argc) args.phe_file = argv[++i];
        else if (arg == "--phe-col" && i + 1 < argc) {
            std::istringstream iss(argv[++i]); int val;
            while (iss >> val) { args.phe_cols.push_back(val); if (iss.peek() == ',') iss.ignore(); }
        }
        else if (arg == "--fix-col" && i + 1 < argc) {
            std::istringstream iss(argv[++i]); int val;
            while (iss >> val) { args.fix_cols.push_back(val); if (iss.peek() == ',') iss.ignore(); }
        }
        else if (arg == "--cov-col" && i + 1 < argc) {
            std::istringstream iss(argv[++i]); int val;
            while (iss >> val) { args.cov_cols.push_back(val); if (iss.peek() == ',') iss.ignore(); }
        }
        else if (arg == "--rand-col" && i + 1 < argc) {
            std::istringstream iss(argv[++i]); int val;
            while (iss >> val) { args.rand_cols.push_back(val); if (iss.peek() == ',') iss.ignore(); }
        }
        else if (arg == "--rand-cov") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string val = argv[++i];
                if (val.size() >= 4 && (val.substr(val.size() - 4) == ".txt" || val.substr(val.size() - 5) == ".list")) {
                    std::ifstream lst(val);
                    if (!lst) throw std::runtime_error("Cannot open Rand-Cov list file: " + val);
                    std::string line;
                    while (std::getline(lst, line)) {
                        if (!line.empty()) args.rand_covs.push_back(line);
                        else args.rand_covs.push_back("1");
                    }
                    lst.close();
                } else {
                    args.rand_covs.push_back(val);
                }
            }
        }
        
        else if (arg == "--test" && i + 1 < argc) args.test_method = argv[++i];
        else if (arg == "--criterion" && i + 1 < argc) args.criterion = std::stoi(argv[++i]);
        else if (arg == "--solver" && i + 1 < argc) args.solver = argv[++i];
        else if (arg == "--num-ranvec" && i + 1 < argc) args.num_ranvec = std::stoi(argv[++i]);
        else if (arg == "--mom") args.init_mom = true;
        else if (arg == "--seed" && i + 1 < argc) args.random_seed = std::stoi(argv[++i]);
        else if (arg == "--re-gamma-size" && i + 1 < argc) args.re_gamma_size = std::stoi(argv[++i]);
        else if (arg == "--re-est-gamma") args.re_estimate_gamma = true;
        else if (arg == "--maxit" && i + 1 < argc) args.maxit = std::stoi(argv[++i]);
        else if (arg == "--cg-maxit" && i + 1 < argc) args.cg_maxit = std::stoi(argv[++i]);
        else if (arg == "--tol" && i + 1 < argc) args.tol = std::stod(argv[++i]);
        else if (arg == "--cg-tol" && i + 1 < argc) args.cg_tol = std::stod(argv[++i]);
        else if (arg == "--repeat") args.is_repeat = true;
        else if (arg == "--pev") args.get_pev = true;
        else if (arg == "--gwas") args.do_gwas = true;
    }

    // 容错处理：如果未指定表型列，默认选择第 1 列
    if (args.phe_cols.empty()) args.phe_cols.push_back(1);

    return args;
}


// ============================================================================
// 主程序入口
// ============================================================================
int main(int argc, char* argv[]) {
    auto start_system_time = std::chrono::system_clock::now();
    std::time_t start_time_c = std::chrono::system_clock::to_time_t(start_system_time);
    auto start_time_steady = std::chrono::steady_clock::now();

    CommandLineArgs args;
    try { args = parse_args(argc, argv); } catch (...) { return 1; }
    
    // 初始化时统一接管线程
    if (args.ncpus > 0) { 
        rbp::blas_set_num_threads(args.ncpus); 
    }

    std::string log_file_name = args.out_prefix + ".log";
    std::ofstream log_file(log_file_name);
    if (!log_file) {
        std::cerr << "Fatal Error: Cannot create log file at: " << log_file_name << "\n";
        return 1;
    }
    
    std::streambuf* old_cout_buf = nullptr;
    TeeBuf tee_buf(std::cout.rdbuf(), log_file.rdbuf());
    
    if (args.verbose) {
        old_cout_buf = std::cout.rdbuf(&tee_buf); // 屏幕和日志文件双写
    } else {
        old_cout_buf = std::cout.rdbuf(log_file.rdbuf()); // 屏幕静默，所有 cout 仅写入 .log 文件
    }
    std::streambuf* old_cerr_buf = std::cerr.rdbuf(&tee_buf);

    try {
        print_logo();
        
        std::cout << "Commands:\n" << args.original_argv[0];
        for (size_t i = 1; i < args.original_argv.size(); ++i) {
            if (args.original_argv[i][0] == '-') std::cout << " \\\n  " << args.original_argv[i];
            else std::cout << " " << args.original_argv[i];
        }
        std::cout << "\n\n";
        
        std::cout << "[Start Time]: " << std::put_time(std::localtime(&start_time_c), "%Y-%m-%d %H:%M:%S") << "\n";
        std::cout << "Threads: " << args.ncpus << "\n";
        std::cout << "Precision: " << (args.use_float ? "Single (Float)" : "Double") << "\n";
        std::cout << "Output Prefix: " << args.out_prefix << "\n\n";
        
        switch (args.mode) {
            case CommandMode::MAKE_GRM: {
                std::cout << "███████████████████████████████████████████████████████████\n";
                std::cout << "█ ► GRM Calculation (--make-grm)                          █\n";
                std::cout << "███████████████████████████████████████████████████████████\n\n";
                
                if (args.geno_file.empty()) throw std::runtime_error("Genotype file (--geno) is required.");
                
                std::string base_path = args.geno_file;
                if (base_path.size() >= 4 && base_path.substr(base_path.size()-4) == ".bed") {
                    base_path = base_path.substr(0, base_path.size()-4);
                }
                
                size_t n_samples = rbp::get_file_lines(base_path + ".fam");
                size_t n_snps = rbp::get_file_lines(base_path + ".bim");
                
                arma::uvec ind_row, ind_col;
                if (!args.geno_ind.empty()) ind_row = arma::conv_to<arma::uvec>::from(args.geno_ind) - 1;
                else ind_row = arma::regspace<arma::uvec>(0, n_samples - 1);
                
                if (!args.geno_snp.empty()) ind_col = arma::conv_to<arma::uvec>::from(args.geno_snp) - 1;
                else ind_col = arma::regspace<arma::uvec>(0, n_snps - 1);

                arma::vec weights_double;
                if (!args.weight_file.empty()) {
                    std::cout << "Loading SNP weights from " << args.weight_file << " ...\n";
                    if (!weights_double.load(args.weight_file, arma::raw_ascii)) throw std::runtime_error("Failed to load weight file.");
                }
                
                if (args.use_float) {
                    arma::fvec weights_float = arma::conv_to<arma::fvec>::from(weights_double);
                    rbp::make_KIN<float>(args.geno_file, args.out_prefix, ind_row, ind_col, weights_float, 
                                         args.method, args.snp_chunk_size, args.ind_chunk_size, args.is_chunk, args.calc_inverse, args.ncpus);
                } else {
                    rbp::make_KIN<double>(args.geno_file, args.out_prefix, ind_row, ind_col, weights_double, 
                                          args.method, args.snp_chunk_size, args.ind_chunk_size, args.is_chunk, args.calc_inverse, args.ncpus);
                }
                break;
            }
            
            case CommandMode::KPCA: {
                std::cout << "███████████████████████████████████████████████████████████\n";
                std::cout << "█ ► Principal Component Analysis (--kpca)                 █\n";
                std::cout << "███████████████████████████████████████████████████████████\n\n";
                
                if (args.grm_files.empty()) throw std::runtime_error("GRM file (--grm) is required.");
                if (args.grm_files.size() > 1) {
                    std::cout << "[WARNING] Multiple GRMs provided. kPCA module will ONLY process the first one: " 
                              << args.grm_files[0] << std::endl;
                }
                std::string kpca_grm = args.grm_files[0];
                
                if (args.use_float) rbp::kPCA<float>(kpca_grm, args.npc, args.tol_lanczos, args.is_chunk, args.ncpus);
                else                rbp::kPCA<double>(kpca_grm, args.npc, args.tol_lanczos, args.is_chunk, args.ncpus);
                break;
            }

            case CommandMode::SINGLE_TRAIT: {
                std::cout << "███████████████████████████████████████████████████████████\n";
                std::cout << "█ ► Single Trait Analysis (--single-trait)                █\n";
                std::cout << "███████████████████████████████████████████████████████████\n\n";
                
                
                if (args.phe_file.empty()) throw std::runtime_error("Phenotype file (--phe) is required.");
                if (args.geno_file.empty()) throw std::runtime_error("Genotype prefix (--geno) is required for matching IDs.");
                
                // 将用户的 1-based 索引转换为 C++ 底层的 0-based 索引
                std::vector<int> phe_pos = { args.phe_cols[0] - 1 }; // 单性状仅取第一个表型
                std::vector<int> fix_pos; for (int c : args.fix_cols) fix_pos.push_back(c - 1);
                std::vector<int> cov_pos; for (int c : args.cov_cols) cov_pos.push_back(c - 1);
                std::vector<int> rand_pos; for (int c : args.rand_cols) rand_pos.push_back(c - 1);
                
                // 对齐附加随机效应的协方差文件 (不足部分用 "1" 表示单位阵)
                std::vector<std::string> rand_cov_files = args.rand_covs;
                while (rand_cov_files.size() < rand_pos.size()) rand_cov_files.push_back("1"); 
                
                std::string base_path = args.geno_file;
                if (base_path.size() >= 4 && base_path.substr(base_path.size()-4) == ".bed") {
                    base_path = base_path.substr(0, base_path.size()-4);
                }
                size_t n_samples = rbp::get_file_lines(base_path + ".fam");
                size_t n_snps = rbp::get_file_lines(base_path + ".bim");
                
                arma::uvec ind_row, ind_col;
                if (!args.geno_ind.empty()) ind_row = arma::conv_to<arma::uvec>::from(args.geno_ind) - 1;
                else ind_row = arma::regspace<arma::uvec>(0, n_samples - 1);
                
                if (!args.geno_snp.empty()) ind_col = arma::conv_to<arma::uvec>::from(args.geno_snp) - 1;
                else ind_col = arma::regspace<arma::uvec>(0, n_snps - 1);
                
                std::vector<size_t> geno_row_selected = arma::conv_to<std::vector<size_t>>::from(ind_row);

                if (args.use_float) {
                    auto data = rbp::load_data_cpp<float>(
                        args.phe_file, phe_pos, fix_pos, cov_pos, rand_pos, rand_cov_files, 
                        args.grm_files, args.geno_file, geno_row_selected, args.is_chunk
                    );
                    
                    arma::Col<float> y = data.y.col(0);
                    arma::Col<float> pars0; // 空向量，如果不通过 --mom 就会在内部初始化
                    
                    auto res = rbp::single_model_cpp<float>(
                        y, data.X, data.Z, data.K, args.solver, args.test_method, args.criterion,
                        args.out_prefix, args.num_ranvec, args.geno_file, ind_row, ind_col, data.Z_geno,
                        args.is_chunk, pars0, args.init_mom, args.random_seed, args.re_gamma_size,
                        args.re_estimate_gamma, args.maxit, args.cg_maxit, args.tol, args.cg_tol,
                        args.ncpus, args.is_repeat, args.get_pev, args.do_gwas
                    );
                    
                    rbp::write_out_single<float>(
                        res, data, args.out_prefix, args.geno_file, ind_col, args.get_pev, args.do_gwas
                    );
                } else {
                    auto data = rbp::load_data_cpp<double>(
                        args.phe_file, phe_pos, fix_pos, cov_pos, rand_pos, rand_cov_files, 
                        args.grm_files, args.geno_file, geno_row_selected, args.is_chunk
                    );
                    
                    arma::Col<double> y = data.y.col(0);
                    arma::Col<double> pars0; 
                    
                    auto res = rbp::single_model_cpp<double>(
                        y, data.X, data.Z, data.K, args.solver, args.test_method, args.criterion,
                        args.out_prefix, args.num_ranvec, args.geno_file, ind_row, ind_col, data.Z_geno,
                        args.is_chunk, pars0, args.init_mom, args.random_seed, args.re_gamma_size,
                        args.re_estimate_gamma, args.maxit, args.cg_maxit, args.tol, args.cg_tol,
                        args.ncpus, args.is_repeat, args.get_pev, args.do_gwas
                    );
                    
                    rbp::write_out_single<double>(
                        res, data, args.out_prefix, args.geno_file, ind_col, args.get_pev, args.do_gwas
                    );
                }
                break;
            }

            case CommandMode::BATCH_TRAIT: {
                std::cout << "███████████████████████████████████████████████████████████\n";
                std::cout << "█ ► Batch Trait Analysis (--batch-trait)                  █\n";
                std::cout << "███████████████████████████████████████████████████████████\n\n";
                
                if (args.phe_file.empty()) throw std::runtime_error("Phenotype file (--phe) is required.");
                if (args.geno_file.empty()) throw std::runtime_error("Genotype prefix (--geno) is required.");
                
                // Batch 模式特有限制：
                if (args.is_chunk) {
                    std::cout << "[WARNING] Chunk mode is NOT supported in batch-trait module. Forcing full memory mapping (--chunk ignored).\n";
                    args.is_chunk = false;
                }
                if (args.grm_files.size() != 1) {
                    throw std::runtime_error("Batch-trait module strictly requires exactly ONE GRM input (--grm).");
                }
                if (!args.rand_cols.empty() || !args.rand_covs.empty()) {
                    throw std::runtime_error("Batch-trait module does NOT support additional random effects (--rand-col / --rand-cov).");
                }

                // Batch 模式读取所有由 --phe-col 传进来的性状列
                std::vector<int> phe_pos; for (int c : args.phe_cols) phe_pos.push_back(c - 1);
                std::vector<int> fix_pos; for (int c : args.fix_cols) fix_pos.push_back(c - 1);
                std::vector<int> cov_pos; for (int c : args.cov_cols) cov_pos.push_back(c - 1);
                std::vector<int> rand_pos; 
                std::vector<std::string> rand_cov_files; 
                
                std::string base_path = args.geno_file;
                if (base_path.size() >= 4 && base_path.substr(base_path.size()-4) == ".bed") {
                    base_path = base_path.substr(0, base_path.size()-4);
                }
                size_t n_samples = rbp::get_file_lines(base_path + ".fam");
                size_t n_snps = rbp::get_file_lines(base_path + ".bim");
                
                arma::uvec ind_row, ind_col;
                if (!args.geno_ind.empty()) ind_row = arma::conv_to<arma::uvec>::from(args.geno_ind) - 1;
                else ind_row = arma::regspace<arma::uvec>(0, n_samples - 1);
                
                if (!args.geno_snp.empty()) ind_col = arma::conv_to<arma::uvec>::from(args.geno_snp) - 1;
                else ind_col = arma::regspace<arma::uvec>(0, n_snps - 1);
                
                std::vector<size_t> geno_row_selected = arma::conv_to<std::vector<size_t>>::from(ind_row);

                if (args.use_float) {
                    auto data = rbp::load_data_cpp<float>(
                        args.phe_file, phe_pos, fix_pos, cov_pos, rand_pos, rand_cov_files, 
                        args.grm_files, args.geno_file, geno_row_selected, args.is_chunk // 强制 false
                    );
                    
                    auto res = rbp::batch_model_cpp<float>(
                        data.y, data.X, data.Z, data.K, args.geno_file, ind_row, ind_col,
                        args.test_method, data.Z_geno, args.do_gwas, args.random_seed,
                        args.num_ranvec, args.re_gamma_size, args.ncpus
                    );
                    
                    rbp::write_out_batch<float>(
                        res, data, args.out_prefix, args.geno_file, ind_col, args.do_gwas
                    );
                } else {
                    auto data = rbp::load_data_cpp<double>(
                        args.phe_file, phe_pos, fix_pos, cov_pos, rand_pos, rand_cov_files, 
                        args.grm_files, args.geno_file, geno_row_selected, args.is_chunk // 强制 false
                    );
                    
                    auto res = rbp::batch_model_cpp<double>(
                        data.y, data.X, data.Z, data.K, args.geno_file, ind_row, ind_col,
                        args.test_method, data.Z_geno, args.do_gwas, args.random_seed,
                        args.num_ranvec, args.re_gamma_size, args.ncpus
                    );
                    
                    rbp::write_out_batch<double>(
                        res, data, args.out_prefix, args.geno_file, ind_col, args.do_gwas
                    );
                }
                break;
            }
            default: break;
        }
        
        auto end_system_time = std::chrono::system_clock::now();
        std::time_t end_time_c = std::chrono::system_clock::to_time_t(end_system_time);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time_steady).count();
        
        std::cout << "\n";
        std::cout << "  ╭────────────────────────────────────────────────────────────────╮\n";
        std::cout << "  │                                                                │\n";
        std::cout << "  │     ^,^        [ ALL TASKS COMPLETED ]                         │\n";
        std::cout << "  │    (o,o)       Finish Time:   " << std::put_time(std::localtime(&end_time_c), "%Y-%m-%d %H:%M:%S") << "              │\n";
        std::cout << "  │    (   )       Total Runtime: " << std::left << std::setw(33) << rbp::string_time(elapsed) << "│\n";
        std::cout << "  │                                                                │\n";
        std::cout << "  ╰────────────────────────────────────────────────────────────────╯\n";
    } catch (const std::exception& e) {
        std::string err_msg = e.what();
        std::replace(err_msg.begin(), err_msg.end(), '\n', ' '); 
        
        std::cerr << "\n";
        std::cerr << "  ╭────────────────────────────────────────────────────────────────╮\n";
        std::cerr << "  │                                                                │\n";
        std::cerr << "  │      /\\        [ FATAL ERROR ENCOUNTERED ]                     │\n";
        std::cerr << "  │     /  \\                                                       │\n";
        size_t pos = 0;
        bool first_line = true;
        while (pos < err_msg.length() || first_line) {
            std::string chunk = err_msg.substr(pos, 48);
            std::string prefix = first_line ? "  │    /____\\      " : "  │                ";
            std::cerr << prefix << std::left << std::setw(48) << chunk << "│\n";
            pos += 48;
            first_line = false;
        }
        std::cerr << "  │                                                                │\n";
        std::cerr << "  ╰────────────────────────────────────────────────────────────────╯\n";
        
        std::cout.rdbuf(old_cout_buf);
        std::cerr.rdbuf(old_cerr_buf);
        log_file.close();
        return 1;
    }
    
    std::cout.rdbuf(old_cout_buf);
    std::cerr.rdbuf(old_cerr_buf);
    log_file.close();
    
    return 0;
}
