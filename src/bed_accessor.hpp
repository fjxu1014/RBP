#ifndef BED_ACCESSOR_HPP
#define BED_ACCESSOR_HPP

#include <armadillo>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <algorithm>
#include <omp.h>

namespace rbp {

    size_t get_file_lines(const std::string& filename);

/**
 * PLINK BED文件读取器 (0拷贝按需映射 + 极限性能优化版)
 * @tparam T 数据类型，仅支持 float 或 double
 */
template<typename T = double>
class BedAccessor {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "BedAccessor only supports float and double types");

private:
    int fd_;
    size_t file_size_;
    bool full_mapping_;            

    unsigned char* mapped_data_;
    unsigned char* bed_data_;      

    size_t n_samples_total_;
    size_t n_variants_total_;
    size_t n_bytes_per_variant_;
    
    arma::Mat<T> lookup_table_;
    
    arma::uvec ind_row_;
    arma::uvec ind_col_;
    size_t n_ind_;

    static arma::Mat<T> create_lookup_table(int na_val = 3, int coding_scheme = 0) {
        arma::Col<T> num(4);
        if (coding_scheme == 1) {
            num = {static_cast<T>(0.0), static_cast<T>(na_val), static_cast<T>(1.0), static_cast<T>(0.0)};
        } else {
            num = {static_cast<T>(2.0), static_cast<T>(na_val), static_cast<T>(1.0), static_cast<T>(0.0)};
        }
        arma::Mat<T> code(4, 256);
        int coeff = 1;
        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 256; k++) {
                code(i, k) = num[(k / coeff) % 4];
            }
            coeff *= 4;
        }
        return code;
    }

    static std::string ensure_bed_extension(const std::string& filepath) {
        if (filepath.size() >= 4 && filepath.substr(filepath.size() - 4) == ".bed") return filepath;
        return filepath + ".bed";
    }

public:
    BedAccessor(const std::string& bed_file, 
                const arma::uvec& ind_row, 
                const arma::uvec& ind_col,
                bool full_mapping = false,
                int na_val = 3,
                int coding_scheme = 0)
        : fd_(-1), full_mapping_(full_mapping), mapped_data_(reinterpret_cast<unsigned char*>(MAP_FAILED)), 
          bed_data_(nullptr), ind_row_(ind_row), ind_col_(ind_col) {
        
        std::string bed_path = ensure_bed_extension(bed_file);
        std::string base_path = bed_path.substr(0, bed_path.size() - 4);
        
        n_samples_total_ = get_file_lines(base_path + ".fam");
        n_variants_total_ = get_file_lines(base_path + ".bim");
        n_bytes_per_variant_ = (n_samples_total_ + 3) / 4;
        n_ind_ = ind_row_.n_elem;
        
        lookup_table_ = create_lookup_table(na_val, coding_scheme);
        
        fd_ = open(bed_path.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("Cannot open BED file: " + bed_path);

        struct stat st;
        if (fstat(fd_, &st) == -1) { close(fd_); throw std::runtime_error("Failed to stat BED"); }
        file_size_ = st.st_size;

        unsigned char magic[3];
        if (pread(fd_, magic, 3, 0) != 3) { close(fd_); throw std::runtime_error("Failed to read magic"); }
        if (magic[0] != 0x6C || magic[1] != 0x1B || magic[2] != 0x01) {
            close(fd_); throw std::runtime_error("Invalid BED format or not SNP-major");
        }

        size_t expected_size = 3 + n_bytes_per_variant_ * n_variants_total_;
        if (file_size_ != expected_size) {
            close(fd_);
            throw std::runtime_error("BED file size mismatch.");
        }
        
        if (full_mapping_) {
            mapped_data_ = static_cast<unsigned char*>(mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0));
            if (mapped_data_ == MAP_FAILED) { close(fd_); throw std::runtime_error("mmap failed"); }
            bed_data_ = mapped_data_ + 3;
            madvise(mapped_data_, file_size_, MADV_RANDOM);
        }
    }
    
    ~BedAccessor() {
        if (full_mapping_ && mapped_data_ != MAP_FAILED) munmap(mapped_data_, file_size_);
        if (fd_ != -1) close(fd_);
    }
    
    BedAccessor(const BedAccessor&) = delete;
    BedAccessor& operator=(const BedAccessor&) = delete;
    
    size_t nrow() const { return n_ind_; }
    size_t ncol() const { return ind_col_.n_elem; }
    bool isFullMapping() const { return full_mapping_; }

    inline T operator()(size_t i, size_t j) const {
        if (!full_mapping_) throw std::runtime_error("operator() is only available in full mapping mode.");
        size_t sample_idx = ind_row_[i];
        size_t variant_idx = ind_col_[j];
        unsigned char byte = bed_data_[variant_idx * n_bytes_per_variant_ + sample_idx / 4];
        
        return lookup_table_(sample_idx % 4, byte);
    }

    void cols(size_t start, size_t end, T* out_ptr) const {
        if (start > end || end >= ind_col_.n_elem) {
            throw std::out_of_range("Invalid column range requested.");
        }
        
        arma::Mat<T> out(out_ptr, n_ind_, (end - start + 1), false, true);
        
        if (full_mapping_) {
            for (size_t j = start; j <= end; j++) {
                size_t col_idx = j - start;
                size_t variant_idx = ind_col_[j];
                size_t col_byte_start = variant_idx * n_bytes_per_variant_;
                
                for (size_t i = 0; i < n_ind_; i++) {
                    size_t sample_idx = ind_row_[i];
                    unsigned char byte = bed_data_[col_byte_start + sample_idx / 4];
                    out(i, col_idx) = lookup_table_(sample_idx % 4, byte);
                }
            }
            
            size_t start_byte = ind_col_[start] * n_bytes_per_variant_;
            size_t end_byte = ind_col_[end] * n_bytes_per_variant_ + n_bytes_per_variant_;
            long page_size = sysconf(_SC_PAGESIZE);
            size_t page_mask = ~(page_size - 1);
            size_t aligned_start = (reinterpret_cast<size_t>(bed_data_) + start_byte) & page_mask;
            size_t aligned_end = (reinterpret_cast<size_t>(bed_data_) + end_byte + page_size - 1) & page_mask;
            if (aligned_end > aligned_start) {
                madvise(reinterpret_cast<void*>(aligned_start), aligned_end - aligned_start, MADV_DONTNEED);
            }
        } else {
            size_t min_idx = ind_col_[start];
            size_t max_idx = ind_col_[start];
            for (size_t j = start + 1; j <= end; ++j) {
                if (ind_col_[j] < min_idx) min_idx = ind_col_[j];
                if (ind_col_[j] > max_idx) max_idx = ind_col_[j];
            }
            
            off_t start_offset = 3 + static_cast<off_t>(min_idx) * n_bytes_per_variant_;
            off_t end_offset = 3 + static_cast<off_t>(max_idx + 1) * n_bytes_per_variant_;
            
            long page_size = sysconf(_SC_PAGESIZE);
            off_t aligned_offset = (start_offset / page_size) * page_size;
            size_t map_len = end_offset - aligned_offset;
            
            unsigned char* chunk_map = static_cast<unsigned char*>(
                mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd_, aligned_offset)
            );
            
            if (chunk_map == MAP_FAILED) {
                int saved_errno = errno;
                throw std::runtime_error(std::string("Chunk mmap failed: ") + std::strerror(saved_errno));
            }
            madvise(chunk_map, map_len, MADV_SEQUENTIAL);
            
            for (size_t j = start; j <= end; ++j) {
                size_t col_idx = j - start;
                size_t variant_idx = ind_col_[j];
                size_t byte_offset_in_map = 3 + variant_idx * n_bytes_per_variant_ - aligned_offset;
                const unsigned char* col_bytes = chunk_map + byte_offset_in_map;
                
                for (size_t i = 0; i < n_ind_; i++) {
                    size_t sample_idx = ind_row_[i];
                    unsigned char byte = col_bytes[sample_idx / 4];
                    out(i, col_idx) = lookup_table_(sample_idx % 4, byte);
                }
            }
            munmap(chunk_map, map_len);
        }
    }

    /**
     * 计算所有选定 SNP 的均值
     * 极致性能优化版：OpenMP 多线程 + 字节级查表 (Byte-level LUT)
     */
    arma::Row<T> calculate_snp_stats(int ncpus = 1) const {
        size_t num_snps = ind_col_.n_elem;
        arma::Row<T> out_means(num_snps);
        
        // ====================================================================
        // 核心优化 1：构建 256 字节状态查表 (Byte-level LUT)
        // 直接预计算 1 个字节 (包含 4 个个体) 的 sum 和 valid_count
        // ====================================================================
        struct ByteStat { int sum; int valid; };
        static bool lut_initialized = false;
        static ByteStat byte_lut[256];
        
        if (!lut_initialized) {
            for (int i = 0; i < 256; ++i) {
                int s = 0, v = 0;
                for (int j = 0; j < 4; ++j) {
                    int g = (i >> (j * 2)) & 0x03;
                    if (g == 0) { s += 2; v += 1; }       // 00 -> 2
                    else if (g == 2) { s += 1; v += 1; }  // 10 -> 1
                    else if (g == 3) { s += 0; v += 1; }  // 11 -> 0
                    // g == 1 (01) 为缺失，s 不变，v 不加
                }
                byte_lut[i] = {s, v};
            }
            lut_initialized = true;
        }

        struct GenoDec { int sum; int valid; };
        GenoDec bit_lut[4] = { {2, 1}, {0, 0}, {1, 1}, {0, 1} };

        // ====================================================================
        // 核心优化 2：探测极速路径
        // 如果没有进行样本 Subset (抽取全部个体)，则开启免索引的 Byte 级狂奔
        // ====================================================================
        bool is_full_samples = (n_ind_ == n_samples_total_);
        size_t full_bytes = n_samples_total_ / 4;
        size_t remainder = n_samples_total_ % 4;

        if (full_mapping_) {
            // 核心优化 3：开启 OpenMP 跨 SNP 多线程并发
            #pragma omp parallel for schedule(static) num_threads(ncpus)
            for (size_t j = 0; j < num_snps; ++j) {
                size_t variant_idx = ind_col_[j];
                const unsigned char* col_bytes = bed_data_ + variant_idx * n_bytes_per_variant_;
                int sum = 0;
                int valid_count = 0;

                if (is_full_samples) {
                    for (size_t b = 0; b < full_bytes; ++b) {
                        ByteStat stat = byte_lut[col_bytes[b]];
                        sum += stat.sum;
                        valid_count += stat.valid;
                    }
                    if (remainder > 0) {
                        unsigned char b = col_bytes[full_bytes];
                        for (size_t k = 0; k < remainder; ++k) {
                            int g = (b >> (k * 2)) & 0x03;
                            sum += bit_lut[g].sum;
                            valid_count += bit_lut[g].valid;
                        }
                    }
                } else {
                    for (size_t i = 0; i < n_ind_; ++i) {
                        size_t sample_idx = ind_row_[i];
                        unsigned char b = col_bytes[sample_idx >> 2];
                        int g = (b >> ((sample_idx & 3) << 1)) & 0x03;
                        sum += bit_lut[g].sum;
                        valid_count += bit_lut[g].valid;
                    }
                }
                
                // 恢复原始计算逻辑，不强制进行频率翻转
                out_means[j] = valid_count > 0 ? static_cast<T>(sum) / valid_count : 0;
            }
        } else {
            // 分块映射模式
            size_t chunk_size = 10000; 
            long page_size = sysconf(_SC_PAGESIZE);
            
            for (size_t start = 0; start < num_snps; start += chunk_size) {
                size_t end = std::min(num_snps - 1, start + chunk_size - 1);
                
                size_t min_idx = ind_col_[start];
                size_t max_idx = ind_col_[start];
                for (size_t j = start + 1; j <= end; ++j) {
                    if (ind_col_[j] < min_idx) min_idx = ind_col_[j];
                    if (ind_col_[j] > max_idx) max_idx = ind_col_[j];
                }
                
                off_t start_offset = 3 + static_cast<off_t>(min_idx) * n_bytes_per_variant_;
                off_t end_offset = 3 + static_cast<off_t>(max_idx + 1) * n_bytes_per_variant_;
                
                off_t aligned_offset = (start_offset / page_size) * page_size;
                size_t map_len = end_offset - aligned_offset;
                
                unsigned char* chunk_map = static_cast<unsigned char*>(
                    mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd_, aligned_offset)
                );
                
                if (chunk_map == MAP_FAILED) {
                    int saved_errno = errno;
                    throw std::runtime_error(std::string("Chunk mmap failed: ") + std::strerror(saved_errno));
                }
                madvise(chunk_map, map_len, MADV_SEQUENTIAL);
                
                #pragma omp parallel for schedule(static) num_threads(ncpus)
                for (size_t j = start; j <= end; ++j) {
                    size_t variant_idx = ind_col_[j];
                    size_t byte_offset_in_map = 3 + variant_idx * n_bytes_per_variant_ - aligned_offset;
                    const unsigned char* col_bytes = chunk_map + byte_offset_in_map;
                    
                    int sum = 0;
                    int valid_count = 0;
                    
                    if (is_full_samples) {
                        for (size_t b = 0; b < full_bytes; ++b) {
                            ByteStat stat = byte_lut[col_bytes[b]];
                            sum += stat.sum;
                            valid_count += stat.valid;
                        }
                        if (remainder > 0) {
                            unsigned char b = col_bytes[full_bytes];
                            for (size_t k = 0; k < remainder; ++k) {
                                int g = (b >> (k * 2)) & 0x03;
                                sum += bit_lut[g].sum;
                                valid_count += bit_lut[g].valid;
                            }
                        }
                    } else {
                        for (size_t i = 0; i < n_ind_; ++i) {
                            size_t sample_idx = ind_row_[i];
                            unsigned char b = col_bytes[sample_idx >> 2];
                            int g = (b >> ((sample_idx & 3) << 1)) & 0x03;
                            sum += bit_lut[g].sum;
                            valid_count += bit_lut[g].valid;
                        }
                    }
                    
                    // 恢复原始计算逻辑，不强制进行频率翻转
                    out_means[j] = valid_count > 0 ? static_cast<T>(sum) / valid_count : 0;
                }
                munmap(chunk_map, map_len);
            }
        }
        return out_means;
    }
};

} // namespace rbp

#endif // BED_ACCESSOR_HPP