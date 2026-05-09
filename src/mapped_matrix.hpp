#ifndef MAPPED_MATRIX_HPP
#define MAPPED_MATRIX_HPP

#include <armadillo>
#include <string>
#include <type_traits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdexcept>
#include <cstring>
#include <cerrno>

namespace rbp {

/**
 * 极致零拷贝内存映射矩阵类 (并发安全版)
 * 支持多线程下同时读取、修改并安全落盘多个不同分块。
 * @tparam T 数据类型，仅支持 float 或 double
 */
template<typename T = double>
class MappedMatrix {
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                  "MappedMatrix only supports float and double types");
    
private:
    T* data_;                // 全局数据指针（仅在完整映射模式下有效）
    size_t nrow_;            
    size_t ncol_;            
    std::string filename_;   
    int fd_;                 
    bool is_chunk_;          
    bool col_major_;         

public:
    // 新增：默认构造函数
    MappedMatrix() : data_(nullptr), nrow_(0), ncol_(0), fd_(-1), is_chunk_(false), col_major_(true) {}

    // 新增：移动构造函数
    MappedMatrix(MappedMatrix&& other) noexcept 
        : data_(other.data_), nrow_(other.nrow_), ncol_(other.ncol_), 
          filename_(std::move(other.filename_)), fd_(other.fd_), 
          is_chunk_(other.is_chunk_), col_major_(other.col_major_) {
        other.data_ = nullptr;
        other.fd_ = -1;
    }

    // 新增：移动赋值运算符
    MappedMatrix& operator=(MappedMatrix&& other) noexcept {
        if (this != &other) {
            if (!is_chunk_ && data_ && data_ != MAP_FAILED) munmap(data_, nrow_ * ncol_ * sizeof(T));
            if (fd_ != -1) close(fd_);
            data_ = other.data_; nrow_ = other.nrow_; ncol_ = other.ncol_;
            filename_ = std::move(other.filename_); fd_ = other.fd_;
            is_chunk_ = other.is_chunk_; col_major_ = other.col_major_;
            other.data_ = nullptr; other.fd_ = -1;
        }
        return *this;
    }
    // ========================================================================
    // 核心并发组件：RAII 分块视图代理
    // 每个 ChunkView 拥有独立的生命周期和 mmap 窗口，完美支持 OpenMP
    // ========================================================================
    class ChunkView {
    private:
        void* map_ptr_;
        size_t map_len_;
        arma::Mat<T> mat_;

    public:
        // 构造函数：接管底层 mmap 指针并绑定 Arma 矩阵
        ChunkView(void* map_ptr, size_t map_len, T* data_ptr, size_t rows, size_t cols)
            : map_ptr_(map_ptr), map_len_(map_len), mat_(data_ptr, rows, cols, false, true) {}

        // 析构函数：生命周期结束时自动安全释放局部内存映射
        ~ChunkView() {
            if (map_ptr_ && map_ptr_ != MAP_FAILED) {
                munmap(map_ptr_, map_len_);
            }
        }

        // 禁用拷贝语义（防止指针被重复 munmap）
        ChunkView(const ChunkView&) = delete;
        ChunkView& operator=(const ChunkView&) = delete;

        // 启用移动语义（允许从函数返回）
        ChunkView(ChunkView&& other) noexcept 
            : map_ptr_(other.map_ptr_), map_len_(other.map_len_), mat_(std::move(other.mat_)) {
            other.map_ptr_ = nullptr;
            other.map_len_ = 0;
        }

        // 暴露底层的 Armadillo 矩阵供数学运算
        arma::Mat<T>& mat() { return mat_; }
        const arma::Mat<T>& mat() const { return mat_; }

        // 原地安全落盘同步
        void sync_chunk() {
            if (map_ptr_ && map_ptr_ != MAP_FAILED) {
                if (msync(map_ptr_, map_len_, MS_SYNC) == -1) {
                    int saved_errno = errno;
                    throw std::runtime_error(std::string("Chunk sync failed: ") + std::strerror(saved_errno));
                }
            }
        }
    };
    // ========================================================================

    MappedMatrix(const std::string& filename, size_t nrow, size_t ncol, 
                 bool create = false, bool isChunk = false, bool col_major = true)
        : data_(nullptr), nrow_(nrow), ncol_(ncol), filename_(filename), 
          fd_(-1), is_chunk_(isChunk), col_major_(col_major) {
        
        size_t file_size = nrow_ * ncol_ * sizeof(T);
        
        if (create) {
            fd_ = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
            if (fd_ == -1) throw std::runtime_error("Failed to create file: " + filename);
            if (ftruncate(fd_, static_cast<off_t>(file_size)) == -1) {
                int saved_errno = errno; close(fd_); unlink(filename.c_str());
                throw std::runtime_error(std::string("Failed to set file size: ") + std::strerror(saved_errno));
            }
        } else {
            fd_ = open(filename.c_str(), O_RDWR);
            if (fd_ == -1) throw std::runtime_error("Failed to open file: " + filename);
            struct stat st;
            if (fstat(fd_, &st) == -1) {
                int saved_errno = errno; close(fd_);
                throw std::runtime_error(std::string("Failed to get file size: ") + std::strerror(saved_errno));
            }
        }

        if (!is_chunk_) {
            void* mapped = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (mapped == MAP_FAILED) {
                int saved_errno = errno; close(fd_);
                throw std::runtime_error(std::string("Memory mapping failed: ") + std::strerror(saved_errno));
            }
            data_ = static_cast<T*>(mapped);
            madvise(data_, file_size, MADV_RANDOM);
        }
    }
    
    ~MappedMatrix() {
        if (!is_chunk_ && data_ && data_ != MAP_FAILED) {
            munmap(data_, nrow_ * ncol_ * sizeof(T));
        }
        if (fd_ != -1) close(fd_);
    }
    
    MappedMatrix(const MappedMatrix&) = delete;
    MappedMatrix& operator=(const MappedMatrix&) = delete;
    
    size_t nrow() const { return nrow_; }
    size_t ncol() const { return ncol_; }
    std::string filename() const { return filename_; }
    bool isChunkedMode() const { return is_chunk_; }
    bool isColMajor() const { return col_major_; }

    /**
     * 转换为 Armadillo 矩阵
     * 注意：分块模式下无效
     */
    arma::Mat<T> to_arma_mat() const {
        if (is_chunk_) {
            throw std::runtime_error("to_arma_mat() not work in chunked mode");
            
        } else {
            return arma::Mat<T>(data_, nrow_, ncol_, false, true);
        }
    }

    /**
     * 线程安全的获取局部视图。
     * 返回一个包含私有 mmap 生命周期的 ChunkView 代理对象。
     */
    ChunkView get_chunk(size_t start_col = 0, size_t end_col = static_cast<size_t>(-1)) const {
        if (!col_major_) throw std::runtime_error("Mapping is only supported for column-major storage.");
        
        if (end_col == static_cast<size_t>(-1)) end_col = ncol_ - 1;
        if (start_col > end_col || end_col >= ncol_) {
            throw std::out_of_range("Column range out of bounds in get_chunk.");
        }

        size_t num_cols = end_col - start_col + 1;

        if (!is_chunk_) {
            // 完整映射模式下：通过虚假的 map_ptr (nullptr) 返回视图，不触发子区域的 munmap
            return ChunkView(nullptr, 0, data_ + start_col * nrow_, nrow_, num_cols);
        } else {
            off_t start_offset = static_cast<off_t>(start_col) * static_cast<off_t>(nrow_) * sizeof(T);
            size_t bytes_to_read = num_cols * nrow_ * sizeof(T);
            off_t end_offset = start_offset + bytes_to_read;
            
            long page_size = sysconf(_SC_PAGESIZE);
            off_t aligned_offset = (start_offset / page_size) * page_size;
            size_t map_len = end_offset - aligned_offset;

            // 线程独立的临时映射
            void* chunk_map = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, aligned_offset);
            if (chunk_map == MAP_FAILED) {
                int saved_errno = errno;
                throw std::runtime_error(std::string("Chunk mmap failed: ") + std::strerror(saved_errno));
            }
            // 仅对当前线程的这块内存声明 SEQUENTIAL 访问
            madvise(chunk_map, map_len, MADV_SEQUENTIAL);

            T* chunk_data = reinterpret_cast<T*>(
                static_cast<char*>(chunk_map) + (start_offset - aligned_offset)
            );
            
            // 将映射状态打包移交给 ChunkView 对象
            return ChunkView(chunk_map, map_len, chunk_data, nrow_, num_cols);
        }
    }

    /**
     * 全局完整模式同步
     */
    void sync() {
        if (!is_chunk_ && data_ && data_ != MAP_FAILED) {
            if (msync(data_, nrow_ * ncol_ * sizeof(T), MS_SYNC) == -1) {
                int saved_errno = errno;
                throw std::runtime_error(std::string("Failed to sync full matrix: ") + std::strerror(saved_errno));
            }
        }
    }
};

} // namespace rbp

#endif // MAPPED_MATRIX_HPP