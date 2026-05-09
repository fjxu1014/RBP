#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>

namespace rbp {

/**
 * 进度条类
 * 用于在控制台显示任务执行进度
 */
class ProgressBar {
private:
    unsigned int ticks_ = 0;
    const unsigned int total_ticks_;
    const unsigned int bar_width_;
    const std::chrono::steady_clock::time_point start_time_;

    std::string time_to_string(double seconds, float progress) const {
        double timed = progress < 1.0 ? (seconds * (1.0 - progress) / (progress + 1e-10)) : seconds;
        int time = static_cast<int>(timed);
        int hour = time / 3600;
        time %= 3600;
        int min = time / 60;
        int sec = time % 60;

        std::ostringstream oss;
        oss << (progress < 1.0 ? "TimeLeft:" : "RunTime:");
        if (hour != 0) oss << hour << "h";
        if (hour != 0 || min != 0) oss << min << "m";
        oss << sec << "s";
        return oss.str();
    }

public:
    /**
     * 构造函数
     * @param total: 总步数
     * @param width: 进度条宽度（字符数）
     */
    ProgressBar(unsigned int total, unsigned int width) 
        : total_ticks_(total), bar_width_(width), 
          start_time_(std::chrono::steady_clock::now()) {}

    /**
     * 递增进度
     * @return: 当前进度值
     */
    unsigned int operator++() { return ++ticks_; }

    /**
     * 显示进度条
     */
    void display() const {
        float progress = std::min(1.0f, static_cast<float>(ticks_) / total_ticks_);
        unsigned int pos = static_cast<unsigned int>(bar_width_ * progress);

        auto now = std::chrono::steady_clock::now();
        auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time_).count();

        std::ostringstream oss;
        oss << "[>>>";
        for (unsigned int i = 0; i < bar_width_; ++i) {
            if (i < pos) oss << "=";
            else if (i == pos) oss << ">";
            else oss << "-";
        }
        oss << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "%  "
            << time_to_string(time_elapsed / 1000.0, progress);

        std::cout << "\r" << std::setw(60 + bar_width_) << std::left << oss.str() << std::flush;
    }

    /**
     * 完成进度条显示
     */
    void done() {
        ticks_ = total_ticks_; // 确保进度条显示为 100%
        display();
        std::cout << std::endl;
    }
};

} // namespace rbp

#endif // PROGRESS_BAR_HPP
