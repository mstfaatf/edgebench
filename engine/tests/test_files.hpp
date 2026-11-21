// Test helper: a small file standing in for a model, removed on destruction.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace edgebench::test {

class TempModel {
public:
    explicit TempModel(const std::string& file_name = "model.onnx") {
        static std::atomic<int> counter{0};
        dir_ = std::filesystem::temp_directory_path() /
               ("edgebench_unit_" + std::to_string(counter++) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        std::filesystem::create_directories(dir_);
        path_ = dir_ / file_name;
        std::ofstream(path_, std::ios::binary) << "not a real model";
    }
    ~TempModel() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }
    TempModel(const TempModel&) = delete;
    TempModel& operator=(const TempModel&) = delete;

    const std::filesystem::path& path() const { return path_; }
    std::uintmax_t size() const { return std::filesystem::file_size(path_); }

private:
    std::filesystem::path dir_;
    std::filesystem::path path_;
};

}  // namespace edgebench::test
