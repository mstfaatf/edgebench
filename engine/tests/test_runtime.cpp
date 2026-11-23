#include "test_runtime.hpp"

#include <memory>
#include <stdexcept>

namespace edgebench::test {

Runtime& shared_runtime() {
    static std::unique_ptr<Runtime> runtime = [] {
        const OrtLoadResult load = load_onnxruntime();
        if (!load.ok())
            throw std::runtime_error(load.message);
        Result<std::unique_ptr<Runtime>> created = Runtime::create(load);
        if (!created)
            throw std::runtime_error(created.error().message);
        return std::move(created).value();
    }();
    return *runtime;
}

}  // namespace edgebench::test
