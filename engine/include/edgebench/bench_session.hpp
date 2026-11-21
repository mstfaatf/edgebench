// BenchSession: one inference session, ready to run, as created by a Provider.
//
// A session owns its input buffers (generated once, before any run, from the
// run configuration) and, with RunBinding::iobinding_preallocated, its output
// buffers. The timing code only calls run_once() between clock reads, so
// everything run_once() does is part of the measured time.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "edgebench/placement.hpp"
#include "edgebench/result.hpp"
#include "edgebench/run_config.hpp"
#include "edgebench/tensor_info.hpp"

namespace edgebench {

class BenchSession {
public:
    virtual ~BenchSession() = default;

    /// Inputs and outputs as declared by the model; every shape is fixed.
    virtual const std::vector<TensorInfo>& inputs() const = 0;
    virtual const std::vector<TensorInfo>& outputs() const = 0;

    /// Dimension 0 of the first input (1 for a scalar first input).
    virtual std::int64_t batch_size() const = 0;

    virtual RunBinding run_binding() const = 0;

    /// The owned buffer of float32 input `index`. Writing to it changes the data
    /// of the next run (e.g. to feed reference inputs); its address never changes.
    virtual std::span<float> input_data(std::size_t index) = 0;

    /// Runs one inference on the current inputs. With iobinding_preallocated it
    /// allocates no input or output tensors.
    virtual Result<void> run_once() = 0;

    /// The outputs of the latest run_once() (empty views before the first run).
    /// Valid until the next run_once().
    virtual TensorView output(std::size_t index) const = 0;

    /// Node placement, recorded from a separate session with the same options.
    virtual const NodeAssignment& node_assignment() const = 0;

    /// Wall-clock time to create this session, in milliseconds.
    virtual double session_create_ms() const = 0;
};

}  // namespace edgebench
