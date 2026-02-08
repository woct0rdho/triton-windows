#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUTESTPIPELINELOWERLOOP
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct TestPipelineLowerLoop
    : public impl::TritonGPUTestPipelineLowerLoopBase<TestPipelineLowerLoop> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPipelineLowerLoop)
  using impl::TritonGPUTestPipelineLowerLoopBase<
      TestPipelineLowerLoop>::TritonGPUTestPipelineLowerLoopBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    lowerLoops(m);
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
