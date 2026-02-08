#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;
namespace gluon = mlir::triton::gluon;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONINLINE
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {
struct Inline : public gluon::impl::GluonInlineBase<Inline> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Inline)
  void runOnOperation() override;
};
} // namespace

void Inline::runOnOperation() {
  mlir::PassManager pm(&getContext());
  pm.addPass(createInlinerPass(/*opPipelines=*/{}, [](OpPassManager &pm) {
    pm.addPass(gluon::createGluonSimplifyControlFlow());
  }));
  if (failed(pm.run(getOperation())))
    return signalPassFailure();
}
