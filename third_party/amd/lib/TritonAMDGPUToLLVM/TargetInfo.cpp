#include "TargetInfo.h"
#include "TritonAMDGPUToLLVM/GCNAsmFormat.h"
#include "TritonAMDGPUToLLVM/TargetUtils.h"
#include "Utility.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using mlir::triton::AMD::DppCtrl;
namespace mlir::triton::AMD {

namespace {
template <typename T>
LLVM::LLVMFuncOp getOrInsertFunction(T &moduleOp, const Location loc,
                                     RewriterBase &rewriter, StringRef name,
                                     LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

// Extend all values to 64-bit per printf call requirements.
Value printfPromoteValue(RewriterBase &rewriter, Value value, bool isSigned) {
  auto *context = rewriter.getContext();
  auto loc = UnknownLoc::get(context);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto type = value.getType();

  if (isa<LLVM::LLVMPointerType>(type)) {
    // The llvm.ptrtoint op requires signless integer types.
    return b.ptrtoint(i64_ty, value);
  }

  assert(type.getIntOrFloatBitWidth() <= 64);

  if (auto floatType = dyn_cast<FloatType>(type)) {
    Value newValue = value;
    if (!floatType.isF64())
      newValue = b.fpext(f64_ty, newValue);
    return b.bitcast(newValue, i64_ty);
  }

  assert(type.isIntOrIndex());
  if (type.getIntOrFloatBitWidth() < 64) {
    if (isSigned) {
      return b.sext(i64_ty, value);
    } else {
      // Signless and unsigned integers are printed using unsigned integer
      // formats.
      return b.zext(i64_ty, value);
    }
  }

  return value;
}
} // namespace

llvm::AMDGPU::GPUKind TargetInfo::getGPUKind() const {
  return llvm::AMDGPU::parseArchAMDGCN(arch);
}

int TargetInfo::getWarpSize() const { return isCDNA(getISAFamily()) ? 64 : 32; }

int TargetInfo::getSharedMemorySize() const {
  int kbytes = getISAFamily() == ISAFamily::CDNA4 ? 160 : 64;
  return kbytes * 1024;
}

bool TargetInfo::supportMaximumMinimum() const {
  return getISAFamily() == ISAFamily::CDNA4;
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  // On AMD hardware we don't have CTA clusters like NVIDIA. So this will always
  // be zero. Whoever calling into this should make sure the whole program does
  // not try to utilize CTA clusters.
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  return rewriter.create<ROCDL::BallotOp>(loc, type, cmp);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  mlir::LLVM::AMD::llStore(rewriter, loc, ptr, val, pred);
}

bool TargetInfo::canUseLDSTransLoad(int bitwidth) const {
  return getISAFamily() == ISAFamily::CDNA4 &&
         llvm::is_contained({16, 8, 4, 6}, bitwidth);
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  if (ctaId.has_value()) {
    llvm::report_fatal_error(
        "AMDGPU does not support cross-CTA shared memory transfers");
  }
  Value falseVal = rewriter.create<LLVM::ConstantOp>(
      loc, elemTy, rewriter.getZeroAttr(elemTy));
  bool addAliasGroup = localLoadOp && isSyncedViaAsyncWait(localLoadOp);
  return mlir::LLVM::AMD::llLoad(rewriter, loc, ptr, elemTy, pred, falseVal,
                                 triton::CacheModifier::NONE, addAliasGroup);
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleXor(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::AMD::shuffleUp(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::AMD::shuffleIdx(loc, rewriter, val, i, getISAFamily());
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  return LLVM::AMD::llGetPid(loc, rewriter, moduleOp, axis);
}

// Cast and sext values into specific-length int to meet the requirements of
// instructions like UpdateDpp or readlane if necessary.
static inline Type castToAndSExtInt(RewriterBase &rewriter, Location loc,
                                    Value &val, Type fromType,
                                    unsigned toBits) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned originalBits = fromType.getIntOrFloatBitWidth();
  Type toType = fromType;

  if (!fromType.isIntOrIndex()) {
    val = b.bitcast(val, int_ty(originalBits));
    toType = int_ty(originalBits);
  }

  if (originalBits < toBits) {
    val = b.sext(int_ty(toBits), val);
    toType = int_ty(toBits);
  }

  return toType;
}

// Trunc the value to specific length and then cast it to given type if
// necessary. This function is typically used in conjunction with
// castToAndSExtInt.
static inline Value truncAndCastFromInt(RewriterBase &rewriter, Location loc,
                                        Value val, Type valType,
                                        unsigned fromBits) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned originalBits = valType.getIntOrFloatBitWidth();
  Value toVal = val;

  if (originalBits < fromBits) {
    toVal = b.trunc(int_ty(originalBits), toVal);
  }

  if (!valType.isIntOrIndex()) {
    toVal = b.bitcast(toVal, valType);
  }

  return toVal;
}

// Permute lanes of the input val and apply reduction to permuted values.
static Value permuteAndReduce(RewriterBase &rewriter, Location loc,
                              StringRef intrinsic, Value val,
                              Operation *reduxOp) {
  Type valType = val.getType();
  assert(valType.getIntOrFloatBitWidth() <= 32);

  Type actualType = valType;
  if (!valType.isInteger(32))
    actualType = castToAndSExtInt(rewriter, loc, val, valType, 32);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value falseVal = b.false_val();
  MLIRContext *ctx = rewriter.getContext();
  Type retType = struct_ty({i32_ty, i32_ty});
  Value perm =
      LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, retType,
                                      ValueRange{val, val, falseVal, falseVal})
          ->getResult(0);
  Value v0 = b.extract_val(i32_ty, perm, 0);
  Value v1 = b.extract_val(i32_ty, perm, 1);

  if (!valType.isInteger(32)) {
    v0 = truncAndCastFromInt(rewriter, loc, v0, valType, 32);
    v1 = truncAndCastFromInt(rewriter, loc, v1, valType, 32);
  }
  IRMapping mapping;
  mapping.map(reduxOp->getOperand(0), v0);
  mapping.map(reduxOp->getOperand(1), v1);
  Value redx = rewriter.clone(*reduxOp, mapping)->getResult(0);
  return redx;
}

// Apply warp reduction across lanes using llvm intrinsics in GFX950.
// The input acc has the partial accumulated values from reduction within
// threads. The output acc has the final accumulated values.
//
// Two special cases are supported:
// When numLaneToReduce == 2 && interleave == 32:
//   step 1: use permlane32_swap() to swap the row 2 and 3 of acc and
//           the row 0 and 1 of the copy of acc
//   step 2: apply reduction to the result values to get final result
// When numLaneToReduce == 4 && interleave == 16:
//   step 1: use permlane32_swap() to swap the row 2 and 3 of acc and
//           the row 0 and 1 of the copy of acc
//   step 2: apply reduction to the result values to get the partial result
//   step 3: use permlane16_swap() to swap the odd and even rows of
//           the partial results
//   step 4: apply reduction to get the final results
static bool warpReduceSwap16or32(RewriterBase &rewriter, Location loc,
                                 SmallVector<Value> &acc, triton::ReduceOp op,
                                 unsigned numLaneToReduce,
                                 unsigned interleave) {
  Operation *reduxOp = op.getSingleCombiner();
  if (!reduxOp)
    return false;

  bool mfma32Case = numLaneToReduce == 2 && interleave == 32;
  bool mfma16Case = numLaneToReduce == 4 && interleave == 16;
  if (!(mfma32Case || mfma16Case))
    return false;

  Value val = acc[0];
  unsigned bits = val.getType().getIntOrFloatBitWidth();
  if (bits > 32)
    return false;

  StringRef intrinsic = "llvm.amdgcn.permlane32.swap";
  for (auto i = 0; i < acc.size(); i++) {
    Value redx = permuteAndReduce(rewriter, loc, intrinsic, acc[i], reduxOp);

    if (mfma16Case) {
      intrinsic = "llvm.amdgcn.permlane16.swap";
      redx = permuteAndReduce(rewriter, loc, intrinsic, redx, reduxOp);
    }

    acc[i] = redx;
  }
  return true;
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  if (getISAFamily() == ISAFamily::CDNA4 &&
      warpReduceSwap16or32(rewriter, loc, acc, op, numLaneToReduce, interleave))
    return true;
  if (numLaneToReduce != getWarpSize())
    return false;
  if (isCDNA(getISAFamily()) && getISAFamily() == ISAFamily::CDNA1)
    return false;
  if (isRDNA(getISAFamily()) && getISAFamily() != ISAFamily::RDNA3)
    return false;

  Operation *reduxOp = op.getSingleCombiner();
  if (!reduxOp)
    return false;

  auto createDppReduxOpWithBoundCtrl = [&](Type valType, Value &src,
                                           uint32_t dppCtrl, int rowMask,
                                           int bankMask) -> Value {
    // DPP has limited support for data types, so here we need to
    // cast non-integer types or integer types shorter than 32 bits
    // to int32, except for fp32.
    Type actualType = valType;
    if (!valType.isF32()) {
      actualType = castToAndSExtInt(rewriter, loc, src, valType, 32);
    }

    Value dppResult =
        rewriter
            .create<ROCDL::DPPUpdateOp>(loc, actualType, src, src,
                                        rewriter.getI32IntegerAttr(dppCtrl),
                                        rewriter.getI32IntegerAttr(rowMask),
                                        rewriter.getI32IntegerAttr(bankMask),
                                        rewriter.getBoolAttr(true))
            .getRes();

    if (!valType.isF32()) {
      src = truncAndCastFromInt(rewriter, loc, src, valType, 32);
      dppResult = truncAndCastFromInt(rewriter, loc, dppResult, valType, 32);
    }

    IRMapping mapping;
    mapping.map(reduxOp->getOperand(0), src);
    mapping.map(reduxOp->getOperand(1), dppResult);
    return rewriter.clone(*reduxOp, mapping)->getResult(0);
  };

  for (int i = 0; i < acc.size(); i++) {
    Value buf;
    auto valType = acc[i].getType();

    // Here's the implementation of full-wavefront reduction using dpp.
    // https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/
    //
    // Each step has a v_mov_dpp instruction following the redux op. In
    // some cases, the lower-level compiler could merge them into single
    // instruction. For example, v_mov_dpp + max => v_max_dpp.
    //
    // For gfx9, we have 64 threads per warp. These 64 threads are arranged
    // into 4 rows, with each row being 16 threads. Each 16 threads are arranged
    // further into 4 banks, with each bank being 4 threads. Overall it's in a
    // (row, bank, thread) structure. When shuffling, we use row/bank mask to
    // indicate which row/bank to participate. Then modifier like row_shr and
    // row_bcast means exact data movement schemes. In the following
    // instructions, taking row 0 as an example:
    //
    // Step 1: Right shift for 8 lanes.
    //     lane 8-15 = redux(lane 0-7, lane 8-15)
    //
    // Step 2: Right shift for 4 lanes.
    //     lane 12-15 = redux(lane 8-11, lane 12-15)
    //
    // Step 3: Right shift for 2 lanes.
    //     lane 14-15 = redux(lane 12-13, lane 14-15)
    //
    // Step 4: Right shift for 1 lane.
    //     lane 15 = redux(lane 14, lane 15)
    //
    // Step 5: Broadcast lane 15 of each row to all the lanes of its next row.
    //     lane 16-31 = redux(lane 15, lane 16-31)
    //
    // Step 6: Broadcast lane 31 to lane 32-63.
    //     lane 32-63 = redux(lane 31, lane 32-63)
    //
    // Now the reduction result is stored in lane 63.
    //
    // Step 7: Read the reduction result from lane 63 and broadcast with
    // readlane.

    const int allRows = 0xf;
    const int allBanks = 0xf;

    const uint32_t dppCtrlRowShr = static_cast<uint32_t>(DppCtrl::ROW_SHR0);

    // row_shr:8
    buf = createDppReduxOpWithBoundCtrl(valType, acc[i], 8 + dppCtrlRowShr,
                                        allRows, allBanks);

    // row_shr:4
    buf = createDppReduxOpWithBoundCtrl(valType, buf, 4 + dppCtrlRowShr,
                                        allRows, allBanks);

    // row_shr:2
    buf = createDppReduxOpWithBoundCtrl(valType, buf, 2 + dppCtrlRowShr,
                                        allRows, allBanks);

    // row_shr:1
    buf = createDppReduxOpWithBoundCtrl(valType, buf, 1 + dppCtrlRowShr,
                                        allRows, allBanks);

    if (isCDNA(getISAFamily())) {
      // row_bcast:15 row_mask:0xa
      buf = createDppReduxOpWithBoundCtrl(
          valType, buf, static_cast<uint32_t>(DppCtrl::BCAST15), 0xa, allBanks);

      // row_bcast:31
      buf = createDppReduxOpWithBoundCtrl(
          valType, buf, static_cast<uint32_t>(DppCtrl::BCAST31), allRows,
          allBanks);
    } else {
      // RDNA doesn't have broadcast dpp mode
      Type actualType = castToAndSExtInt(rewriter, loc, buf, valType, 32);

      // Lanes 0-15 read from lane 31 and lanes 16-31 read from lane 15.
      Value permlaneResult = rewriter
                                 .create<ROCDL::PermlaneX16Op>(
                                     loc, actualType, buf, buf, b.i32_val(-1),
                                     b.i32_val(-1), true, false)
                                 .getRes();
      buf = truncAndCastFromInt(rewriter, loc, buf, valType, 32);
      permlaneResult =
          truncAndCastFromInt(rewriter, loc, permlaneResult, valType, 32);
      IRMapping mapping;
      mapping.map(reduxOp->getOperand(0), buf);
      mapping.map(reduxOp->getOperand(1), permlaneResult);
      buf = rewriter.clone(*reduxOp, mapping)->getResult(0);
    }

    // Similarly, we need to cast data types for readlane instruction.
    Type actualType = castToAndSExtInt(rewriter, loc, buf, valType, 16);

    // Get reduction result from the last lane of the warp
    Value lastLaneId = b.i32_val(gpu::lookupThreadsPerWarp(rewriter) - 1);
    Value result =
        rewriter.create<ROCDL::ReadlaneOp>(loc, actualType, buf, lastLaneId);

    result = truncAndCastFromInt(rewriter, loc, result, valType, 16);

    acc[i] = result;
  }

  return true;
}

void TargetInfo::printfImpl(Value formatStrStart, int formatStrByteCount,
                            ValueRange args, ArrayRef<bool> isSigned,
                            RewriterBase &rewriter, bool useStdErr) const {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto *ctx = rewriter.getContext();
  mlir::Location loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  // See
  // https://github.com/ROCm/ROCm-Device-Libs/blob/rocm-6.0.x/ockl/src/services.cl#L263-L361
  // for details about the following HIP device print functions.
  LLVM::LLVMFuncOp printBeginFn = getOrInsertFunction(
      moduleOp, loc, rewriter,
      useStdErr ? "__ockl_fprintf_stderr_begin" : "__ockl_printf_begin",
      LLVM::LLVMFunctionType::get(i64_ty,
                                  useStdErr ? ArrayRef<Type>() : i64_ty));
  LLVM::LLVMFuncOp printStrFn = getOrInsertFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          i64_ty, {i64_ty, ptr_ty(ctx), /*length=*/i64_ty, /*isLast=*/i32_ty}));
  LLVM::LLVMFuncOp printArgsFn;
  if (!args.empty()) {
    printArgsFn = getOrInsertFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            i64_ty, {i64_ty, /*numArgs=*/i32_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                     i64_ty, i64_ty, i64_ty, /*isLast=*/i32_ty}));
  }

  // Emit the intrinsic function call to begin the printf.
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64_ty, 0);
  Value message =
      b.call(printBeginFn, useStdErr ? ValueRange() : zeroI64).getResult();

  // Emit the intrinsic function call to handle the printf format string.
  Value oneI32 = b.i32_val(1);
  Value zeroI32 = b.i32_val(0);
  Value formatStrLen =
      rewriter.create<LLVM::ConstantOp>(loc, i64_ty, formatStrByteCount);
  SmallVector<Value, 4> arguments = {message, formatStrStart, formatStrLen,
                                     args.empty() ? oneI32 : zeroI32};
  message = b.call(printStrFn, arguments).getResult();

  // Emit the intrinsic function call to handle arguments iteratively.
  // We can only handle at most 7 values each time.
  constexpr size_t kArgsPerGroup = 7;
  for (size_t group = 0; group < args.size(); group += kArgsPerGroup) {
    size_t bound = std::min(group + kArgsPerGroup, args.size());
    size_t numArgs = bound - group;

    SmallVector<Value, 2 + kArgsPerGroup + 1> arguments;
    arguments.push_back(message);
    arguments.push_back(b.i32_val(numArgs));
    for (size_t i = group; i < bound; ++i) {
      arguments.push_back(printfPromoteValue(
          rewriter, args[i], isSigned.empty() ? true : isSigned[i]));
    }
    // Pad out to 7 arguments since the function always needs 7 args.
    for (size_t extra = numArgs; extra < kArgsPerGroup; ++extra) {
      arguments.push_back(zeroI64);
    }

    Value isLast = (bound == args.size()) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    message = b.call(printArgsFn, arguments).getResult();
  }
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__ockl_mul_hi_u32" : "__ockl_mul_hi_u64";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int formatStrByteCount, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  return printfImpl(formatStrStart, formatStrByteCount, args, isSigned,
                    rewriter,
                    /*useStdError=*/false);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue =
      LLVM::addStringToModule(UnknownLoc::get(rewriter.getContext()), rewriter,
                              "printfFormat_", msgNewline);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Compose and print an assert message.
  llvm::SmallString<256> msgBuffer;
  llvm::Twine("device assertion failed: '" + message + "', in " + func +
              " at " + file + ":" + llvm::Twine(line) + "\n\0")
      .toStringRef(msgBuffer);
  Value msgValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", msgBuffer);
  printfImpl(msgValue, msgBuffer.size_in_bytes(), /*args=*/ValueRange(),
             /*isSigned=*/{}, rewriter, /*useStdError=*/true);

  // Set block barrier before aborting kernel, give a chance for all
  // the threads in a block to check/print the assert failure.
  b.barrier();
  // Perform the trap to abort the kernel.
  rewriter.create<LLVM::Trap>(loc);
}

int TargetInfo::getSharedAddressSpace() const { return 3; }

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace for now");
  }
  return spaceId;
}

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used, but AMD generally supports vectorized
  // atomics.
  return true;
}

bool TargetInfo::supportsDirectToLdsLoadBitWidth(int bitWidth) const {
  switch (getISAFamily()) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
    // Disable 8 and 16 bits because they get extended to 32 bit.
    return llvm::is_contained({32, /*16, 8*/}, bitWidth);
  case ISAFamily::CDNA4:
    // Disable 8, 16, 96 bits because they get extended to 32/128 bit.
    return llvm::is_contained({128, /*96, */ 32, /*16, 8*/}, bitWidth);
  default:
    break;
  }

  return false;
}

void TargetInfo::localLoadOpAnnotation(triton::gpu::LocalLoadOp localLoadOp,
                                       Operation *llLoadOp) const {
  AMD::addLocalLoadNoAliasScope(localLoadOp, cast<LLVM::LoadOp>(llLoadOp));
}

} // namespace mlir::triton::AMD
