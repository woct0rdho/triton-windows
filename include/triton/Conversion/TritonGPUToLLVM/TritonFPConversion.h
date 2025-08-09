// This file is adapted for Windows on older hardware
#pragma once

#ifdef _MSC_VER
// Suppress implicit conversion warnings between different bit widths
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif

#include "llvm/Support/Casting.h"               // llvm::isa
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace fp_conv {

// fp32 -> fp16 (simplified)
inline mlir::Value fp32_to_fp16(mlir::Location loc,
                                mlir::OpBuilder &builder,
                                mlir::Value fp32Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto f16Type = mlir::Float16Type::get(ctx);

  auto bits = builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);
  auto shifted = builder.create<mlir::LLVM::LShrOp>(loc, bits, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i16Type, shifted);
  return builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, truncated);
}

// fp16 -> fp32 (simplified)
inline mlir::Value fp16_to_fp32(mlir::Location loc,
                                mlir::OpBuilder &builder,
                                mlir::Value fp16Val) {
  auto *ctx = builder.getContext();
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto f32Type = mlir::Float32Type::get(ctx);

  if (fp16Val.getType().isF16()) {
    fp16Val = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);
  }

  auto ext = builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, fp16Val);
  auto shifted = builder.create<mlir::LLVM::ShlOp>(loc, ext, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, shifted);
}

// fp32 -> fp8 e4m3 (simplified)
inline mlir::Value fp32_to_fp8_e4m3(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    mlir::Value fp32Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto i8Type = mlir::IntegerType::get(ctx, 8);
  auto f16Type = mlir::Float16Type::get(ctx);

  // For older hardware, we'll convert fp32 -> fp16 -> truncated to simulate fp8
  // This is a simplified approach for compatibility
  auto bits = builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);
  auto shifted = builder.create<mlir::LLVM::LShrOp>(loc, bits, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i16Type, shifted);
  // Further truncate to 8 bits to simulate fp8
  auto fp16AsI16 = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, truncated);
  auto finalTrunc = builder.create<mlir::LLVM::TruncOp>(loc, i8Type, fp16AsI16);
  return finalTrunc;
}

// fp8 e4m3 -> fp32 (simplified)
inline mlir::Value fp8_e4m3_to_fp32(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    mlir::Value fp8Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto f32Type = mlir::Float32Type::get(ctx);
  auto f16Type = mlir::Float16Type::get(ctx);

  // For older hardware, we'll extend fp8 -> fp16 -> fp32 to simulate the conversion
  // This is a simplified approach for compatibility
  auto extended = builder.create<mlir::LLVM::ZExtOp>(loc, i16Type, fp8Val);
  auto fp16Val = builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, extended);
  // Convert fp16 to fp32
  auto fp16Bits = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);
  auto ext = builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, fp16Bits);
  auto shifted = builder.create<mlir::LLVM::ShlOp>(loc, ext, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, shifted);
}

// fp32 -> fp8 e5m2 (simplified)
inline mlir::Value fp32_to_fp8_e5m2(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    mlir::Value fp32Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto i8Type = mlir::IntegerType::get(ctx, 8);

  // For older hardware, we'll convert fp32 -> fp16 -> truncated to simulate fp8
  // This is a simplified approach for compatibility
  auto bits = builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);
  auto shifted = builder.create<mlir::LLVM::LShrOp>(loc, bits, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i16Type, shifted);
  // Further truncate to 8 bits to simulate fp8
  auto fp16AsI16 = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, truncated);
  auto finalTrunc = builder.create<mlir::LLVM::TruncOp>(loc, i8Type, fp16AsI16);
  return finalTrunc;
}

// fp8 e5m2 -> fp32 (simplified)
inline mlir::Value fp8_e5m2_to_fp32(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    mlir::Value fp8Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto f32Type = mlir::Float32Type::get(ctx);
  auto f16Type = mlir::Float16Type::get(ctx);

  // For older hardware, we'll extend fp8 -> fp16 -> fp32 to simulate the conversion
  // This is a simplified approach for compatibility
  auto extended = builder.create<mlir::LLVM::ZExtOp>(loc, i16Type, fp8Val);
  auto fp16Val = builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, extended);
  // Convert fp16 to fp32
  auto fp16Bits = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);
  auto ext = builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, fp16Bits);
  auto shifted = builder.create<mlir::LLVM::ShlOp>(loc, ext, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, shifted);
}

// bf16 -> fp32
inline mlir::Value bf16_to_fp32(mlir::Location loc,
                                mlir::OpBuilder &builder,
                                mlir::Value bf16Val) {
  auto *ctx = builder.getContext();
  auto i16Type = mlir::IntegerType::get(ctx, 16);
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto f32Type = mlir::Float32Type::get(ctx);

  if (bf16Val.getType().isBF16()) {
    bf16Val = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, bf16Val);
  }

  auto ext = builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, bf16Val);
  auto shamt16 = builder.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerAttr::get(i32Type, 16));
  auto shifted = builder.create<mlir::LLVM::ShlOp>(loc, ext, shamt16);
  return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, shifted);
}

// fp32 -> bf16
inline mlir::Value fp32_to_bf16(mlir::Location loc,
                                mlir::OpBuilder &builder,
                                mlir::Value fp32Val) {
  auto *ctx = builder.getContext();
  auto i32Type = mlir::IntegerType::get(ctx, 32);
  auto i16Type = mlir::IntegerType::get(ctx, 16);

  auto bits = builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);
  auto shifted = builder.create<mlir::LLVM::LShrOp>(loc, bits, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
  auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i16Type, shifted);
  return truncated;
}

// fp8 -> fp32
inline mlir::Value fp8_to_fp32(mlir::Location loc, mlir::OpBuilder &builder,
                               mlir::Value fp8Val, mlir::Type elemTy) {
  // Check if the element type is actually an FP8 type before conversion
  if (llvm::isa<mlir::Float8E4M3FNType>(elemTy)) {
    return fp8_e4m3_to_fp32(loc, builder, fp8Val);
  } else if (llvm::isa<mlir::Float8E5M2Type>(elemTy)) {
    return fp8_e5m2_to_fp32(loc, builder, fp8Val);
  }
  // For older hardware or unsupported types, return nullptr
  return nullptr;
}

// fp32 -> fp8
inline mlir::Value triton_gpu_fp32_to_fp8_emulated_conversion(
    mlir::Location loc, mlir::OpBuilder &builder, mlir::Value fp32Val,
    mlir::Type elemTy) {
  // Check if the element type is actually an FP8 type before conversion
  if (llvm::isa<mlir::Float8E4M3FNType>(elemTy) || llvm::isa<mlir::Float8E4M3FNUZType>(elemTy)) {
    return fp32_to_fp8_e4m3(loc, builder, fp32Val);
  } else if (llvm::isa<mlir::Float8E5M2Type>(elemTy)) {
    return fp32_to_fp8_e5m2(loc, builder, fp32Val);
  }
  // For older hardware or unsupported types, return nullptr
  return nullptr;
}

} // namespace fp_conv

#ifdef _MSC_VER
#pragma warning(pop)
#endif
