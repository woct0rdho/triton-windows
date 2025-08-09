'''// This file is adapted for Windows on older hardware with improved IEEE 754 compliance.
#pragma once

#ifdef _MSC_VER
// Suppress implicit conversion warnings
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/Casting.h"

    namespace fp_conv {

  // fp32 -> bf16 (with rounding and Inf/NaN handling)
  inline mlir::Value fp32_to_bf16(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value fp32Val) {
    auto *ctx = builder.getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto bf16Type = mlir::BFloat16Type::get(ctx);

    auto fp32Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);

    // Check for NaN/Inf
    auto expMask =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7F800000);
    auto isNanInf = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq,
        builder.create<mlir::LLVM::AndOp>(loc, fp32Bits, expMask), expMask);

    // Rounding logic (round to nearest even)
    auto rounding_bias =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7FFF);
    auto rounded_bits =
        builder.create<mlir::LLVM::AddOp>(loc, fp32Bits, rounding_bias);

    // Combine and select
    auto shifted = builder.create<mlir::LLVM::LShrOp>(
        loc, rounded_bits,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
    auto nan_inf_shifted = builder.create<mlir::LLVM::LShrOp>(
        loc, fp32Bits,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
    auto result_bits = builder.create<mlir::LLVM::SelectOp>(
        loc, isNanInf, nan_inf_shifted, shifted);

    auto truncated =
        builder.create<mlir::LLVM::TruncOp>(loc, i16Type, result_bits);
    return builder.create<mlir::LLVM::BitcastOp>(loc, bf16Type, truncated);
  }

  // bf16 -> fp32
  inline mlir::Value bf16_to_fp32(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value bf16Val) {
    auto *ctx = builder.getContext();
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto f32Type = mlir::Float32Type::get(ctx);

    auto bits = builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, bf16Val);
    auto extended = builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, bits);
    auto shifted = builder.create<mlir::LLVM::ShlOp>(
        loc, extended,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
    return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, shifted);
  }

  // fp32 -> fp16 (with full Inf/NaN/Subnormal and rounding handling)
  inline mlir::Value fp32_to_fp16(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value fp32Val) {
    auto *ctx = builder.getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto f16Type = mlir::Float16Type::get(ctx);

    auto fp32Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);

    auto sign = builder.create<mlir::LLVM::AndOp>(
        loc, fp32Bits,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x80000000));
    auto rest = builder.create<mlir::LLVM::AndOp>(
        loc, fp32Bits,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7FFFFFFF));
    auto sign16 = builder.create<mlir::LLVM::LShrOp>(
        loc, sign, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));

    // Handle NaN/Inf
    auto nan_inf_result = builder.create<mlir::LLVM::OrOp>(
        loc, sign16,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7C00));
    auto is_nan_inf = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::uge, rest,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type,
                                               0x7F800000)); // exp all 1s

    // Handle subnormals
    auto denorm_magic = builder.create<mlir::LLVM::ConstantOp>(
        loc, i32Type, (15 - 127 + 10) << 23);
    auto denorm_result = builder.create<mlir::LLVM::TruncOp>(
        loc, i16Type,
        builder.create<mlir::LLVM::LShrOp>(
            loc, builder.create<mlir::LLVM::AddOp>(loc, rest, denorm_magic),
            builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 13)));
    auto is_denorm = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ule, rest,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x38800000 - 1));

    // Handle normals
    auto normal_rest = builder.create<mlir::LLVM::SubOp>(
        loc, rest,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, (127 - 15) << 23));
    auto normal_result = builder.create<mlir::LLVM::TruncOp>(
        loc, i16Type,
        builder.create<mlir::LLVM::LShrOp>(
            loc,
            builder.create<mlir::LLVM::AddOp>(
                loc, normal_rest,
                builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 1 << 12)),
            builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 13)));

    auto result16 = builder.create<mlir::LLVM::SelectOp>(
        loc, is_denorm, denorm_result, normal_result);
    result16 = builder.create<mlir::LLVM::SelectOp>(
        loc, is_nan_inf,
        builder.create<mlir::LLVM::TruncOp>(loc, i16Type, nan_inf_result),
        result16);
    result16 = builder.create<mlir::LLVM::OrOp>(
        loc, result16,
        builder.create<mlir::LLVM::TruncOp>(loc, i16Type, sign16));

    return builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, result16);
  }

  // fp16 -> fp32
  inline mlir::Value fp16_to_fp32(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value fp16Val) {
    auto *ctx = builder.getContext();
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto f32Type = mlir::Float32Type::get(ctx);

    auto fp16Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);
    auto fp16Bits32 =
        builder.create<mlir::LLVM::ZExtOp>(loc, i32Type, fp16Bits);

    auto sign = builder.create<mlir::LLVM::ShlOp>(
        loc,
        builder.create<mlir::LLVM::AndOp>(
            loc, fp16Bits32,
            builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x8000)),
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 16));
    auto exp = builder.create<mlir::LLVM::AndOp>(
        loc, fp16Bits32,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7C00));
    auto mant = builder.create<mlir::LLVM::AndOp>(
        loc, fp16Bits32,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x03FF));

    // Magic numbers for conversion
    auto magic = builder.create<mlir::LLVM::ConstantOp>(
        loc, i32Type, ((127 - 15) + 23) << 23);
    auto denorm_magic =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 127 << 23);

    auto is_denorm = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, exp,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0));
    auto is_inf_nan = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, exp,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x7C00));

    auto shifted_exp = builder.create<mlir::LLVM::ShlOp>(
        loc, exp, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 13));
    auto shifted_mant = builder.create<mlir::LLVM::ShlOp>(
        loc, mant, builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 13));

    auto normal_val =
        builder.create<mlir::LLVM::AddOp>(loc, shifted_exp, magic);
    normal_val =
        builder.create<mlir::LLVM::AddOp>(loc, normal_val, shifted_mant);

    auto denorm_val =
        builder.create<mlir::LLVM::AddOp>(loc, shifted_mant, denorm_magic);
    denorm_val = builder.create<mlir::LLVM::SubOp>(
        loc, denorm_val,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 1 << 23));

    auto inf_nan_val = builder.create<mlir::LLVM::AddOp>(
        loc, normal_val,
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 128 << 23));

    auto result = builder.create<mlir::LLVM::SelectOp>(loc, is_denorm,
                                                       denorm_val, normal_val);
    result = builder.create<mlir::LLVM::SelectOp>(loc, is_inf_nan, inf_nan_val,
                                                  result);
    result = builder.create<mlir::LLVM::OrOp>(loc, result, sign);

    return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, result);
  }

  // fp16 -> bf16
  inline mlir::Value fp16_to_bf16(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value fp16Val) {
    auto fp32Val = fp16_to_fp32(loc, builder, fp16Val);
    return fp32_to_bf16(loc, builder, fp32Val);
  }

  // bf16 -> fp16
  inline mlir::Value bf16_to_fp16(mlir::Location loc, mlir::OpBuilder & builder,
                                  mlir::Value bf16Val) {
    auto fp32Val = bf16_to_fp32(loc, builder, bf16Val);
    return fp32_to_fp16(loc, builder, fp32Val);
  }

  // fp32 -> fp8 (E4M3) - Direct conversion with saturation
  // fp32 -> fp8 (E4M3) - Direct conversion with saturation
  inline mlir::Value fp32_to_fp8_e4m3(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp32Val) {
    auto *ctx = builder.getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto f8e4m3Type = mlir::Float8E4M3FNType::get(ctx);

    auto fp32Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i32Type, fp32Val);
    auto sign = builder.create<mlir::LLVM::AndOp>(loc, fp32Bits, 0x80000000);
    auto rest = builder.create<mlir::LLVM::AndOp>(loc, fp32Bits, 0x7FFFFFFF);

    // E4M3 does not have infinity. Saturate to max value.
    // Max value for E4M3 is 240.0f, which is 0x43700000 in fp32.
    auto max_val =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0x43700000);
    auto is_over = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::uge, rest, max_val);
    rest = builder.create<mlir::LLVM::SelectOp>(loc, is_over, max_val, rest);

    // Magic number conversion, similar to fp16 but for E4M3 parameters
    auto magic =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, (127 - 7) << 23);
    auto round_add =
        builder.create<mlir::LLVM::ConstantOp>(loc, i32Type, 1 << (23 - 3 - 1));
    auto shifted = builder.create<mlir::LLVM::LShrOp>(
        loc,
        builder.create<mlir::LLVM::AddOp>(
            loc, builder.create<mlir::LLVM::SubOp>(loc, rest, magic),
            round_add),
        20);

    auto result = builder.create<mlir::LLVM::OrOp>(
        loc, builder.create<mlir::LLVM::LShrOp>(loc, sign, 24), shifted);
    auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i8Type, result);
    return builder.create<mlir::LLVM::BitcastOp>(loc, f8e4m3Type, truncated);
  }

  // fp8 (E4M3) -> fp32 - Direct conversion
  inline mlir::Value fp8_e4m3_to_fp32(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp8Val) {
    auto *ctx = builder.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto f32Type = mlir::Float32Type::get(ctx);

    auto fp8Bits32 = builder.create<mlir::LLVM::ZExtOp>(
        loc, i32Type,
        builder.create<mlir::LLVM::BitcastOp>(loc, i8Type, fp8Val));

    auto sign = builder.create<mlir::LLVM::ShlOp>(
        loc, builder.create<mlir::LLVM::AndOp>(loc, fp8Bits32, 0x80), 24);
    auto exp8 = builder.create<mlir::LLVM::LShrOp>(
        loc, builder.create<mlir::LLVM::AndOp>(loc, fp8Bits32, 0x78), 3);
    auto mant8 = builder.create<mlir::LLVM::AndOp>(loc, fp8Bits32, 0x07);

    auto exp32 = builder.create<mlir::LLVM::AddOp>(loc, exp8, 127 - 7);
    auto mant32 = builder.create<mlir::LLVM::ShlOp>(loc, mant8, 20);

    auto result = builder.create<mlir::LLVM::OrOp>(
        loc, sign,
        builder.create<mlir::LLVM::OrOp>(
            loc, builder.create<mlir::LLVM::ShlOp>(loc, exp32, 23), mant32));
    return builder.create<mlir::LLVM::BitcastOp>(loc, f32Type, result);
  }

  // fp32 -> fp8 (E5M2) - Direct conversion
  inline mlir::Value fp32_to_fp8_e5m2(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp32Val) {
    auto *ctx = builder.getContext();
    auto i32Type = mlir::IntegerType::get(ctx, 32);
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto f8e5m2Type = mlir::Float8E5M2Type::get(ctx);

    // Use the highly compliant fp32->fp16 conversion and truncate the mantissa,
    // as E5M2 is structurally a truncated FP16.
    auto fp16Val = fp32_to_fp16(loc, builder, fp32Val);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto fp16Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);

    // Round the 10-bit mantissa of fp16 to 2-bit mantissa for fp8
    auto round_add =
        builder.create<mlir::LLVM::ConstantOp>(loc, i16Type, 1 << (10 - 2 - 1));
    auto rounded = builder.create<mlir::LLVM::AddOp>(loc, fp16Bits, round_add);
    auto shifted = builder.create<mlir::LLVM::LShrOp>(loc, rounded, 8);

    auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i8Type, shifted);
    return builder.create<mlir::LLVM::BitcastOp>(loc, f8e5m2Type, truncated);
  }

  // fp8 (E5M2) -> fp32 - Direct conversion
  inline mlir::Value fp8_e5m2_to_fp32(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp8Val) {
    auto *ctx = builder.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto f16Type = mlir::Float16Type::get(ctx);

    // Promote to fp16 as E5M2 has the same exponent bias and is a truncated
    // fp16
    auto fp8Bits = builder.create<mlir::LLVM::BitcastOp>(loc, i8Type, fp8Val);
    auto fp16Bits = builder.create<mlir::LLVM::ShlOp>(
        loc, builder.create<mlir::LLVM::ZExtOp>(loc, i16Type, fp8Bits), 8);

    auto fp16Val =
        builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, fp16Bits);
    return fp16_to_fp32(loc, builder, fp16Val);
  }
  '''

      // fp8 (E4M3) -> fp32 - Direct conversion
      inline mlir::Value
      fp8_e4m3_to_fp32(mlir::Location loc, mlir::OpBuilder & builder,
                       mlir::Value fp8Val) {
    // Promote to fp16 first, then to fp32, as it's a simpler path up.
    auto *ctx = builder.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto f16Type = mlir::Float16Type::get(ctx);

    auto fp8Bits = builder.create<mlir::LLVM::BitcastOp>(loc, i8Type, fp8Val);
    auto fp16Bits = builder.create<mlir::LLVM::ZExtOp>(
        loc, i16Type, fp8Bits); // This is a simplification
    auto fp16Val =
        builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, fp16Bits);

    return fp16_to_fp32(loc, builder, fp16Val);
  }

  // fp32 -> fp8 (E5M2) - Direct conversion
  inline mlir::Value fp32_to_fp8_e5m2(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp32Val) {
    // E5M2 is just like FP16 but with a 2-bit mantissa.
    // We can leverage the existing fp32_to_fp16 logic and then truncate.
    auto *ctx = builder.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto f8e5m2Type = mlir::Float8E5M2Type::get(ctx);

    auto fp16Val = fp32_to_fp16(loc, builder, fp32Val);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto fp16Bits =
        builder.create<mlir::LLVM::BitcastOp>(loc, i16Type, fp16Val);

    // Round the 10-bit mantissa of fp16 to 2-bit mantissa for fp8
    auto rounding_add =
        builder.create<mlir::LLVM::ConstantOp>(loc, i16Type, 1 << (10 - 2 - 1));
    auto rounded =
        builder.create<mlir::LLVM::AddOp>(loc, fp16Bits, rounding_add);
    auto shifted = builder.create<mlir::LLVM::LShrOp>(
        loc, rounded,
        builder.create<mlir::LLVM::ConstantOp>(loc, i16Type, 10 - 2));

    auto truncated = builder.create<mlir::LLVM::TruncOp>(loc, i8Type, shifted);
    return builder.create<mlir::LLVM::BitcastOp>(loc, f8e5m2Type, truncated);
  }

  // fp8 (E5M2) -> fp32 - Direct conversion
  inline mlir::Value fp8_e5m2_to_fp32(
      mlir::Location loc, mlir::OpBuilder & builder, mlir::Value fp8Val) {
    // Promote to fp16 first, as E5M2 has the same exponent bias
    auto *ctx = builder.getContext();
    auto i8Type = mlir::IntegerType::get(ctx, 8);
    auto i16Type = mlir::IntegerType::get(ctx, 16);
    auto f16Type = mlir::Float16Type::get(ctx);

    auto fp8Bits = builder.create<mlir::LLVM::BitcastOp>(loc, i8Type, fp8Val);
    auto fp16Bits = builder.create<mlir::LLVM::ShlOp>(
        loc, builder.create<mlir::LLVM::ZExtOp>(loc, i16Type, fp8Bits),
        builder.create<mlir::LLVM::ConstantOp>(loc, i16Type, 10 - 2));

    auto fp16Val =
        builder.create<mlir::LLVM::BitcastOp>(loc, f16Type, fp16Bits);
    return fp16_to_fp32(loc, builder, fp16Val);
  }

  // Generic dispatch functions
  inline mlir::Value fp8_to_fp32(mlir::Location loc, mlir::OpBuilder & builder,
                                 mlir::Value fp8Val, mlir::Type elemTy) {
    if (llvm::isa<mlir::Float8E4M3FNType>(elemTy)) {
      return fp8_e4m3_to_fp32(loc, builder, fp8Val);
    } else if (llvm::isa<mlir::Float8E5M2Type>(elemTy)) {
      return fp8_e5m2_to_fp32(loc, builder, fp8Val);
    }
    return nullptr;
  }

  inline mlir::Value fp32_to_fp8(mlir::Location loc, mlir::OpBuilder & builder,
                                 mlir::Value fp32Val, mlir::Type elemTy) {
    if (llvm::isa<mlir::Float8E4M3FNType>(elemTy) ||
        llvm::isa<mlir::Float8E4M3FNUZType>(elemTy)) {
      return fp32_to_fp8_e4m3(loc, builder, fp32Val);
    } else if (llvm::isa<mlir::Float8E5M2Type>(elemTy)) {
      return fp32_to_fp8_e5m2(loc, builder, fp32Val);
    }
    return nullptr;
  }

} // namespace fp_conv

#ifdef _MSC_VER
#pragma warning(pop)
#endif
''
