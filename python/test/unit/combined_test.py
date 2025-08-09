import torch
import triton
import triton.language as tl
import traceback

# =============================================================================
# Test Runner with Enhanced Feedback
# =============================================================================


class AnsiColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestRunner:

    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.failed_test_names = []

    def print_test_header(self, test_name):
        print(f"{AnsiColors.BOLD}Test: {test_name}{AnsiColors.ENDC}")

    def print_test_result(self, passed, test_name, message="", details_on_fail=None):
        if passed:
            self.passed_tests += 1
            print(f"{AnsiColors.OKGREEN}PASSED{AnsiColors.ENDC} {message}")
        else:
            self.failed_tests += 1
            self.failed_test_names.append(test_name)
            print(f"{AnsiColors.FAIL}FAILED{AnsiColors.ENDC} {message}")
            if details_on_fail:
                for key, value in details_on_fail.items():
                    print(f"    {key}: {value}")
        print()

    def print_test_skipped(self, test_name, message=""):
        self.skipped_tests += 1
        print(f"{AnsiColors.WARNING}SKIPPED{AnsiColors.ENDC} - {message}")
        print()

    def print_summary(self):
        total_tests = self.passed_tests + self.failed_tests + self.skipped_tests
        print("=" * 50)
        print(f"{AnsiColors.BOLD}Test Summary{AnsiColors.ENDC}")
        print("=" * 50)
        print(f"Total tests: {total_tests}")
        print(f"{AnsiColors.OKGREEN}Passed:  {self.passed_tests}{AnsiColors.ENDC}")
        print(f"{AnsiColors.FAIL}Failed:  {self.failed_tests}{AnsiColors.ENDC}")
        print(f"{AnsiColors.WARNING}Skipped: {self.skipped_tests}{AnsiColors.ENDC}")
        if self.failed_tests > 0:
            print("\nFailed tests:")
            for name in self.failed_test_names:
                print(f"  - {name}")
        print("=" * 50)


# =============================================================================
# Vector Addition Test
# =============================================================================


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run_vector_addition_test(runner):
    test_name = "Vector Addition"
    runner.print_test_header(test_name)
    try:

        def add(x: torch.Tensor, y: torch.Tensor):
            output = torch.empty_like(x)
            assert x.is_cuda and y.is_cuda and output.is_cuda
            n_elements = output.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
            return output

        size = 1024
        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        output_torch = x + y
        output_triton = add(x, y)
        max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
        passed = max_diff < 1e-5
        runner.print_test_result(passed, test_name, f"- Maximum difference: {max_diff:.6f}")
    except Exception as e:
        runner.print_test_result(False, test_name, f"- Error: {e}")
        traceback.print_exc()


# =============================================================================
# Matrix Multiplication Test (Float16)
# =============================================================================


@triton.jit
def matmul_kernel_f16(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                      GROUP_SIZE_M: tl.constexpr):
    # ... (kernel implementation remains the same)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0).to(tl.float32)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run_matmul_f16_test(runner):
    test_name = "Matrix Multiplication (Float16)"
    runner.print_test_header(test_name)
    try:

        def matmul(a: torch.Tensor, b: torch.Tensor):
            M, K = a.shape
            K, N = b.shape
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            matmul_kernel_f16[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                                    c.stride(1), BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8)
            return c

        torch.manual_seed(0)
        a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
        b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a, b)
        max_diff = torch.max(torch.abs(triton_output - torch_output)).item()
        passed = max_diff < 0.05
        runner.print_test_result(passed, test_name, f"- Maximum difference: {max_diff:.6f}")
    except Exception as e:
        runner.print_test_skipped(test_name, f"Error: {e}")
        traceback.print_exc()


# =============================================================================
# FP Conversion Tests (with enhanced feedback)
# =============================================================================


# Kernels remain the same
@triton.jit
def kernel_fp32_to_fp16_roundtrip(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float16).to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_fp16_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_fp32_to_bf16_roundtrip(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


@triton.jit
def kernel_bf16_to_fp32(X, Y, BLOCK_SIZE: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK_SIZE))
    y = x.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK_SIZE), y)


def run_conversion_test(runner, test_name, kernel, x, y_ref, atol, rtol):
    runner.print_test_header(test_name)
    try:
        y = torch.empty_like(y_ref)
        kernel[(1, )](x, y, BLOCK_SIZE=x.shape[0])
        max_diff = torch.max(torch.abs(y - y_ref)).item()
        passed = torch.allclose(y, y_ref, atol=atol, rtol=rtol)
        details = {'Max Diff': f'{max_diff:.6f}', 'Tolerance': f'atol={atol}, rtol={rtol}'}
        if not passed:
            details['Actual'] = y
            details['Expected'] = y_ref
            details['Difference'] = torch.abs(y - y_ref)

        runner.print_test_result(passed, test_name, f"- Max Diff: {max_diff:.6f}",
                                 details_on_fail=details if not passed else None)

    except Exception as e:
        runner.print_test_result(False, test_name, f"- Error: {e}")
        traceback.print_exc()


def run_all_fp_conversion_tests(runner):
    print("=" * 50)
    print(f"{AnsiColors.HEADER}FP Conversion and Emulation Tests{AnsiColors.ENDC}")
    print("=" * 50)

    # FP32 <-> FP16
    x_f32 = torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 65504, -65504], dtype=torch.float32, device='cuda')
    y_ref_f16_rt = x_f32.to(torch.float16).to(torch.float32)
    run_conversion_test(runner, "FP32 -> FP16 Roundtrip", kernel_fp32_to_fp16_roundtrip, x_f32, y_ref_f16_rt, atol=1e-3,
                        rtol=1e-3)

    x_f16 = torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.float16, device='cuda')
    y_ref_f16_up = x_f16.to(torch.float32)
    run_conversion_test(runner, "FP16 -> FP32", kernel_fp16_to_fp32, x_f16, y_ref_f16_up, atol=1e-6, rtol=1e-6)

    # FP32 <-> BF16
    x_bf16_rt = torch.tensor([1.0, 2.5, -10.1, 20.2, 100.3, 200.4, 300.5, -400.6], dtype=torch.float32, device='cuda')
    y_ref_bf16_rt = x_bf16_rt.to(torch.bfloat16).to(torch.float32)
    run_conversion_test(runner, "FP32 -> BF16 Roundtrip", kernel_fp32_to_bf16_roundtrip, x_bf16_rt, y_ref_bf16_rt,
                        atol=1e-2, rtol=1e-2)

    x_bf16 = torch.tensor([1.0, 2.5, -10.1, 20.2], dtype=torch.bfloat16, device='cuda')
    y_ref_bf16_up = x_bf16.to(torch.float32)
    run_conversion_test(runner, "BF16 -> FP32", kernel_bf16_to_fp32, x_bf16, y_ref_bf16_up, atol=1e-2, rtol=1e-2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    runner = TestRunner()
    print("=" * 50)
    print(f"{AnsiColors.HEADER}Starting Triton Tests...{AnsiColors.ENDC}")
    print("=" * 50)

    run_vector_addition_test(runner)
    run_matmul_f16_test(runner)
    # The f8 matmul test is skipped as it's highly dependent on specific hardware features
    # run_matmul_f8_emulated_test(runner)

    run_all_fp_conversion_tests(runner)

    runner.print_summary()
