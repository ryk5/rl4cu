"""
Modal kernel evaluation worker.

Each call takes a kernel code string + reference problem source, compiles it,
checks correctness against the PyTorch reference, benchmarks it, and returns
structured results. Designed to be called via .map() for parallel evaluation
of many kernel candidates at once.
"""

import modal

# ---------------------------------------------------------------------------
# Image: CUDA 12.4, PyTorch, kernelbench installed from our submodule
# ---------------------------------------------------------------------------

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .pip_install(
        "torch>=2.9.0",
        "ninja",
        "numpy",
        "einops",
        "tqdm",
        "packaging",
        "pydra-config",
        "tomli",
        "tabulate",
        "python-dotenv",
        "setuptools",
    )
    # copy kernelbench submodule into the image and install it
    .add_local_dir("./kernelbench", remote_path="/root/kernelbench", copy=True)
    .run_commands("pip install -e /root/kernelbench")
)

app = modal.App("rl4cu-kernel-eval")


@app.function(
    image=cuda_image,
    gpu="H100",
    timeout=300,
    retries=1,
)
def eval_kernel(
    kernel_code: str,
    ref_code: str,
    problem_id: int,
    num_correct_trials: int = 5,
    num_perf_trials: int = 20,
    measure_performance: bool = True,
) -> dict:
    """
    Evaluate a single generated kernel against the reference PyTorch implementation.

    Args:
        kernel_code:         The generated CUDA kernel source code.
        ref_code:            The reference PyTorch model source (from KernelBench).
        problem_id:          KernelBench problem ID, for logging.
        num_correct_trials:  How many correctness check runs.
        num_perf_trials:     How many timing runs for benchmarking.
        measure_performance: Whether to run the perf benchmark (skip during cold-start data gen).

    Returns:
        dict with keys:
            compiled (bool)
            correct (bool)
            speedup (float)        -- ref_runtime / kernel_runtime, -1 if not measured
            kernel_runtime_ms (float)
            ref_runtime_ms (float)
            error_msg (str | None) -- compiler / runtime error if any
            problem_id (int)
    """
    from kernelbench.eval import eval_kernel_against_ref
    from kernelbench.kernel_static_checker import validate_kernel_static

    result = {
        "compiled": False,
        "correct": False,
        "speedup": -1.0,
        "kernel_runtime_ms": -1.0,
        "ref_runtime_ms": -1.0,
        "error_msg": None,
        "problem_id": problem_id,
    }

    # static anti-cheat check before we even run anything
    is_valid, errors, _warnings = validate_kernel_static(kernel_code)
    if not is_valid:
        result["error_msg"] = f"static_checker: {'; '.join(errors)}"
        return result

    try:
        exec_result = eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=kernel_code,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            measure_performance=measure_performance,
            timing_method="cuda_event",
            verbose=False,
        )

        result["compiled"] = exec_result.compiled
        result["correct"] = exec_result.correctness

        if exec_result.compiled and exec_result.correctness and measure_performance:
            result["kernel_runtime_ms"] = exec_result.runtime
            result["ref_runtime_ms"] = exec_result.ref_runtime
            if exec_result.runtime > 0:
                result["speedup"] = exec_result.ref_runtime / exec_result.runtime

        if not exec_result.compiled or not exec_result.correctness:
            # surface whatever kernelbench captured — try multiple metadata keys
            err = (
                exec_result.metadata.get("compile_error")
                or exec_result.metadata.get("error")
                or exec_result.metadata.get("correctness_error")
                or str(exec_result.metadata)
            )
            result["error_msg"] = str(err)

    except Exception as e:
        result["error_msg"] = f"{type(e).__name__}: {e}"

    return result


@app.function(
    image=cuda_image,
    gpu="H100",
    timeout=600,
)
def eval_kernels_batch(items: list[dict]) -> list[dict]:
    """
    Evaluate a batch of kernels sequentially on one GPU container.
    Useful when you want to amortize cold-start cost across multiple evals
    (e.g. all G=16 rollouts for one problem on the same container).

    Each item in `items` should be a dict with the same keys as eval_kernel args.
    Returns a list of result dicts in the same order.
    """
    results = []
    for item in items:
        results.append(
            eval_kernel.local(  # run locally (we're already on GPU)
                kernel_code=item["kernel_code"],
                ref_code=item["ref_code"],
                problem_id=item["problem_id"],
                num_correct_trials=item.get("num_correct_trials", 5),
                num_perf_trials=item.get("num_perf_trials", 20),
                measure_performance=item.get("measure_performance", True),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Local entrypoint for quick smoke test
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def smoke_test():
    """
    Quick test: evaluate a trivially correct (but unoptimized) kernel for problem 1.
    Run with: python -m modal run modal_app/kernel_eval.py
    """
    import pathlib

    ref_path = pathlib.Path("kernelbench/KernelBench/level1/1_Square_matrix_multiplication_.py")
    ref_code = ref_path.read_text()

    # minimal CUDA kernel using torch cpp extension — correct but not fast
    naive_kernel = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = \"\"\"
__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    return C;
}
\"\"\"

cpp_src = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_ext = load_inline(
    name="matmul_ext",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_ext.matmul_cuda(A.cuda(), B.cuda())
"""

    print("submitting eval job to Modal H100...")
    result = eval_kernel.remote(
        kernel_code=naive_kernel,
        ref_code=ref_code,
        problem_id=1,
        measure_performance=True,
    )

    print(f"compiled:    {result['compiled']}")
    print(f"correct:     {result['correct']}")
    print(f"speedup:     {result['speedup']:.3f}x")
    print(f"kernel_ms:   {result['kernel_runtime_ms']:.3f}")
    print(f"ref_ms:      {result['ref_runtime_ms']:.3f}")
    print(f"error:       {result['error_msg']}")
