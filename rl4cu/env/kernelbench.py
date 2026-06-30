"""
KernelBench problem loader and prompt formatter.

Wraps the kernelbench dataset API to give us a clean interface for:
  - loading problems by level / id
  - formatting them into LLM prompts
  - iterating over problems for baseline eval or rollout generation
"""

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

KERNELBENCH_ROOT = pathlib.Path(__file__).parent.parent.parent / "kernelbench"

SYSTEM_PROMPT = """\
You are an expert CUDA kernel developer. Your task is to replace a PyTorch \
model implementation with a custom CUDA kernel that is both correct and \
significantly faster than the PyTorch baseline.

Rules:
- Implement a class called `ModelNew` with the exact same `__init__` and `forward` signature as `Model`.
- Write custom CUDA kernels using `torch.utils.cpp_extension.load_inline` with inline source strings — do NOT use external .cu files or `load()` with file paths.
- The CUDA kernel source and C++ binding must be passed as strings to `load_inline`, not as file references.
- Call the kernel from `forward` using the loaded extension object.
- The output must be numerically equivalent to the reference (within float32 tolerance).
- Do not cache results between calls, modify input tensors, or use non-default CUDA streams.
- Do not just wrap PyTorch ops — write an actual `__global__` CUDA kernel.

Use this pattern:
```python
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
__global__ void my_kernel(...) { ... }

torch::Tensor my_op(torch::Tensor x) {
    // launch kernel, return result
}
\"\"\"

cpp_source = "torch::Tensor my_op(torch::Tensor x);"

ext = load_inline(name="my_ext", cpp_sources=cpp_source, cuda_sources=cuda_source,
                  functions=["my_op"], verbose=False, with_cuda=True)

class ModelNew(nn.Module):
    def forward(self, x):
        return ext.my_op(x)
```

Respond with only the complete Python source code for `ModelNew`, no explanation.
"""

USER_PROMPT_TEMPLATE = """\
Optimize the following PyTorch model by replacing it with a custom CUDA kernel:

```python
{ref_code}
```

Write `ModelNew` using `load_inline` with inline CUDA source strings. The kernel must be faster than the PyTorch baseline.
"""

REFINEMENT_PROMPT_TEMPLATE = """\
Your previous kernel attempt had the following result:

- Compiled: {compiled}
- Correct:  {correct}
- Speedup:  {speedup}
- Error:    {error_msg}

Here is your previous attempt:
```python
{prev_kernel}
```

Please fix the issues and provide an improved implementation of `ModelNew`.
"""


@dataclass
class KernelProblem:
    problem_id: int
    level: int
    name: str
    ref_code: str

    def format_prompt(self) -> list[dict]:
        """Format as OpenAI-style messages list."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(ref_code=self.ref_code)},
        ]

    def format_refinement_turn(
        self,
        prev_kernel: str,
        compiled: bool,
        correct: bool,
        speedup: float,
        error_msg: Optional[str],
    ) -> dict:
        """Format a refinement turn (added to messages after model response)."""
        speedup_str = f"{speedup:.3f}x" if speedup > 0 else "N/A"
        return {
            "role": "user",
            "content": REFINEMENT_PROMPT_TEMPLATE.format(
                compiled=compiled,
                correct=correct,
                speedup=speedup_str,
                error_msg=error_msg or "none",
                prev_kernel=prev_kernel,
            ),
        }


def load_problems(
    level: int,
    problem_ids: Optional[list[int]] = None,
    source: str = "local",
) -> list[KernelProblem]:
    """
    Load KernelBench problems for a given level.

    Args:
        level:       1, 2, or 3.
        problem_ids: Optional subset of problem IDs to load. Loads all if None.
        source:      "local" (from submodule) or "huggingface".

    Returns:
        List of KernelProblem objects.
    """
    from kernelbench.dataset import construct_kernelbench_dataset

    dataset = construct_kernelbench_dataset(
        level=level,
        source=source,
    )

    ids = problem_ids if problem_ids is not None else dataset.get_problem_ids()

    problems = []
    for pid in ids:
        problem = dataset.get_problem_by_id(pid)
        problems.append(
            KernelProblem(
                problem_id=pid,
                level=level,
                name=getattr(problem, "name", str(pid)),
                ref_code=problem.code,
            )
        )

    return problems
