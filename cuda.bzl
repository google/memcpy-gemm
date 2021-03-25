""" Generate build rule for cuda dependencies"""

def _build_cuda_libraries(repository_ctx):
    cuda_path = "/usr/local/cuda"
    if "CUDA_PATH" in repository_ctx.os.environ:
        cuda_path = repository_ctx.os.environ["CUDA_PATH"]

    repository_ctx.symlink(cuda_path, "cuda")

    repository_ctx.file(
        "BUILD",
        """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda_headers",
    hdrs = glob(
        include = ["cuda/include/**/*.h*"],
        exclude = ["cuda/include/cudnn.h"]
    ),
    # Allows including CUDA headers with angle brackets.
    includes = ["cuda/include"],
)

cc_library(
    name = "cuda_runtime",
    srcs = ["cuda/lib64/libcudart_static.a"],
    linkopts = ["-ldl", "-lrt"],
)

cc_library(
    name = "cublas_static",
    srcs = [
        "cuda/lib64/libcublas_static.a",
        "cuda/lib64/libcublasLt_static.a",
        "cuda/lib64/libculibos.a",
    ],
    deps = [
        ":cuda_headers"
    ],
)

exports_files([
    "cuda/bin/nvcc"
])

""",
    )

cuda_configure = repository_rule(
    implementation = _build_cuda_libraries,
)
