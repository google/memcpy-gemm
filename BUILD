load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load(":nvcc.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cuda_library(
    name = "cuda_compute_copy_kernel",
    srcs = [
        "src/cuda_compute_copy.cu",
    ],
    hdrs = [
        "src/cuda_compute_copy.cu.h",
        "src/cuda_compute_copy.h",
    ],
)

cc_library(
    name = "matrix_lib",
    srcs = [
        "src/matrix_lib.cc",
    ],
    hdrs = [
        "src/matrix_lib.h",
        "src/matrix_lib_impl.h",
    ],
    deps = [
        "@absl//absl/base",
        "@absl//absl/random",
        "@absl//absl/random:distributions",
        "@glog",
        "@half//:includes",
    ],
)

# Augments matrix_lib with nv_bfloat16, if available.
cc_library(
    name = "matrix_lib_cuda",
    srcs = [
        "src/matrix_lib_cuda.cc",
    ],
    hdrs = ["src/matrix_lib_cuda.h"],
    deps = [
        ":matrix_lib",
        "@cuda//:cuda_headers",
    ],
)

cc_library(
    name = "multi_gemm_lib",
    srcs = ["src/multi_gemm_lib.cc"],
    hdrs = [
        "src/cuda_check.h",
        "src/memory_allocator_interface.h",
        "src/multi_gemm_lib.h",
    ],
    deps = [
        ":matrix_lib",
        ":matrix_lib_cuda",
        "@absl//absl/random",
        "@cuda//:cublas_static",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@glog",
        "@half//:includes",
    ],
)

cc_library(
    name = "memcpy_gemm_lib",
    srcs = ["src/memcpy_gemm_lib.cc"],
    hdrs = ["src/memcpy_gemm_lib.h"],
    deps = [
        ":cuda_compute_copy_kernel",
        ":gemm_test_lib",
        ":multi_gemm_lib",
        "@absl//absl/container:flat_hash_map",
        "@absl//absl/memory",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
        "@absl//absl/synchronization",
        "@absl//absl/time",
        "@absl//absl/types:optional",
        "@cuda//:cuda_headers",
        "@glog",
        "@libnuma//:numa",
    ],
)

cc_binary(
    name = "memcpy_gemm",
    srcs = ["src/memcpy_gemm.cc"],
    deps = [
        ":gemm_test_lib",
        ":memcpy_gemm_lib",
        ":multi_gemm_lib",
        "@absl//absl/container:flat_hash_map",
        "@absl//absl/flags:flag",
        "@absl//absl/flags:parse",
        "@absl//absl/memory",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
        "@absl//absl/time",
        "@cuda//:cuda_headers",
        "@glog",
        "@re2",
    ],
)

py_test(
    name = "memcpy_gemm_test",
    size = "medium",
    srcs = ["src/memcpy_gemm_test.py"],
    data = [
        ":memcpy_gemm",
    ],
    python_version = "PY3",
    tags = [
        "nomsan",
        "requires-gpu-sm70-only",
    ],
)

cc_library(
    name = "gemm_test_lib",
    srcs = [
        "src/gemm_test_lib.cc",
    ],
    hdrs = [
        "src/gemm_test_lib.h",
    ],
    deps = [
        ":gemm_test_lib_internal",
        ":matrix_lib",
        ":matrix_lib_cuda",
        "@absl//absl/memory",
        "@absl//absl/random",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
        "@cuda//:cuda_headers",
        "@glog",
        "@half//:includes",
    ],
)

cc_library(
    name = "gemm_test_lib_internal",
    srcs = [
        "src/gemm_test_lib.h",
        "src/gemm_test_lib_internal.cc",
    ],
    hdrs = [
        "src/gemm_test_lib_internal.h",
    ],
    deps = [
        ":matrix_lib",
        ":matrix_lib_cuda",
        ":multi_gemm_lib",
        "@absl//absl/container:flat_hash_map",
        "@absl//absl/memory",
        "@absl//absl/random",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
        "@cuda//:cublas_static",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
    ],
)

cc_library(
    name = "gemm_test_lib_mock",
    hdrs = ["src/gemm_test_lib_mock.h"],
    deps = [
        ":gemm_test_lib",
        ":multi_gemm_lib",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "distribution_tests",
    srcs = ["src/distribution_tests.cc"],
    hdrs = ["src/distribution_tests.h"],
    deps = [
        "@absl//absl/types:span",
        "@glog",
    ],
)

cc_test(
    name = "distribution_tests_test",
    size = "small",
    srcs = ["src/distribution_tests_test.cc"],
    deps = [
        ":distribution_tests",
        "@absl//absl/memory",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "matrix_lib_test",
    srcs = ["src/matrix_lib_test.cc"],
    deps = [
        ":distribution_tests",
        ":matrix_lib",
        "@absl//absl/random",
        "@gtest//:gtest_main",
        "@half//:includes",
    ],
)

cc_test(
    name = "matrix_lib_cuda_test",
    srcs = ["src/matrix_lib_test.cc"],
    defines = ["RUN_CUDA_TESTS"],
    deps = [
        ":distribution_tests",
        ":matrix_lib",
        ":matrix_lib_cuda",
        "@absl//absl/random",
        "@cuda//:cuda_headers",
        "@gtest//:gtest_main",
        "@half//:includes",
    ],
)

cc_test(
    name = "gemm_test_lib_internal_test",
    srcs = [
        "src/gemm_test_lib_internal_test.cc",
    ],
    tags = [
        "nomsan",
        "requires-gpu-sm70-only",
    ],
    deps = [
        ":gemm_test_lib",
        ":gemm_test_lib_internal",
        ":multi_gemm_lib",
        "@gtest//:gtest_main",
    ],
)

bzl_library(
    name = "numa_bzl",
    srcs = ["numa.bzl"],
)

bzl_library(
    name = "cuda_bzl",
    srcs = ["cuda.bzl"],
)
