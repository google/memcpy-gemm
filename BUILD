package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "matrix_lib",
    srcs = ["matrix_lib.cc"],
    hdrs = [
        "matrix_lib.h",
    ],
    deps = [
        "@absl//absl/base",
        "@absl//absl/flags:flag",
        "@absl//absl/random",
        "@glog",
        "@half//:includes",
    ],
)

cc_library(
    name = "multi_gemm_lib",
    srcs = ["multi_gemm_lib.cc"],
    hdrs = [
        "cuda_check.h",
        "memory_allocator_interface.h",
        "multi_gemm_lib.h",
    ],
    deps = [
        ":matrix_lib",
        "@absl//absl/random",
        "@absl//absl/strings:str_format",
        "@absl//absl/time",
        "@cuda//:cublas_static",
        "@cuda//:cuda_headers",
        "@cuda//:cuda_runtime",
        "@glog",
        "@half//:includes",
    ],
)

cc_library(
    name = "memcpy_gemm_lib",
    srcs = ["memcpy_gemm_lib.cc"],
    hdrs = ["memcpy_gemm_lib.h"],
    deps = [
        ":gemm_test_lib",
        ":multi_gemm_lib",
        "@absl//absl/memory",
        "@absl//absl/strings",
        "@absl//absl/strings:str_format",
        "@absl//absl/synchronization",
        "@absl//absl/time",
        "@cuda//:cuda_headers",
        "@glog",
        "@libnuma//:numa",
    ],
)

cc_binary(
    name = "memcpy_gemm",
    srcs = ["memcpy_gemm.cc"],
    deps = [
        ":gemm_test_lib",
        ":memcpy_gemm_lib",
        ":multi_gemm_lib",
        "@absl//absl/container:flat_hash_set",
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
    srcs = ["memcpy_gemm_test.py"],
    data = [
        ":memcpy_gemm",
    ],
    python_version = "PY3",
    tags = [
        "nomsan",
        "nvcc",
        "requires-gpu-sm35",
    ],
)

cc_library(
    name = "gemm_test_lib",
    srcs = [
        "gemm_test_lib.cc",
    ],
    hdrs = [
        "gemm_test_lib.h",
    ],
    tags = ["cuda8"],
    deps = [
        ":gemm_test_lib_internal",
        ":matrix_lib",
        "@absl//absl/memory",
        "@absl//absl/random",
        "@cuda//:cuda_headers",
        "@half//:includes",
    ],
)

cc_library(
    name = "gemm_test_lib_internal",
    srcs = [
        "gemm_test_lib.h",
        "gemm_test_lib_internal.cc",
    ],
    hdrs = [
        "gemm_test_lib_internal.h",
    ],
    tags = ["cuda8"],
    deps = [
        ":matrix_lib",
        ":multi_gemm_lib",
        "@absl//absl/memory",
        "@absl//absl/random",
        "@absl//absl/strings:str_format",
        "@cuda//:cublas_static",
        "@cuda//:cuda_headers",
    ],
)

cc_library(
    name = "gemm_test_lib_mock",
    hdrs = ["gemm_test_lib_mock.h"],
    tags = ["cuda8"],
    deps = [
        ":gemm_test_lib",
        ":multi_gemm_lib",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "distribution_tests",
    srcs = ["distribution_tests.cc"],
    hdrs = ["distribution_tests.h"],
    deps = [
        "@absl//absl/types:span",
        "@glog",
    ],
)

cc_test(
    name = "distribution_tests_test",
    size = "small",
    srcs = ["distribution_tests_test.cc"],
    deps = [
        ":distribution_tests",
        "@absl//absl/memory",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "matrix_lib_test",
    srcs = ["matrix_lib_test.cc"],
    deps = [
        ":distribution_tests",
        ":matrix_lib",
        "@absl//absl/random",
        "@gtest//:gtest_main",
        "@half//:includes",
    ],
)
