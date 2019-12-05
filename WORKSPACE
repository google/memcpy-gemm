workspace(name = "memcpy_gemm")

load ("//:cuda.bzl", "cuda_configure")
load ("//:numa.bzl", "numa_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")


# Repositories that come with a build file
http_archive(
    name = "absl",
    strip_prefix = "abseil-cpp-0514227d2547793b23e209809276375e41c76617",
    urls = [
      "https://github.com/abseil/abseil-cpp/archive/0514227d2547793b23e209809276375e41c76617.zip", # 2019-11-26
    ],
)

http_archive(
    name = "gtest",
    strip_prefix = "googletest-34e92be31cf457ad4054b7908ee5e0e214dbcddc",
    urls = [
      "https://github.com/google/googletest/archive/34e92be31cf457ad4054b7908ee5e0e214dbcddc.zip", # 2019-11-26
    ],
)

http_archive(
    name = "re2",
    strip_prefix = "re2-bb8e777557ddbdeabdedea4f23613c5021ffd7b1",
    urls = [
      "https://github.com/google/re2/archive/bb8e777557ddbdeabdedea4f23613c5021ffd7b1.zip", # 2019-11-25
    ],
)

# Repositories without a build file - we link our own.
# glog has a build file, but we use our own to disable gflags.
http_archive(
    name = "glog",
    build_file = "//third_party:glog.BUILD",
    strip_prefix = "glog-925858d9969d8ee22aabc3635af00a37891f4e25",
    urls = [
      "https://github.com/google/glog/archive/925858d9969d8ee22aabc3635af00a37891f4e25.zip", # 2019-11-20
    ],
)

http_archive(
    name = "half",
    build_file = "//third_party:half.BUILD",
    urls = [
      "https://ayera.dl.sourceforge.net/project/half/half/1.12.0/half-1.12.0.zip",
    ],
)

# ===== cuda and libnuma need to be configured for the platform =====
cuda_configure(name = "cuda")
numa_configure(name = "libnuma")
