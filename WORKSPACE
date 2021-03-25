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
    sha256 = "2eba659382882cb4b484aee49c34ad483673848974e56f45547f5abff18d0dbf",
)

http_archive(
    name = "bazel_skylib",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
)
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

http_archive(
    name = "gtest",
    strip_prefix = "googletest-34e92be31cf457ad4054b7908ee5e0e214dbcddc",
    urls = [
      "https://github.com/google/googletest/archive/34e92be31cf457ad4054b7908ee5e0e214dbcddc.zip", # 2019-11-26
    ],
    sha256 = "32b379d8e704e374f687f90ea8be610f83578c1ba7d55e3f2a5be11fe991ec2e",
)

http_archive(
    name = "re2",
    strip_prefix = "re2-bb8e777557ddbdeabdedea4f23613c5021ffd7b1",
    urls = [
      "https://github.com/google/re2/archive/bb8e777557ddbdeabdedea4f23613c5021ffd7b1.zip", # 2019-11-25
    ],
    sha256 = "128fa6a017e7cbf6b9ce5614f9842196d384c3563eb2f2e8d7f37f4c64e62857",
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
    sha256 = "dbe787f2a7cf1146f748a191c99ae85d6b931dd3ebdcc76aa7ccae3699149c67",
)

http_archive(
    name = "half",
    build_file = "//third_party:half.BUILD",
    urls = [
      "https://downloads.sourceforge.net/project/half/half/2.1.0/half-2.1.0.zip",
    ],
    sha256 = "ad1788afe0300fa2b02b0d1df128d857f021f92ccf7c8bddd07812685fa07a25",
)

# ===== cuda and libnuma need to be configured for the platform =====
cuda_configure(name = "cuda")
numa_configure(name = "libnuma")
