"""
    Contains build rules for compiling CUDA source via NVCC
"""

def cuda_library(name, srcs, hdrs):
    """ Compiles CUDA sources via nvcc 

    Args:
      name: name of the library to build
      srcs: CUDA source files
      hdrs: cc headers

    """
    genrule_name = name + "_nvcc"
    static_out = genrule_name + ".a"
    native.genrule(
        name = genrule_name,
        srcs = srcs,
        outs = [static_out],
        exec_tools = ["@cuda//:cuda/bin/nvcc"],
        cmd = "$(execpath @cuda//:cuda/bin/nvcc) -I. -ccbin=$(CC) $(SRCS) -o $@ -lib -O3",
        local = True,
        toolchains = ["@bazel_tools//tools/cpp:current_cc_toolchain"],
    )

    native.cc_import(
        name = name,
        hdrs = hdrs,
        static_library = static_out,
        alwayslink = 1,
    )
