load("//toolchain:cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ['//visibility:public'])
cc_toolchain_config(name = "clangstatic_toolchain_config")

cc_toolchain_suite(
    name = "clangstatic",
    toolchains = {
        "k8|clang-static": ":clangstatic_toolchain",
        "k8": ":clangstatic_toolchain",

    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "clangstatic_toolchain",
    toolchain_identifier = "clangstatic-toolchain",
    toolchain_config = ":clangstatic_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
)