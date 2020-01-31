""" Configure and create build file for libnuma """

def build_numa_lib(repository_ctx):
    repository_ctx.download_and_extract(
        url = "https://github.com/numactl/numactl/archive/3648aa5bf6e29bf618195c615ff2ced4bb995327.zip",
        stripPrefix = "numactl-3648aa5bf6e29bf618195c615ff2ced4bb995327",
        sha256 = "70b41bac88587ee980f266b3b2a9f32e9efef7c003a258eb50677cd8c66f358e",
    )
    repository_ctx.execute(["./autogen.sh"])
    repository_ctx.execute(["./configure"])
    repository_ctx.file(
        "BUILD",
        """
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "numa",
    srcs = [
        "affinity.c",
        "config.h",
        "distance.c",
        "libnuma.c",
        "rtnetlink.c",
        "syscall.c",
        "sysfs.c",
    ],
    hdrs = [
        "affinity.h",
        "numa.h",
        "numaif.h",
        "numaint.h",
        "rtnetlink.h",
        "sysfs.h",
        "util.h",
    ],
)
""",
    )

numa_configure = repository_rule(
    implementation = build_numa_lib,
)
