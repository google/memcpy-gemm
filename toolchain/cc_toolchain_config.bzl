def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "clangstatic-toolchain",
        host_system_name = "unknown-linux-gnu",
        target_system_name = "unknown-linux-gnu",
        target_cpu = "k8",
        target_libc = "unknown",
        compiler = "clang-static",
        abi_version = "unknown",
        abi_libc_version = "unknown",
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)