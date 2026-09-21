const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Paths to the ggml we link against. Defaults point at the standalone ggml checkout
    // next to this project; override with -Dggml-include / -Dggml-lib.
    const ggml_include = b.option([]const u8, "ggml-include", "directory holding ggml.h") orelse
        "../../ggml/include";
    const ggml_lib = b.option([]const u8, "ggml-lib", "directory holding libggml*.so") orelse
        ".build/ggml/src";

    // The C bindings are generated from the real headers by translate-c, not hand written.
    const tc = b.addTranslateC(.{
        .root_source_file = b.path("include/ggml_all.h"),
        .target = target,
        .optimize = optimize,
    });
    tc.addIncludePath(.{ .cwd_relative = ggml_include });
    const ggml_mod = tc.createModule();

    const mod = b.createModule(.{
        .root_source_file = b.path("src/head.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("ggml", ggml_mod);
    // standalone ggml drops base/ggml/cpu in <build>/src and the Vulkan backend one level down
    mod.addLibraryPath(.{ .cwd_relative = ggml_lib });
    mod.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/ggml-vulkan", .{ggml_lib}) });
    mod.linkSystemLibrary("ggml", .{});
    mod.linkSystemLibrary("ggml-base", .{});
    mod.linkSystemLibrary("ggml-cpu", .{});
    mod.linkSystemLibrary("ggml-vulkan", .{});

    const exe = b.addExecutable(.{ .name = "head", .root_module = mod });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.addPassthruArgs();
    const run_step = b.step("run", "run the head against an oracle case");
    run_step.dependOn(&run.step);
}
