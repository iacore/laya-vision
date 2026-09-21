const std = @import("std");

fn linkGgml(m: *std.Build.Module, ggml_mod: *std.Build.Module, lib_dir: []const u8, b: *std.Build) void {
    m.addImport("ggml", ggml_mod);
    // standalone ggml puts ggml/ggml-base/ggml-cpu in <build>/src and the Vulkan backend
    // one level down in <build>/src/ggml-vulkan.
    m.addLibraryPath(.{ .cwd_relative = lib_dir });
    m.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/ggml-vulkan", .{lib_dir}) });
    m.linkSystemLibrary("ggml", .{});
    m.linkSystemLibrary("ggml-base", .{});
    m.linkSystemLibrary("ggml-cpu", .{});
    m.linkSystemLibrary("ggml-vulkan", .{});
}

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

    const head_mod = b.createModule(.{
        .root_source_file = b.path("src/head.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(head_mod, ggml_mod, ggml_lib, b);
    const head_exe = b.addExecutable(.{ .name = "head", .root_module = head_mod });
    b.installArtifact(head_exe);

    const enc_mod = b.createModule(.{
        .root_source_file = b.path("src/encoder.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(enc_mod, ggml_mod, ggml_lib, b);
    const enc_exe = b.addExecutable(.{ .name = "encoder", .root_module = enc_mod });
    b.installArtifact(enc_exe);

    const run = b.addRunArtifact(head_exe);
    run.addPassthruArgs();
    b.step("run", "run the head against an oracle case").dependOn(&run.step);

    const run_enc = b.addRunArtifact(enc_exe);
    run_enc.addPassthruArgs();
    b.step("run-encoder", "run the encoder against an oracle case").dependOn(&run_enc.step);
}
