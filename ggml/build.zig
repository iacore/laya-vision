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

    const vis_mod = b.createModule(.{
        .root_source_file = b.path("src/vision.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(vis_mod, ggml_mod, ggml_lib, b);
    const vis_exe = b.addExecutable(.{ .name = "vision", .root_module = vis_mod });
    b.installArtifact(vis_exe);

    const gp_mod = b.createModule(.{
        .root_source_file = b.path("src/gelu_probe.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(gp_mod, ggml_mod, ggml_lib, b);
    const gp_exe = b.addExecutable(.{ .name = "gelu-probe", .root_module = gp_mod });
    b.installArtifact(gp_exe);
    const run_gp = b.addRunArtifact(gp_exe);
    run_gp.addPassthruArgs();
    b.step("run-gelu-probe", "check whether the CPU gelu op is exact").dependOn(&run_gp.step);

    const laya_mod = b.createModule(.{
        .root_source_file = b.path("src/laya.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(laya_mod, ggml_mod, ggml_lib, b);
    laya_mod.addIncludePath(b.path("vendor/stb"));
    laya_mod.addCSourceFile(.{ .file = b.path("vendor/stb/stb_impl.c"), .flags = &.{} });
    const laya_exe = b.addExecutable(.{ .name = "laya", .root_module = laya_mod });
    b.installArtifact(laya_exe);
    const run_laya = b.addRunArtifact(laya_exe);
    run_laya.addPassthruArgs();
    b.step("run-laya", "png + question -> decision, end to end").dependOn(&run_laya.step);

    const tok_mod = b.createModule(.{
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(tok_mod, ggml_mod, ggml_lib, b);
    const tok_exe = b.addExecutable(.{ .name = "tokenizer", .root_module = tok_mod });
    b.installArtifact(tok_exe);
    const run_tok = b.addRunArtifact(tok_exe);
    run_tok.addPassthruArgs();
    b.step("run-tokenizer", "check the tokenizer against the dumped corpus").dependOn(&run_tok.step);

    const pre_mod = b.createModule(.{
        .root_source_file = b.path("src/preprocess.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    linkGgml(pre_mod, ggml_mod, ggml_lib, b);
    pre_mod.addIncludePath(b.path("vendor/stb"));
    pre_mod.addCSourceFile(.{ .file = b.path("vendor/stb/stb_impl.c"), .flags = &.{} });
    const pre_exe = b.addExecutable(.{ .name = "preprocess", .root_module = pre_mod });
    b.installArtifact(pre_exe);
    const run_pre = b.addRunArtifact(pre_exe);
    run_pre.addPassthruArgs();
    b.step("run-preprocess", "check preprocessing against the dumped pixels").dependOn(&run_pre.step);

    const run = b.addRunArtifact(head_exe);
    run.addPassthruArgs();
    b.step("run", "run the head against an oracle case").dependOn(&run.step);

    const run_enc = b.addRunArtifact(enc_exe);
    run_enc.addPassthruArgs();
    b.step("run-encoder", "run the encoder against an oracle case").dependOn(&run_enc.step);

    const run_vis = b.addRunArtifact(vis_exe);
    run_vis.addPassthruArgs();
    b.step("run-vision", "run the vision tower against an oracle case").dependOn(&run_vis.step);
}
