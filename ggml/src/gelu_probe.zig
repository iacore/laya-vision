//! Probe: is the CPU `ggml_gelu` exact, or does it quantise through the f16 table?
//!
//! Not part of the model. This exists to settle one question raised by the vision tower's
//! accuracy on CPU (see CPU-PRECISION.md): ggml-cpu's vec.h defines GGML_GELU_FP16
//! unconditionally, and the f32 GELU it selects rounds its input AND its output to f16 --
//! about 4.9e-4 relative, per call. If that is what is happening, the vision tower is the
//! only graph affected, because it is the only one that calls ggml_gelu; the head calls
//! ggml_gelu_erf and the encoder uses SwiGLU.
//!
//! So: run both ops over the same inputs on CPU and Vulkan and compare against the exact
//! float32 formulas, computed here in plain Zig.

const std = @import("std");
const c = @import("ggml");

const T = c.struct_ggml_tensor;
const Ctx = c.struct_ggml_context;
const N: i64 = 4096;

fn asT(p: [*c]T) *T {
    if (p == null) @panic("ggml returned a null tensor");
    return @ptrCast(p);
}

/// ggml_gelu_f32's formula, in f32.
fn tanhGelu(x: f32) f32 {
    const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
    return 0.5 * x * (1.0 + std.math.tanh(SQRT_2_OVER_PI * x * (1.0 + 0.044715 * x * x)));
}

/// nn.GELU(), the erf form, in f32.
extern "c" fn erff(x: f32) f32;
fn erfGelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erff(x * 0.70710678118654752440084436210484));
}

fn report(name: []const u8, got: []const f32, ref_fn: *const fn (f32) f32, x: []const f32) void {
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var n_over: usize = 0;
    // one f16 ulp, as a relative bound: 2^-11
    const f16_rel: f32 = 4.8828125e-4;
    for (got, 0..) |v, i| {
        const r = ref_fn(x[i]);
        const d = @abs(v - r);
        if (d > max_abs) max_abs = d;
        if (@abs(r) > 1e-3) {
            const rel = d / @abs(r);
            if (rel > max_rel) max_rel = rel;
            if (rel > f16_rel * 0.25) n_over += 1;
        }
    }
    std.debug.print("  {s:<14} max|abs|={e:.3}  max|rel|={e:.3}  elems > {e:.1} rel: {d}\n", .{ name, max_abs, max_rel, f16_rel * 0.25, n_over });
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const backend_name: []const u8 = if (args.len >= 2) args[1] else "cpu";

    const params = c.ggml_init_params{ .mem_size = 32 * 1024 * 1024, .mem_buffer = null, .no_alloc = true };
    const ctx: *Ctx = @ptrCast(c.ggml_init(params));
    defer c.ggml_free(ctx);

    const backend = if (std.mem.eql(u8, backend_name, "vulkan"))
        c.ggml_backend_vk_init(0)
    else
        c.ggml_backend_cpu_init();
    if (backend == null) return error.NoBackend;
    defer c.ggml_backend_free(backend);

    const x = asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, N));
    const xp = asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, N));

    const host = try alloc.alloc(f32, @intCast(N));
    defer alloc.free(host);
    const host_copy = try alloc.alloc(f32, @intCast(N));
    defer alloc.free(host_copy);
    // span the range a real activation covers, including the tails where an f16 table is
    // coarsest and where the two GELU forms differ most
    var seed: u64 = 0x2545F4914F6CDD1D;
    for (0..@intCast(N)) |i| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const u = @as(f32, @floatFromInt(@as(u32, @truncate(seed >> 32)))) / 4294967296.0;
        host[i] = (u * 24.0) - 12.0;
    }
    @memcpy(host_copy, host);

    const g = asT(c.ggml_gelu(ctx, x));
    const ge = asT(c.ggml_gelu_erf(ctx, xp));

    if (c.ggml_backend_alloc_ctx_tensors(ctx, backend) == null) return error.NoAlloc;
    c.ggml_backend_tensor_set(x, host.ptr, 0, @intCast(N * 4));
    c.ggml_backend_tensor_set(xp, host_copy.ptr, 0, @intCast(N * 4));

    const gf = c.ggml_new_graph(ctx).?;
    c.ggml_build_forward_expand(gf, g);
    c.ggml_build_forward_expand(gf, ge);
    if (c.ggml_backend_graph_compute(backend, gf) != c.GGML_STATUS_SUCCESS) return error.ComputeFailed;

    const gy = try alloc.alloc(f32, @intCast(N));
    defer alloc.free(gy);
    const gey = try alloc.alloc(f32, @intCast(N));
    defer alloc.free(gey);
    c.ggml_backend_tensor_get(g, gy.ptr, 0, @intCast(N * 4));
    c.ggml_backend_tensor_get(ge, gey.ptr, 0, @intCast(N * 4));

    std.debug.print("gelu probe on {s}, {d} values in [-12, 12]\n", .{ backend_name, N });
    report("ggml_gelu", gy, tanhGelu, host);
    report("ggml_gelu_erf", gey, erfGelu, host);

    const tanh_err = blk: {
        var m: f32 = 0;
        for (gy, 0..) |v, i| m = @max(m, @abs(v - tanhGelu(host[i])));
        break :blk m;
    };
    const erf_err = blk: {
        var m: f32 = 0;
        for (gey, 0..) |v, i| m = @max(m, @abs(v - erfGelu(host[i])));
        break :blk m;
    };
    if (tanh_err > erf_err * 100 and tanh_err > 1e-4) {
        std.debug.print("  VERDICT: ggml_gelu is inexact ({e:.3}) while ggml_gelu_erf is not ({e:.3})\n", .{ tanh_err, erf_err });
    } else {
        std.debug.print("  VERDICT: both within {e:.3} / {e:.3}\n", .{ tanh_err, erf_err });
    }
}
