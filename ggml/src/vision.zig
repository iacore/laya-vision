//! SmolVLM vision tower + Idefics3 connector on ggml: pixels -> 64 image vectors.
//!
//! This is the last piece of compute before the decision head. SigLIP-base with a
//! pixel-shuffle connector:
//!
//!     patches = im2col(pixels)                     [768, 1024]   32x32 patches of 16x16
//!     h = patch_embd(patches) + position_embd      [768, 1024]
//!     12 x (LayerNorm -> MHA(12 heads, no rope, bias) -> + ; LayerNorm -> MLP(GELU) -> +)
//!     h = post_layernorm(h)
//!     image_features = fc(reshape(h, [12288, 64]))  [576, 64]
//!
//! Config read from the GGUF / HF config rather than assumed:
//!   d=768, 12 layers, 12 heads, head_dim 64, ffn 3072, patch 16, 512px, 1024 patches,
//!   layer_norm_eps 1e-6 (note: the *text* model uses 1e-5, they differ),
//!   activation gelu_pytorch_tanh -> ggml_gelu (the tanh form, NOT gelu_erf).
//!   No RoPE, no causal mask: the vision tower is a plain bidirectional encoder.
//!
//! The connector is a plain contiguous reshape: [768, 1024] -> [12288, 64] lines up
//! exactly, because 16 consecutive 768-wide patches are contiguous and 16*768 == 12288.

const std = @import("std");
const c = @import("ggml");

const T = c.struct_ggml_tensor;
const Ctx = c.struct_ggml_context;

const DV: i64 = 768;
const NHV: i64 = 12;
const HDV: i64 = 64;
const NLV: usize = 12;
const FFV: i64 = 3072;
const EPSV: f32 = 1e-6;
const PATCH: i64 = 16;
const GRID: i64 = 32;
const NPATCH: i64 = GRID * GRID;
const IMG: i64 = 512;
const CONN_IN: i64 = DV * 16;
const CONN_OUT: i64 = 576;
const NTOK: i64 = 64;
const CTX_BYTES: usize = 1024 * 1024 * 1024;

var ctx: *Ctx = undefined;
var io_g: std.Io = undefined;
var backend_g: ?*c.struct_ggml_backend = null;

const Pending = struct { t: *T, src: [*]const u8, len: usize };
var pending: std.ArrayList(Pending) = .empty;

fn asT(p: [*c]T) *T {
    if (p == null) @panic("ggml returned a null tensor");
    return @ptrCast(p);
}

fn newT1(ne0: i64) *T {
    return asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, ne0));
}
fn newT2(ne0: i64, ne1: i64) *T {
    return asT(c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, ne0, ne1));
}
fn reshape2(a: *T, ne0: i64, ne1: i64) *T {
    return asT(c.ggml_reshape_2d(ctx, a, ne0, ne1));
}
fn reshape3(a: *T, ne0: i64, ne1: i64, ne2: i64) *T {
    return asT(c.ggml_reshape_3d(ctx, a, ne0, ne1, ne2));
}
fn reshape4(a: *T, ne0: i64, ne1: i64, ne2: i64, ne3: i64) *T {
    return asT(c.ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3));
}
fn permute(a: *T, a0: c_int, a1: c_int, a2: c_int, a3: c_int) *T {
    return asT(c.ggml_permute(ctx, a, a0, a1, a2, a3));
}
fn cont(a: *T) *T {
    return asT(c.ggml_cont(ctx, a));
}
fn mulMat(a: *T, b: *T) *T {
    return asT(c.ggml_mul_mat(ctx, a, b));
}
fn add(a: *T, b: *T) *T {
    return asT(c.ggml_add(ctx, a, b));
}
fn gelu(a: *T) *T {
    return asT(c.ggml_gelu(ctx, a));
}
fn layerNorm(a: *T, w: *T, bias: *T, eps: f32) *T {
    const n = asT(c.ggml_norm(ctx, a, eps));
    return add(mul(n, w), bias);
}
fn mul(a: *T, b: *T) *T {
    return asT(c.ggml_mul(ctx, a, b));
}
fn softMaxExt(a: *T, s: f32) *T {
    return asT(c.ggml_soft_max_ext(ctx, a, null, s, 0));
}
fn biasAdd(y: *T, b: *T) *T {
    return add(y, repeatTo(b, y));
}
fn repeatTo(a: *T, b: *T) *T {
    return asT(c.ggml_repeat(ctx, a, b));
}

fn readFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
}

fn findGt(gc: *Ctx, name: []const u8) *T {
    var nbuf: [128]u8 = undefined;
    @memcpy(nbuf[0..name.len], name);
    nbuf[name.len] = 0;
    const p = c.ggml_get_tensor(gc, &nbuf);
    if (p == null) {
        std.debug.print("missing weight: {s}\n", .{name});
        @panic("missing weight");
    }
    return @ptrCast(p);
}

/// Register a weight, optionally reinterpreting its shape (patch_embd is 4-D in the file
/// but is used as a 768x768 matmul operand).
fn weight(alloc: std.mem.Allocator, gc: *Ctx, name: []const u8, ne0: ?i64, ne1: ?i64) !*T {
    const s = findGt(gc, name);
    var n_src: usize = 4;
    for (s.ne, 0..) |v, i| {
        if (i < ggmlNDims(s)) n_src *= @as(usize, @intCast(v));
    }
    const a0 = ne0 orelse s.ne[0];
    const a1 = ne1 orelse s.ne[1];
    const t = newT2(a0, a1);
    const want: usize = @intCast(a0 * a1 * 4);
    if (want != n_src) {
        std.debug.print("{s}: want {d} bytes, source has {d}\n", .{ name, want, n_src });
        return error.BadWeightSize;
    }
    try pending.append(alloc, .{ .t = t, .src = @ptrCast(s.data.?), .len = want });
    return t;
}

fn ggmlNDims(t: *T) usize {
    var n: usize = 0;
    while (n < 4 and t.ne[n] > 1) n += 1;
    return @max(n, 1);
}

fn uploadAll() void {
    if (c.ggml_backend_alloc_ctx_tensors(ctx, backend_g) == null) @panic("backend could not allocate the graph");
    for (pending.items) |it| c.ggml_backend_tensor_set(it.t, it.src, 0, it.len);
    pending.clearRetainingCapacity();
}

fn block(alloc: std.mem.Allocator, gc: *Ctx, i: usize, x: *T, L: i64) !*T {
    var nb: [96]u8 = undefined;
    const nm = struct {
        fn f(buf: []u8, li: usize, tail: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "v.blk.{d}.{s}", .{ li, tail }) catch unreachable;
        }
    };

    const l1w = try weight(alloc, gc, nm.f(&nb, i, "ln1.weight"), null, null);
    const l1b = try weight(alloc, gc, nm.f(&nb, i, "ln1.bias"), null, null);
    const qw = try weight(alloc, gc, nm.f(&nb, i, "attn_q.weight"), null, null);
    const qb = try weight(alloc, gc, nm.f(&nb, i, "attn_q.bias"), null, null);
    const kw = try weight(alloc, gc, nm.f(&nb, i, "attn_k.weight"), null, null);
    const kb = try weight(alloc, gc, nm.f(&nb, i, "attn_k.bias"), null, null);
    const vw = try weight(alloc, gc, nm.f(&nb, i, "attn_v.weight"), null, null);
    const vb = try weight(alloc, gc, nm.f(&nb, i, "attn_v.bias"), null, null);
    const ow = try weight(alloc, gc, nm.f(&nb, i, "attn_out.weight"), null, null);
    const ob = try weight(alloc, gc, nm.f(&nb, i, "attn_out.bias"), null, null);
    const l2w = try weight(alloc, gc, nm.f(&nb, i, "ln2.weight"), null, null);
    const l2b = try weight(alloc, gc, nm.f(&nb, i, "ln2.bias"), null, null);
    const uw = try weight(alloc, gc, nm.f(&nb, i, "ffn_up.weight"), null, null);
    const ub = try weight(alloc, gc, nm.f(&nb, i, "ffn_up.bias"), null, null);
    const dw = try weight(alloc, gc, nm.f(&nb, i, "ffn_down.weight"), null, null);
    const db = try weight(alloc, gc, nm.f(&nb, i, "ffn_down.bias"), null, null);

    // bidirectional multi-head attention, no rope, no mask
    const h = layerNorm(x, l1w, l1b, EPSV);
    const q = cont(permute(reshape3(biasAdd(mulMat(qw, h), qb), HDV, NHV, L), 0, 2, 1, 3));
    const k = cont(permute(reshape3(biasAdd(mulMat(kw, h), kb), HDV, NHV, L), 0, 2, 1, 3));
    const v = cont(permute(reshape3(biasAdd(mulMat(vw, h), vb), HDV, NHV, L), 1, 2, 0, 3));
    const scores = softMaxExt(mulMat(k, q), 1.0 / 8.0);
    const av = cont(permute(mulMat(v, scores), 0, 2, 1, 3));
    const h1 = add(x, biasAdd(mulMat(ow, reshape2(av, DV, L)), ob));

    // pre-norm MLP, tanh-approximated GELU
    const h2 = layerNorm(h1, l2w, l2b, EPSV);
    const up = gelu(biasAdd(mulMat(uw, h2), ub));
    return add(h1, biasAdd(mulMat(dw, up), db));
}

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: vision <mmproj.gguf> <case_dir> <cpu|vulkan> [out_img_feats.f32]\n", .{});
        return error.BadArgs;
    }
    const gguf_path = args[1];
    const cdir = args[2];
    const backend_name: []const u8 = args[3];

    var pbuf: [512]u8 = undefined;
    var zbuf: [512]u8 = undefined;
    @memcpy(zbuf[0..gguf_path.len], gguf_path);
    zbuf[gguf_path.len] = 0;

    var data_ctx: ?*Ctx = null;
    const gp = c.gguf_init_params{ .no_alloc = false, .ctx = &data_ctx };
    if (c.gguf_init_from_file(&zbuf, gp) == null) return error.BadGguf;
    if (data_ctx == null) return error.BadGguf;
    const gc = data_ctx.?;

    const pix_bytes = readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/pixels.f32", .{cdir})) catch
        try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/../pixels.f32", .{cdir}));
    const npix = pix_bytes.len / 4;
    if (npix != @as(usize, @intCast(3 * IMG * IMG))) {
        std.debug.print("pixels.f32 holds {d} floats, expected {d}\n", .{ npix, 3 * IMG * IMG });
        return error.BadPixels;
    }

    const params = c.ggml_init_params{ .mem_size = CTX_BYTES, .mem_buffer = null, .no_alloc = true };
    ctx = @ptrCast(c.ggml_init(params));
    pending = .empty;

    backend_g = if (std.mem.eql(u8, backend_name, "vulkan"))
        c.ggml_backend_vk_init(0)
    else
        c.ggml_backend_cpu_init();
    if (backend_g == null) return error.NoBackend;

    std.debug.print("vision: backend={s}\n", .{backend_name});

    // host-side im2col: column p is the patch at (py, px), row j is (c, h, w)
    const pat = newT2(PATCH * PATCH * 3, NPATCH);
    const patch_buf = try alloc.alloc(f32, @intCast(PATCH * PATCH * 3 * NPATCH));
    defer alloc.free(patch_buf);
    {
        const px: [*]const f32 = @ptrCast(@alignCast(pix_bytes.ptr));
        const plane = @as(usize, @intCast(IMG * IMG));
        var p: usize = 0;
        while (p < @as(usize, @intCast(NPATCH))) : (p += 1) {
            const gy: usize = p / @as(usize, @intCast(GRID));
            const gx: usize = p % @as(usize, @intCast(GRID));
            var j: usize = 0;
            for (0..3) |ch| {
                for (0..@as(usize, @intCast(PATCH))) |hy| {
                    for (0..@as(usize, @intCast(PATCH))) |wx| {
                        const y = gy * @as(usize, @intCast(PATCH)) + hy;
                        const x = gx * @as(usize, @intCast(PATCH)) + wx;
                        patch_buf[p * @as(usize, @intCast(PATCH * PATCH * 3)) + j] = px[ch * plane + y * @as(usize, @intCast(IMG)) + x];
                        j += 1;
                    }
                }
            }
        }
    }
    try pending.append(alloc, .{ .t = pat, .src = @ptrCast(patch_buf.ptr), .len = patch_buf.len * 4 });

    const pw = try weight(alloc, gc, "v.patch_embd.weight", PATCH * PATCH * 3, DV);
    const pb = try weight(alloc, gc, "v.patch_embd.bias", null, null);
    const pos = try weight(alloc, gc, "v.position_embd.weight", null, null);
    const plw = try weight(alloc, gc, "v.post_ln.weight", null, null);
    const plb = try weight(alloc, gc, "v.post_ln.bias", null, null);
    const fcw = try weight(alloc, gc, "mm.model.fc.weight", null, null);

    var h = add(biasAdd(mulMat(pw, pat), pb), pos);
    for (0..NLV) |i| h = try block(alloc, gc, i, h, NPATCH);
    const hn = layerNorm(h, plw, plb, EPSV);

    // Idefics3 pixel shuffle: NOT a plain reshape. HF does a real 2-D space-to-depth --
    // 32x32 patches of 768 become 8x8 tokens of 12288, where each token holds a 4x4 block
    // of patches, so the sub-patch index interleaves rows and columns.
    //
    //   out[t][h2*3072 + w2*768 + e] = in[(h1*4+h2)*32 + (w1*4+w2)][e],  t = h1*8 + w1
    //
    // ggml is ne0-fastest while torch is last-dim-fastest, so the steps are not the same
    // sequence HF writes; this is the ggml equivalent, derived from flat-index equality:
    //   [768, 1024]                     flat = e + w2*768 + w1*3072  + h2*24576 + h1*98304
    //   -> [3072, 8, 4, 8]                (split p into w2|w1 and h into h2|h1)
    //   -> permute(0,2,1,3) -> [3072, 4, 8, 8]   flat = c + w1*12288 + h1*98304
    //   -> [12288, 64]                   flat = c + t*12288   with c = h2*3072 + w2*768 + e
    const shuffled = reshape2(cont(permute(reshape4(hn, DV * 4, GRID / 4, 4, GRID / 4), 0, 2, 1, 3)), CONN_IN, NTOK);
    const feats = mulMat(fcw, shuffled); // [576, 64]

    const gf = c.ggml_new_graph(ctx).?;
    c.ggml_build_forward_expand(gf, feats);
    uploadAll();
    if (c.ggml_backend_graph_compute(backend_g, gf) != c.GGML_STATUS_SUCCESS) return error.ComputeFailed;

    const got = try alloc.alloc(f32, @intCast(CONN_OUT * NTOK));
    defer alloc.free(got);
    c.ggml_backend_tensor_get(feats, got.ptr, 0, @intCast(CONN_OUT * NTOK * 4));

    if (args.len == 5) {
        const f = try std.Io.Dir.cwd().createFile(io_g, args[4], .{});
        defer f.close(io_g);
        var wbuf: [4096]u8 = undefined;
        var fw = f.writer(io_g, &wbuf);
        try fw.interface.writeAll(std.mem.sliceAsBytes(got));
        try fw.interface.flush();
        std.debug.print("  wrote img_feats -> {s}\n", .{args[4]});
    }

    const ref_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/img_feats.f32", .{cdir}));
    defer alloc.free(ref_bytes);
    const ref: [*]const f32 = @ptrCast(@alignCast(ref_bytes.ptr));

    var max_abs: f32 = 0;
    // relative error only where |ref| is big enough for a ratio to mean anything
    var max_rel: f32 = 0;
    var sum_abs: f64 = 0;
    for (0..got.len) |i| {
        const d = @abs(got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += @abs(ref[i]);
        if (@abs(ref[i]) > 1.0) {
            const r = d / @abs(ref[i]);
            if (r > max_rel) max_rel = r;
        }
    }

    std.debug.print("  img_feats: {d} values, mean |ref| = {d:.4}\n", .{ got.len, sum_abs / @as(f64, @floatFromInt(got.len)) });
    std.debug.print("  first 4 got: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ got[0], got[1], got[2], got[3] });
    std.debug.print("  first 4 ref: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ ref[0], ref[1], ref[2], ref[3] });
    std.debug.print("  max |diff| = {e}   max rel where |ref|>1 = {e}\n", .{ max_abs, max_rel });

    // Absolute error is the wrong gate here: reference values run to ~50, so a 1e-2
    // absolute bound is really 2e-4 relative. Judge the ratio.
    if (max_rel < 1e-2) {
        std.debug.print("  RESULT: PASS\n", .{});
    } else {
        std.debug.print("  RESULT: FAIL\n", .{});
        return error.Mismatch;
    }
}
