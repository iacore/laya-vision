//! SmolVLM encoder on ggml: image embeddings -> text hidden states (enc_h).
//!
//! This is the half of the model the decision head does not cover. It runs a
//! Llama-30L text stack over a token sequence in which the image positions carry
//! vectors produced by the vision tower:
//!
//!     h = token_embd[input_ids];  h[:, image positions] = img_feats
//!     for 30 layers: pre-norm attention (GQA 9/3, head_dim 64, RoPE) + SwiGLU MLP
//!     enc_h = rms_norm(h, output_norm)                       [576, L]
//!
//! Weights come straight from a GGUF read with ggml's own gguf.h -- no llama.cpp.
//! The forward is verified against a reference dump from the real PyTorch model.
//!
//! Text config, read from the GGUF metadata rather than assumed:
//!   30 layers, d=576, 9 q / 3 kv heads, head_dim 64, ffn 1536, vocab 49280,
//!   rms eps 1e-5, RoPE theta 100000 (NOT the HF default of 10000).
//!
//! RoPE convention: GGML_ROPE_TYPE_NORMAL (interleaved pairs), *not* NEOX. NEOX is what
//! llama.cpp uses for most Llama checkpoints and it is wrong here -- it leaves position 0
//! correct (rope is identity there) and corrupts every later position, so the bug looks
//! like anything but rope. Getting it wrong gives max |diff| ~41; correct gives ~1e-4.
//!
//! Vulkan precision: run with GGML_VK_DISABLE_F16=1. The default f16 path costs about
//! three orders of magnitude over 30 layers (7.1e-2 vs 2.2e-4 against the reference),
//! and it fails the check below. GGML_VK_DISABLE_COOPMAT alone does not help.

const std = @import("std");
const c = @import("ggml");

const T = c.struct_ggml_tensor;
const Ctx = c.struct_ggml_context;

const D: i64 = 576;
const NH: i64 = 9;
const NKV: i64 = 3;
const HD: i64 = 64;
const NL: usize = 30;
const FF: i64 = 1536;
const VOCAB: i64 = 49280;
const N_CTX: i32 = 8192;
const RMS_EPS: f32 = 1e-5;
const ROPE_BASE: f32 = 100000.0;
const IMAGE_TOKEN_ID: i32 = 49190;
// metadata only: tensors are created no_alloc, the data comes from the backend
const CTX_BYTES: usize = 128 * 1024 * 1024;

var ctx: *Ctx = undefined;
var io_g: std.Io = undefined;
var backend_g: ?*c.struct_ggml_backend = null;
/// The backend buffer that owns every tensor:
/// ggml_backend_alloc_ctx_tensors returns it and nothing else frees it, so a
/// process that runs two stages holds both stages' memory to the end. On
/// Vulkan that is device memory, and the second stage then cannot allocate.
var buf_g: ?c.ggml_backend_buffer_t = null;

const Pending = struct { t: *T, src: [*]const u8, len: usize };
var pending: std.ArrayList(Pending) = .empty;

fn asT(p: [*c]T) *T {
    if (p == null) @panic("ggml returned a null tensor");
    return @ptrCast(p);
}

// -- wrappers -----------------------------------------------------------------
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
fn mul(a: *T, b: *T) *T {
    return asT(c.ggml_mul(ctx, a, b));
}
fn repeat(a: *T, b: *T) *T {
    return asT(c.ggml_repeat(ctx, a, b));
}
fn scale(a: *T, s: f32) *T {
    return asT(c.ggml_scale(ctx, a, s));
}
fn silu(a: *T) *T {
    return asT(c.ggml_silu(ctx, a));
}
fn softMaxExt(a: *T, mask: ?*T, s: f32) *T {
    return asT(c.ggml_soft_max_ext(ctx, a, if (mask) |m| m else null, s, 0));
}
fn rmsNorm(a: *T, w: *T, eps: f32) *T {
    return mul(asT(c.ggml_rms_norm(ctx, a, eps)), w);
}
fn ropeInplace(a: *T, pos: *T) *T {
    return asT(c.ggml_rope_ext(ctx, a, pos, null, HD, c.GGML_ROPE_TYPE_NORMAL, N_CTX,
        ROPE_BASE, 1.0, 0.0, 1.0, 32.0, 1.0));
}

// -- io -----------------------------------------------------------------------
fn readFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
}

/// Register a compute-context tensor backed by host bytes; uploaded after allocation.
fn weight(alloc: std.mem.Allocator, gguf_ctx: *Ctx, name: []const u8) !*T {
    var nbuf: [128]u8 = undefined;
    if (name.len >= nbuf.len) return error.NameTooLong;
    @memcpy(nbuf[0..name.len], name);
    nbuf[name.len] = 0;
    const src = c.ggml_get_tensor(gguf_ctx, &nbuf);
    if (src == null) {
        std.debug.print("missing weight: {s}\n", .{name});
        return error.MissingWeight;
    }
    const s: *T = @ptrCast(src);
    const t = newT2(s.ne[0], s.ne[1]);
    try pending.append(alloc, .{
        .t = t,
        .src = @ptrCast(s.data.?),
        .len = @intCast(s.ne[0] * s.ne[1] * 4),
    });
    return t;
}

fn weight1(alloc: std.mem.Allocator, gguf_ctx: *Ctx, name: []const u8) !*T {
    var nbuf: [128]u8 = undefined;
    @memcpy(nbuf[0..name.len], name);
    nbuf[name.len] = 0;
    const src = c.ggml_get_tensor(gguf_ctx, &nbuf);
    if (src == null) return error.MissingWeight;
    const s: *T = @ptrCast(src);
    const t = newT1(s.ne[0]);
    try pending.append(alloc, .{ .t = t, .src = @ptrCast(s.data.?), .len = @intCast(s.ne[0] * 4) });
    return t;
}

fn uploadAll(alloc: std.mem.Allocator) void {
    buf_g = c.ggml_backend_alloc_ctx_tensors(ctx, backend_g);
    if (buf_g == null) @panic("backend could not allocate the graph");
    for (pending.items) |it| c.ggml_backend_tensor_set(it.t, it.src, 0, it.len);
    pending.clearRetainingCapacity();
    _ = alloc;
}

// -- graph --------------------------------------------------------------------
fn layer(alloc: std.mem.Allocator, gc: *Ctx, i: usize, x: *T, pos: *T, mask: *T, L: i64) !*T {
    var nb: [96]u8 = undefined;
    const nm = struct {
        fn f(buf: []u8, li: usize, tail: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "blk.{d}.{s}", .{ li, tail }) catch unreachable;
        }
    };

    const an = try weight1(alloc, gc, nm.f(&nb, i, "attn_norm.weight"));
    const qw = try weight(alloc, gc, nm.f(&nb, i, "attn_q.weight"));
    const kw = try weight(alloc, gc, nm.f(&nb, i, "attn_k.weight"));
    const vw = try weight(alloc, gc, nm.f(&nb, i, "attn_v.weight"));
    const ow = try weight(alloc, gc, nm.f(&nb, i, "attn_output.weight"));
    const fn_ = try weight1(alloc, gc, nm.f(&nb, i, "ffn_norm.weight"));
    const gw = try weight(alloc, gc, nm.f(&nb, i, "ffn_gate.weight"));
    const uw = try weight(alloc, gc, nm.f(&nb, i, "ffn_up.weight"));
    const dw = try weight(alloc, gc, nm.f(&nb, i, "ffn_down.weight"));

    // pre-norm attention
    const h = rmsNorm(x, an, RMS_EPS);
    // [576, L] -> [64, 9, L]; rope runs on the head layout, then we permute to [64, L, 9]
    const q = ropeInplace(reshape3(mulMat(qw, h), HD, NH, L), pos);
    const k = ropeInplace(reshape3(mulMat(kw, h), HD, NKV, L), pos);
    const v = reshape3(mulMat(vw, h), HD, NKV, L);

    const qp = cont(permute(q, 0, 2, 1, 3)); // [64, L, 9]
    const kp = cont(permute(k, 0, 2, 1, 3)); // [64, L, 3]
    const vp = cont(permute(v, 1, 2, 0, 3)); // [L, 64, 3]

    // ggml_mul_mat broadcasts the kv heads: t1->ne2 % t0->ne2 == 0, 9 % 3 == 0
    const kq = softMaxExt(mulMat(kp, qp), mask, 1.0 / 8.0);
    const av = cont(permute(mulMat(vp, kq), 0, 2, 1, 3)); // [64, 9, L]
    const attn = mulMat(ow, reshape2(av, D, L));
    const h1 = add(x, attn);

    // pre-norm SwiGLU
    const hn = rmsNorm(h1, fn_, RMS_EPS);
    const g = silu(mulMat(gw, hn));
    const u = mulMat(uw, hn);
    return add(h1, mulMat(dw, mul(g, u)));
}

/// ids [L] token ids plus [64, 576] image vectors -> enc_h [L, 576], row-major.
/// The caller owns the returned slice.
pub fn encodeText(
    alloc: std.mem.Allocator,
    io: std.Io,
    gguf_path: []const u8,
    ids: []const i32,
    img_feats: []const f32,
    backend_name: []const u8,
) ![]f32 {
    io_g = io;
    const L: i64 = @intCast(ids.len);
    var zbuf: [512]u8 = undefined;
    @memcpy(zbuf[0..gguf_path.len], gguf_path);
    zbuf[gguf_path.len] = 0;

    // 1. weights straight out of the GGUF via ggml's own reader
    var data_ctx: ?*Ctx = null;
    const gp = c.gguf_init_params{ .no_alloc = false, .ctx = &data_ctx };
    // the gguf_context handle is not kept: the tensors it produced are backed by a mapping
    // that stays alive for the process, which is all a short-lived verifier needs
    if (c.gguf_init_from_file(&zbuf, gp) == null) return error.BadGguf;
    if (data_ctx == null) return error.BadGguf;
    const gctx = data_ctx.?;

    const params = c.ggml_init_params{ .mem_size = CTX_BYTES, .mem_buffer = null, .no_alloc = true };
    ctx = @ptrCast(c.ggml_init(params));
    pending = .empty;
    defer pending.deinit(alloc);

    backend_g = if (std.mem.eql(u8, backend_name, "vulkan"))
        c.ggml_backend_vk_init(0)
    else
        c.ggml_backend_cpu_init();
    if (backend_g == null) return error.NoBackend;

    std.debug.print("encoder: L={d} backend={s}\n", .{ L, backend_name });

    // Host-side embedding build: gather rows for the token ids and overwrite the image
    // positions with the vision features, exactly as the HF model does.
    const inp = newT2(D, L);
    const inp_buf = try alloc.alloc(f32, @intCast(D * L));
    defer alloc.free(inp_buf);
    const tbl: [*]const f32 = @ptrCast(@alignCast(ggmlHostPtr(gctx, "token_embd.weight")));
    const feats: [*]const f32 = img_feats.ptr;
    {
        var img_idx: usize = 0;
        const n_img = img_feats.len / @as(usize, @intCast(D));
        for (0..@intCast(L)) |i| {
            const tok = ids[i];
            if (tok == IMAGE_TOKEN_ID and img_idx < n_img) {
                @memcpy(inp_buf[i * @as(usize, @intCast(D)) ..][0..@intCast(D)],
                    feats[img_idx * @as(usize, @intCast(D)) ..][0..@intCast(D)]);
                img_idx += 1;
            } else {
                const base = @as(usize, @intCast(tok)) * @as(usize, @intCast(D));
                @memcpy(inp_buf[i * @as(usize, @intCast(D)) ..][0..@intCast(D)], tbl[base..][0..@intCast(D)]);
            }
        }
        if (img_idx != n_img) {
            std.debug.print("warning: spliced {d} of {d} image vectors\n", .{ img_idx, n_img });
        }
    }
    try pending.append(alloc, .{
        .t = inp,
        .src = @ptrCast(inp_buf.ptr),
        .len = inp_buf.len * 4,
    });

    // positions + causal mask (mask[k, q] = 0 when k <= q)
    const pos = asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, L));
    const pos_buf = try alloc.alloc(i32, @intCast(L));
    defer alloc.free(pos_buf);
    for (0..@intCast(L)) |i| pos_buf[i] = @intCast(i);
    try pending.append(alloc, .{ .t = pos, .src = @ptrCast(pos_buf.ptr), .len = pos_buf.len * 4 });

    const mask = newT2(L, L);
    const mask_buf = try alloc.alloc(f32, @intCast(L * L));
    defer alloc.free(mask_buf);
    for (0..@intCast(L)) |q| {
        for (0..@intCast(L)) |k| {
            mask_buf[k + q * @as(usize, @intCast(L))] = if (k <= q) 0.0 else -std.math.inf(f32);
        }
    }
    try pending.append(alloc, .{ .t = mask, .src = @ptrCast(mask_buf.ptr), .len = mask_buf.len * 4 });

    // 3. 30 layers, then the final norm
    var h = inp;
    for (0..NL) |i| h = try layer(alloc, gctx, i, h, pos, mask, L);
    const on = try weight1(alloc, gctx, "output_norm.weight");
    const enc_h = rmsNorm(h, on, RMS_EPS);

    // 4. compute
    const gf = c.ggml_new_graph(ctx).?;
    c.ggml_build_forward_expand(gf, enc_h);
    uploadAll(alloc);
    if (c.ggml_backend_graph_compute(backend_g, gf) != c.GGML_STATUS_SUCCESS) return error.ComputeFailed;

    const got = try alloc.alloc(f32, @intCast(D * L));
    errdefer alloc.free(got);
    c.ggml_backend_tensor_get(enc_h, got.ptr, 0, @intCast(D * L * 4));

    // release this stage's context and backend so a chained run does not hold every
    // stage's arena at once
    if (buf_g) |b| {
        c.ggml_backend_buffer_free(b);
        buf_g = null;
    }
    c.ggml_backend_free(backend_g);
    backend_g = null;
    c.ggml_free(ctx);
    return got;
}

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: encoder <text.gguf> <case_dir> <cpu|vulkan> [out_enc_h.f32]\n", .{});
        return error.BadArgs;
    }
    const gguf_path = args[1];
    const cdir = args[2];
    const backend_name: []const u8 = args[3];

    var pbuf: [512]u8 = undefined;
    const ids_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/input_ids.i32", .{cdir}));
    defer alloc.free(ids_bytes);
    const n_ids = ids_bytes.len / 4;
    const ids: []const i32 = @as([*]const i32, @ptrCast(@alignCast(ids_bytes.ptr)))[0..n_ids];
    const feats_bytes = readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/img_feats.f32", .{cdir})) catch
        try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/../img_feats.f32", .{cdir}));
    defer alloc.free(feats_bytes);
    const img_feats: []const f32 = @as([*]const f32, @ptrCast(@alignCast(feats_bytes.ptr)))[0..(feats_bytes.len / 4)];

    const got = try encodeText(alloc, io_g, gguf_path, ids, img_feats, backend_name);
    defer alloc.free(got);

    if (args.len == 5) {
        const out_path = args[4];
        std.debug.print("  wrote enc_h -> {s}\n", .{out_path});
        const f = try std.Io.Dir.cwd().createFile(io_g, out_path, .{});
        defer f.close(io_g);
        var wbuf: [4096]u8 = undefined;
        var fw = f.writer(io_g, &wbuf);
        try fw.interface.writeAll(std.mem.sliceAsBytes(got));
        try fw.interface.flush();
    }

    const ref_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/enc_h.f32", .{cdir}));
    defer alloc.free(ref_bytes);
    const ref: [*]const f32 = @ptrCast(@alignCast(ref_bytes.ptr));
    var max_abs: f32 = 0;
    var max_rel: f32 = 0;
    var sum_abs: f64 = 0;
    for (0..got.len) |i| {
        const d = @abs(got[i] - ref[i]);
        if (d > max_abs) max_abs = d;
        const r = d / (@abs(ref[i]) + 1e-6);
        if (r > max_rel and d > 1e-4) max_rel = r;
        sum_abs += @abs(ref[i]);
    }
    std.debug.print("  enc_h: {d} values, mean |ref| = {d:.4}\n", .{ got.len, sum_abs / @as(f64, @floatFromInt(got.len)) });
    std.debug.print("  first 4 got: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ got[0], got[1], got[2], got[3] });
    std.debug.print("  first 4 ref: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ ref[0], ref[1], ref[2], ref[3] });
    std.debug.print("  max |diff| = {e}   max rel (on |d|>1e-4) = {e}\n", .{ max_abs, max_rel });
    if (max_abs < 1e-3) {
        std.debug.print("  RESULT: PASS\n", .{});
    } else {
        std.debug.print("  RESULT: FAIL\n", .{});
        return error.Mismatch;
    }
}

fn ggmlHostPtr(gc: *Ctx, name: []const u8) *const anyopaque {
    var nbuf: [128]u8 = undefined;
    @memcpy(nbuf[0..name.len], name);
    nbuf[name.len] = 0;
    const t: *T = @ptrCast(c.ggml_get_tensor(gc, &nbuf));
    return t.data.?;
}
