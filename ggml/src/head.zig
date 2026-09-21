//! laya-vision decision head on ggml, in Zig.
//!
//! The head is the 36 tensors the SmolVLM backbone does not provide:
//!
//!     h = enc_h + type_emb[qtype]                       [576, L]
//!     h = 2 x TransformerEncoderLayer(576, 9 heads, 2304 ffn, ReLU, norm_first)
//!     m = h[marker_pos]                                 [576, k]
//!     logits = scorer(m)                                [k]
//!     p = softmax(logits / temperature[qtype])
//!
//! All stock ggml ops, no custom kernels. The encoder is not our business here;
//! this program takes its output (enc_h) as input and is verified against the
//! Python oracle byte for byte.
//!
//! Bindings come from translate-c over the real headers (see build.zig), so the
//! struct layouts are the compiler's, not hand-written guesses.
//!
//! Weight layout: an HF Linear stores [out, in] row-major, which ggml reads as
//! ne0=in, ne1=out -- exactly what ggml_mul_mat wants for y = x W^T. No transpose.

const std = @import("std");
const c = @import("ggml");

const T = c.struct_ggml_tensor;
const Ctx = c.struct_ggml_context;

const D: i64 = 576;
const NH: i64 = 9;
const HD: i64 = 64;
const FF: i64 = 2304;
const NL: usize = 2;
const LN_EPS: f32 = 1e-5;
const ATTN_SCALE: f32 = 1.0 / 8.0; // 1/sqrt(head_dim)
const CTX_BYTES: usize = 256 * 1024 * 1024;

// vlm_agent_config.json, indexed by qtype (0=choice, 1=score, 2=noul)
const TEMPERATURE = [3]f32{ 3.1510276794433594, 1.0, 1.5558207035064697 };

var ctx: *Ctx = undefined;
var io_g: std.Io = undefined;

fn asT(p: [*c]T) *T {
    if (p == null) @panic("ggml returned a null tensor");
    return @ptrCast(p);
}

// -- thin typed wrappers over the C API ---------------------------------------
fn newT1(ne0: i64) *T {
    return asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_F32, ne0));
}
fn newT2(ne0: i64, ne1: i64) *T {
    return asT(c.ggml_new_tensor_2d(ctx, c.GGML_TYPE_F32, ne0, ne1));
}
fn view1(a: *T, ne0: i64, off: usize) *T {
    return asT(c.ggml_view_1d(ctx, a, ne0, off));
}
fn view2(a: *T, ne0: i64, ne1: i64, nb1: usize, off: usize) *T {
    return asT(c.ggml_view_2d(ctx, a, ne0, ne1, nb1, off));
}
fn reshape1(a: *T, ne0: i64) *T {
    return asT(c.ggml_reshape_1d(ctx, a, ne0));
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
fn norm(a: *T, eps: f32) *T {
    return asT(c.ggml_norm(ctx, a, eps));
}
fn relu(a: *T) *T {
    return asT(c.ggml_relu(ctx, a));
}
fn geluErf(a: *T) *T {
    return asT(c.ggml_gelu_erf(ctx, a));
}
fn softMax(a: *T) *T {
    return asT(c.ggml_soft_max(ctx, a));
}
fn softMaxExt(a: *T, mask: [*c]T, s: f32, max_bias: f32) *T {
    return asT(c.ggml_soft_max_ext(ctx, a, mask, s, max_bias));
}
fn getRows(a: *T, b: *T) *T {
    return asT(c.ggml_get_rows(ctx, a, b));
}
fn f32p(t: *T) [*]f32 {
    const p = c.ggml_get_data_f32(t);
    return @as([*]f32, @ptrCast(p));
}

// -- io -----------------------------------------------------------------------
fn readFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
}

/// Load one weight blob into a fresh ctx-allocated tensor.
fn loadWeight(alloc: std.mem.Allocator, dir: []const u8, name: []const u8, ne: []const i64) !*T {
    var path_buf: [512]u8 = undefined;
    var nbuf: [160]u8 = undefined;
    var i: usize = 0;
    for (name) |ch| {
        nbuf[i] = if (ch == '.') '_' else ch;
        i += 1;
    }
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}.f32", .{ dir, nbuf[0..i] });

    const bytes = try readFile(alloc, path);
    defer alloc.free(bytes);

    const t = if (ne.len == 1) newT1(ne[0]) else newT2(ne[0], ne[1]);

    var want: usize = 4;
    for (ne) |d| want *= @as(usize, @intCast(d));
    if (bytes.len != want) {
        std.debug.print("weight {s}: {d} bytes, expected {d}\n", .{ path, bytes.len, want });
        return error.BadWeightSize;
    }
    @memcpy(@as([*]u8, @ptrCast(f32p(t)))[0..want], bytes);
    return t;
}

// -- graph --------------------------------------------------------------------
fn layerNorm(x: *T, w: *T, b: *T) *T {
    return add(mul(norm(x, LN_EPS), w), b);
}

fn biasAdd(y: *T, b: *T) *T {
    return add(y, repeat(b, y));
}

fn encoderLayer(alloc: std.mem.Allocator, dir: []const u8, li: usize, x: *T, L: i64) !*T {
    var pbuf: [96]u8 = undefined;
    const F = struct {
        fn n(p: []u8, comptime fmt: []const u8, li_: usize, tail: []const u8) []const u8 {
            _ = fmt;
            return std.fmt.bufPrint(p, "head.layers.{d}.{s}", .{ li_, tail }) catch unreachable;
        }
    };

    const inw = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "self_attn.in_proj_weight"), &[_]i64{ D, 3 * D });
    const inb = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "self_attn.in_proj_bias"), &[_]i64{3 * D});
    const opw = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "self_attn.out_proj.weight"), &[_]i64{ D, D });
    const opb = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "self_attn.out_proj.bias"), &[_]i64{D});
    const n1w = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "norm1.weight"), &[_]i64{D});
    const n1b = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "norm1.bias"), &[_]i64{D});
    const n2w = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "norm2.weight"), &[_]i64{D});
    const n2b = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "norm2.bias"), &[_]i64{D});
    const l1w = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "linear1.weight"), &[_]i64{ D, FF });
    const l1b = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "linear1.bias"), &[_]i64{FF});
    const l2w = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "linear2.weight"), &[_]i64{ FF, D });
    const l2b = try loadWeight(alloc, dir, F.n(&pbuf, "", li, "linear2.bias"), &[_]i64{D});

    // pre-norm self-attention; the three projections are views into one matmul
    const hn = layerNorm(x, n1w, n1b);
    const qkv = biasAdd(mulMat(inw, hn), inb); // [3*576, L]
    // ggml_reshape_3d requires contiguity, and a slice of the packed matmul is not
    // contiguous, so each projection is materialised before the head split.
    const q = cont(view2(qkv, D, L, qkv.nb[1], 0));
    const k = cont(view2(qkv, D, L, qkv.nb[1], D * 4));
    const v = cont(view2(qkv, D, L, qkv.nb[1], 2 * D * 4));

    // [576, L] -> [64, 9, L] -> [64, L, 9]; v -> [L, 64, 9] so mul_mat contracts keys
    // mul_mat also wants contiguous operands, so the permutes are materialised too
    const q3 = cont(permute(reshape3(q, HD, NH, L), 0, 2, 1, 3));
    const k3 = cont(permute(reshape3(k, HD, NH, L), 0, 2, 1, 3));
    // ggml_permute args are destinations: result->ne[axis[i]] = a->ne[i].
    // [HD, NH, L] -> [L, HD, NH] is the 3-cycle (1,2,0,3), not (2,0,1,3).
    const v3 = cont(permute(reshape3(v, HD, NH, L), 1, 2, 0, 3));

    const scores = softMaxExt(mulMat(k3, q3), null, ATTN_SCALE, 0);
    const av = mulMat(v3, scores); // [64, L, 9]
    const flat = reshape2(cont(permute(av, 0, 2, 1, 3)), D, L);

    const h = add(x, biasAdd(mulMat(opw, flat), opb));

    // pre-norm MLP; ReLU is nn.TransformerEncoderLayer's default activation
    const hn2 = layerNorm(h, n2w, n2b);
    const f = relu(biasAdd(mulMat(l1w, hn2), l1b));
    const g = biasAdd(mulMat(l2w, f), l2b);
    return add(h, g);
}

fn buildHead(
    alloc: std.mem.Allocator,
    dir: []const u8,
    enc: *T,
    idx: *T,
    qtype: usize,
    L: i64,
    k: usize,
) !*T {
    const te = try loadWeight(alloc, dir, "type_emb.weight", &[_]i64{ D, 3 });
    const te_row = view2(te, D, 1, te.nb[1], qtype * te.nb[1]);
    var h = add(enc, repeat(te_row, enc));

    for (0..NL) |li| h = try encoderLayer(alloc, dir, li, h, L);

    const m = getRows(h, idx); // [576, k]

    const s0w = try loadWeight(alloc, dir, "scorer.0.weight", &[_]i64{D});
    const s0b = try loadWeight(alloc, dir, "scorer.0.bias", &[_]i64{D});
    const s1w = try loadWeight(alloc, dir, "scorer.1.weight", &[_]i64{ D, D });
    const s1b = try loadWeight(alloc, dir, "scorer.1.bias", &[_]i64{D});
    const s3w = try loadWeight(alloc, dir, "scorer.3.weight", &[_]i64{ D, 1 });
    const s3b = try loadWeight(alloc, dir, "scorer.3.bias", &[_]i64{1});

    const sm = layerNorm(m, s0w, s0b);
    const h1 = biasAdd(mulMat(s1w, sm), s1b);
    const h2 = geluErf(h1);
    const lg = biasAdd(mulMat(s3w, h2), s3b); // [1, k]
    return reshape1(lg, @intCast(k));
}

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) {
        std.debug.print("usage: head <head_blobs_dir> <oracle_case_dir>\n", .{});
        return error.BadArgs;
    }
    const wdir = args[1];
    const cdir = args[2];

    var pbuf: [512]u8 = undefined;

    const enc_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/enc_h.f32", .{cdir}));
    defer alloc.free(enc_bytes);
    const L: i64 = @intCast(enc_bytes.len / (@as(usize, @intCast(D)) * 4));

    const pos_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/marker_pos.i32", .{cdir}));
    defer alloc.free(pos_bytes);
    const k: usize = pos_bytes.len / 4;

    const qt_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/qtype.i32", .{cdir}));
    defer alloc.free(qt_bytes);
    const qtype: usize = @intCast(std.mem.readInt(i32, qt_bytes[0..4], .little));

    std.debug.print("case {s}: L={d} k={d} qtype={d} temp={d:.4}\n", .{ cdir, L, k, qtype, TEMPERATURE[qtype] });

    const params = c.ggml_init_params{ .mem_size = CTX_BYTES, .mem_buffer = null, .no_alloc = false };
    ctx = @ptrCast(c.ggml_init(params));

    const enc = newT2(D, L);
    @memcpy(@as([*]u8, @ptrCast(f32p(enc)))[0..enc_bytes.len], enc_bytes);

    const idx = asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @intCast(k)));
    @memcpy(@as([*]u8, @ptrCast(c.ggml_get_data(idx)))[0..pos_bytes.len], pos_bytes);

    const logits = try buildHead(alloc, wdir, enc, idx, qtype, L, k);
    const probs = softMax(scale(logits, 1.0 / TEMPERATURE[qtype]));

    const gf = c.ggml_new_graph(ctx).?;
    c.ggml_build_forward_expand(gf, logits);
    c.ggml_build_forward_expand(gf, probs);

    const backend = c.ggml_backend_cpu_init();
    if (c.ggml_backend_graph_compute(backend, gf) != c.GGML_STATUS_SUCCESS) {
        std.debug.print("graph compute failed\n", .{});
        return error.ComputeFailed;
    }

    const exp_logits = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/logits.f32", .{cdir}));
    defer alloc.free(exp_logits);
    const exp_probs = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/probs.f32", .{cdir}));
    defer alloc.free(exp_probs);

    const got_logits = f32p(logits)[0..k];
    const got_probs = f32p(probs)[0..k];

    var max_dl: f32 = 0;
    var max_dp: f32 = 0;
    std.debug.print("\n  {s:>4} {s:>14} {s:>14} {s:>12}\n", .{ "idx", "ref prob", "ggml prob", "abs diff" });
    for (0..k) |i| {
        const rl: f32 = @bitCast(std.mem.readInt(u32, exp_logits[i * 4 ..][0..4], .little));
        const rp: f32 = @bitCast(std.mem.readInt(u32, exp_probs[i * 4 ..][0..4], .little));
        const dl = @abs(got_logits[i] - rl);
        const dp = @abs(got_probs[i] - rp);
        if (dl > max_dl) max_dl = dl;
        if (dp > max_dp) max_dp = dp;
        std.debug.print("  {d:>4} {d:>14.6} {d:>14.6} {e:>14}\n", .{ i, rp, got_probs[i], dp });
    }
    std.debug.print("\n  max |dlogit| = {e}   max |dprob| = {e}\n", .{ max_dl, max_dp });
    if (max_dp < 1e-4) {
        std.debug.print("  RESULT: PASS\n", .{});
    } else {
        std.debug.print("  RESULT: FAIL\n", .{});
        return error.Mismatch;
    }
}
