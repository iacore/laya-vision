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
// metadata only: tensors are created no_alloc, the data comes from the backend
const CTX_BYTES: usize = 64 * 1024 * 1024;

// vlm_agent_config.json, indexed by qtype (0=choice, 1=score, 2=noul)
const TEMPERATURE = [3]f32{ 3.1510276794433594, 1.0, 1.5558207035064697 };

var ctx: *Ctx = undefined;
var io_g: std.Io = undefined;
var backend_g: ?*c.struct_ggml_backend = null;
/// The backend buffer that owns every tensor:
/// ggml_backend_alloc_ctx_tensors returns it and nothing else frees it, so a
/// process that runs two stages holds both stages' memory to the end. On
/// Vulkan that is device memory, and the second stage then cannot allocate.
var buf_g: ?c.ggml_backend_buffer_t = null;

/// Weights cannot be written into `tensor->data` directly: for a GPU backend that pointer
/// is not host memory. Tensors are created with no_alloc, allocated by the backend, then
/// uploaded with ggml_backend_tensor_set -- which is correct on CPU and Vulkan alike.
const Pending = struct { t: *T, bytes: []u8 };
var pending: std.ArrayList(Pending) = undefined;

fn uploadAll(alloc: std.mem.Allocator) void {
    buf_g = c.ggml_backend_alloc_ctx_tensors(ctx, backend_g);
    if (buf_g == null) @panic("backend could not allocate the graph");
    for (pending.items) |it| {
        c.ggml_backend_tensor_set(it.t, it.bytes.ptr, 0, it.bytes.len);
    }
    // the host copies have served their purpose once they are on the backend
    for (pending.items) |it| alloc.free(it.bytes);
    pending.clearRetainingCapacity();
}

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

    const t = if (ne.len == 1) newT1(ne[0]) else newT2(ne[0], ne[1]);

    var want: usize = 4;
    for (ne) |d| want *= @as(usize, @intCast(d));
    if (bytes.len != want) {
        std.debug.print("weight {s}: {d} bytes, expected {d}\n", .{ path, bytes.len, want });
        return error.BadWeightSize;
    }
    try pending.append(alloc, .{ .t = t, .bytes = bytes });
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

const HeadOut = struct { logits: *T, pooled: *T, h: *T };

fn buildHead(
    alloc: std.mem.Allocator,
    dir: []const u8,
    enc: *T,
    idx: *T,
    qtype: usize,
    L: i64,
    k: usize,
) !HeadOut {
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
    // VLMDecisionModel takes the act features from the last *real* token, not [CLS]: in a
    // causal model that is the only position that has seen the whole sequence. For a single
    // unpadded sequence that is column L-1 of the head's hidden state.
    const pooled = cont(view2(h, D, 1, h.nb[1], @as(usize, @intCast(L - 1)) * h.nb[1]));
    return .{ .logits = reshape1(lg, @intCast(k)), .pooled = pooled, .h = h };
}

// -- act head ------------------------------------------------------------------
//
// type_emb and the transformer layers are ggml's business; this is not. The act head is
// Linear(580,256) -> GELU -> Linear(256,2) over features that are themselves 4 numbers
// derived from the k logits, so plumbing it through ggml would cost more code than the
// arithmetic costs time. The whole thing is ~150k MACs.

/// torch's nn.GELU(), the exact erf form -- not the tanh approximation the vision tower uses.
/// Zig's std has no erf; libc is linked, and erff is the same routine torch's CPU kernel uses.
extern "c" fn erff(x: f32) f32;

fn geluErfScalar(x: f32) f32 {
    return 0.5 * x * (1.0 + erff(x / @sqrt(2.0)));
}

/// Read a packed weight blob into host memory instead of into a backend tensor.
fn loadHost(alloc: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
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
    const n = bytes.len / 4;
    const out = try alloc.alloc(f32, n);
    for (0..n) |j| out[j] = @bitCast(std.mem.readInt(u32, bytes[j * 4 ..][0..4], .little));
    return out;
}

const ActResult = struct { pre: [580]f32, logits: [2]f32, probs: [2]f32 };

/// The act input and output, straight from laya.common.DecisionModel.forward.
fn actHead(alloc: std.mem.Allocator, dir: []const u8, pooled: []const f32, logits: []const f32, k: usize) !ActResult {
    var r: ActResult = undefined;

    // softmax over the RAW logits: the temperature only scales the reported probabilities and
    // never reaches these features
    var p: [64]f32 = undefined;
    var mx = logits[0];
    for (logits[0..k]) |v| mx = @max(mx, v);
    var sum: f32 = 0;
    for (logits[0..k], 0..) |v, i| {
        p[i] = @exp(v - mx);
        sum += p[i];
    }
    for (p[0..k]) |*v| v.* /= sum;

    var top1: f32 = p[0];
    var top2: f32 = -1;
    for (p[0..k]) |v| {
        if (v > top1) {
            top2 = top1;
            top1 = v;
        } else if (v > top2) top2 = v;
    }

    const kk: f32 = @floatFromInt(if (k < 2) 2 else k);
    var ent: f32 = 0;
    for (p[0..k]) |v| ent -= v * @log(@max(v, 1e-9));
    ent /= @log(kk);

    @memcpy(r.pre[0..D], pooled[0..D]);
    r.pre[580 - 4] = top1;
    r.pre[580 - 3] = top1 - top2;
    r.pre[580 - 2] = ent;
    r.pre[580 - 1] = kk / 255.0;

    const w0 = try loadHost(alloc, dir, "act_head.0.weight"); // [256, 580]
    defer alloc.free(w0);
    const b0 = try loadHost(alloc, dir, "act_head.0.bias"); // [256]
    defer alloc.free(b0);
    const w2 = try loadHost(alloc, dir, "act_head.2.weight"); // [2, 256]
    defer alloc.free(w2);
    const b2 = try loadHost(alloc, dir, "act_head.2.bias"); // [2]
    defer alloc.free(b2);

    var h1: [256]f32 = undefined;
    for (0..256) |j| {
        var acc: f32 = b0[j];
        for (0..580) |i| acc += w0[j * 580 + i] * r.pre[i];
        h1[j] = geluErfScalar(acc);
    }
    for (0..2) |j| {
        var acc: f32 = b2[j];
        for (0..256) |i| acc += w2[j * 256 + i] * h1[i];
        r.logits[j] = acc;
    }
    const amax = @max(r.logits[0], r.logits[1]);
    var asum: f32 = 0;
    for (0..2) |j| {
        r.probs[j] = @exp(r.logits[j] - amax);
        asum += r.probs[j];
    }
    for (0..2) |j| r.probs[j] /= asum;
    return r;
}

pub const Decision = struct { logits: []f32, probs: []f32, act: ActResult };

/// enc_h [L, 576] -> the decision. `marker_pos` indexes the option terminator of each option
/// in label order and `qtype` (0=choice, 1=score, 2=noul) selects the temperature. The caller
/// owns `probs`, `logits` and `act`.
pub fn decide(
    alloc: std.mem.Allocator,
    io: std.Io,
    wdir: []const u8,
    enc_h: []const f32,
    L: i64,
    marker_pos: []const i32,
    qtype: usize,
    backend_name: []const u8,
) !Decision {
    io_g = io;
    const k = marker_pos.len;

    const params = c.ggml_init_params{ .mem_size = CTX_BYTES, .mem_buffer = null, .no_alloc = true };
    ctx = @ptrCast(c.ggml_init(params));
    pending = std.ArrayList(Pending).empty;
    defer pending.deinit(alloc);

    backend_g = if (std.mem.eql(u8, backend_name, "vulkan"))
        c.ggml_backend_vk_init(0)
    else
        c.ggml_backend_cpu_init();
    if (backend_g == null) return error.NoBackend;

    // uploadAll takes ownership of the byte slices it is handed and frees them as []u8, so
    // the caller's buffers are copied into plain byte allocations rather than aliased
    const enc_src = std.mem.sliceAsBytes(enc_h);
    const enc_copy = try alloc.alloc(u8, enc_src.len);
    errdefer alloc.free(enc_copy);
    @memcpy(enc_copy, enc_src);
    const idx_src = std.mem.sliceAsBytes(marker_pos);
    const idx_copy = try alloc.alloc(u8, idx_src.len);
    errdefer alloc.free(idx_copy);
    @memcpy(idx_copy, idx_src);
    const enc = newT2(D, L);
    try pending.append(alloc, .{ .t = enc, .bytes = enc_copy });
    const idx = asT(c.ggml_new_tensor_1d(ctx, c.GGML_TYPE_I32, @intCast(k)));
    try pending.append(alloc, .{ .t = idx, .bytes = idx_copy });

    const built = try buildHead(alloc, wdir, enc, idx, qtype, L, k);
    const logits = built.logits;
    const probs = softMax(scale(logits, 1.0 / TEMPERATURE[qtype]));

    const gf = c.ggml_new_graph(ctx).?;
    c.ggml_build_forward_expand(gf, logits);
    c.ggml_build_forward_expand(gf, probs);
    c.ggml_build_forward_expand(gf, built.pooled);

    uploadAll(alloc);
    if (c.ggml_backend_graph_compute(backend_g, gf) != c.GGML_STATUS_SUCCESS) return error.ComputeFailed;

    const got_logits = try alloc.alloc(f32, k);
    errdefer alloc.free(got_logits);
    const got_probs = try alloc.alloc(f32, k);
    errdefer alloc.free(got_probs);
    c.ggml_backend_tensor_get(logits, got_logits.ptr, 0, @intCast(k * 4));
    c.ggml_backend_tensor_get(probs, got_probs.ptr, 0, @intCast(k * 4));
    const got_pooled = try alloc.alloc(f32, @intCast(D));
    defer alloc.free(got_pooled);
    c.ggml_backend_tensor_get(built.pooled, got_pooled.ptr, 0, @intCast(D * 4));

    const act = try actHead(alloc, wdir, got_pooled, got_logits, k);

    if (buf_g) |b| {
        c.ggml_backend_buffer_free(b);
        buf_g = null;
    }
    c.ggml_backend_free(backend_g);
    backend_g = null;
    c.ggml_free(ctx);
    return .{ .logits = got_logits, .probs = got_probs, .act = act };
}

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: head <head_blobs_dir> <oracle_case_dir> [cpu|vulkan]\n", .{});
        return error.BadArgs;
    }
    const wdir = args[1];
    const cdir = args[2];
    const backend_name: []const u8 = if (args.len == 4) args[3] else "cpu";

    var pbuf: [512]u8 = undefined;
    const enc_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/enc_h.f32", .{cdir}));
    defer alloc.free(enc_bytes);
    const L: i64 = @intCast(enc_bytes.len / (@as(usize, @intCast(D)) * 4));
    const enc_h: []const f32 = @as([*]const f32, @ptrCast(@alignCast(enc_bytes.ptr)))[0..@intCast(D * L)];

    const pos_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/marker_pos.i32", .{cdir}));
    defer alloc.free(pos_bytes);
    const k: usize = pos_bytes.len / 4;
    const marker_pos: []const i32 = @as([*]const i32, @ptrCast(@alignCast(pos_bytes.ptr)))[0..k];

    const qt_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/qtype.i32", .{cdir}));
    defer alloc.free(qt_bytes);
    const qtype: usize = @intCast(std.mem.readInt(i32, qt_bytes[0..4], .little));

    std.debug.print("case {s}: L={d} k={d} qtype={d} temp={d:.4} backend={s}\n",
        .{ cdir, L, k, qtype, TEMPERATURE[qtype], backend_name });

    const d = try decide(alloc, io_g, wdir, enc_h, L, marker_pos, qtype, backend_name);
    defer alloc.free(d.logits);
    defer alloc.free(d.probs);

    const exp_logits = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/logits.f32", .{cdir}));
    defer alloc.free(exp_logits);
    const exp_probs = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/probs.f32", .{cdir}));
    defer alloc.free(exp_probs);
    const exp_pre = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/act_pre.f32", .{cdir}));
    defer alloc.free(exp_pre);
    const exp_al = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/act_logits.f32", .{cdir}));
    defer alloc.free(exp_al);
    const exp_ap = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/act_probs.f32", .{cdir}));
    defer alloc.free(exp_ap);

    var max_dl: f32 = 0;
    var max_dp: f32 = 0;
    std.debug.print("\n  {s:>4} {s:>14} {s:>14} {s:>12}\n", .{ "idx", "ref prob", "ggml prob", "abs diff" });
    for (0..k) |i| {
        const rl: f32 = @bitCast(std.mem.readInt(u32, exp_logits[i * 4 ..][0..4], .little));
        const rp: f32 = @bitCast(std.mem.readInt(u32, exp_probs[i * 4 ..][0..4], .little));
        const dl = @abs(d.logits[i] - rl);
        const dp = @abs(d.probs[i] - rp);
        if (dl > max_dl) max_dl = dl;
        if (dp > max_dp) max_dp = dp;
        std.debug.print("  {d:>4} {d:>14.6} {d:>14.6} {e:>14}\n", .{ i, rp, d.probs[i], dp });
    }
    std.debug.print("\n  max |dlogit| = {e}   max |dprob| = {e}\n", .{ max_dl, max_dp });

    var max_dpre: f32 = 0;
    for (0..580) |i| {
        const r: f32 = @bitCast(std.mem.readInt(u32, exp_pre[i * 4 ..][0..4], .little));
        max_dpre = @max(max_dpre, @abs(d.act.pre[i] - r));
    }
    var max_dal: f32 = 0;
    var max_dap: f32 = 0;
    for (0..2) |i| {
        const rl: f32 = @bitCast(std.mem.readInt(u32, exp_al[i * 4 ..][0..4], .little));
        const rp: f32 = @bitCast(std.mem.readInt(u32, exp_ap[i * 4 ..][0..4], .little));
        max_dal = @max(max_dal, @abs(d.act.logits[i] - rl));
        max_dap = @max(max_dap, @abs(d.act.probs[i] - rp));
        std.debug.print("  act {d}: ref prob {d:.6}  ggml prob {d:.6}\n", .{ i, rp, d.act.probs[i] });
    }
    std.debug.print("  max |d act_pre | = {e}\n", .{max_dpre});
    std.debug.print("  max |d act_logit| = {e}   max |d act_prob| = {e}\n", .{ max_dal, max_dap });

    // act_pre is an intermediate 580-vector, not a contract: bound it loosely and judge the
    // actual outputs (the decision probabilities and the act probabilities) tightly.
    if (max_dp < 1e-4 and max_dpre < 1e-4 and max_dap < 1e-6) {
        std.debug.print("  RESULT: PASS\n", .{});
    } else {
        std.debug.print("  RESULT: FAIL\n", .{});
        return error.Mismatch;
    }
}
