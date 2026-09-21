//! laya-vision end to end, in one process: a PNG and a question in, decision probabilities out.
//!
//!     png -> preprocess -> pixels        [3, 512, 512]
//!     pixels -> vision tower + connector -> img_feats [64, 576]     (once per image)
//!     question -> layout + tokenizer -> input_ids
//!     input_ids + img_feats -> encoder -> enc_h [L, 576]
//!     enc_h + marker_pos -> head -> logits, probs, act
//!
//! Everything the three verifiers do separately, chained without a file in between and without
//! Python. The layout is `laya/vlm.py`'s `build_vlm_inputs` transcribed; the exact prompt text
//! and budget arithmetic are what the oracle ids were produced from.
//!
//! Usage:
//!   laya <model_dir> <text.gguf> <mmproj.gguf> <head_blobs> <image.png> <cpu|vulkan> [case]
//!
//! With no `case` it runs all three fixture questions and compares against the oracle dumps in
//! the image's own directory, which is the acceptance test. With a case name it runs that one
//! and prints the answer.

const std = @import("std");
const vision = @import("vision.zig");
const encoder = @import("encoder.zig");
const head_mod = @import("head.zig");
const tokenizer = @import("tokenizer.zig");
const preprocess = @import("preprocess.zig");

var io_g: std.Io = undefined;

// -- the layout, from laya/vlm.py -------------------------------------------------------------

const PREFIX_TEXT = "<|im_start|>User:";
const QUESTION_TEXT = "\n{s} question: {s}<end_of_utterance>\nAssistant: Options:\n";
const OPTION_BULLET = "- ";
const OPTION_END = "\n";
const FAKE = "<fake_token_around_image>";
const GLOB = "<global-img>";
const IMAGE = "<image>";

const IMAGE_SIZE: usize = 512;
const IMAGE_SEQ_LEN: usize = 64;
const HEAD_MAX_LEN: usize = 256;
const MAX_LEN: usize = 1024;
/// Exactly what laya.vlm.serialize_state produces for the fixture state.
const STATE_TEXT = "{\"note\": \"synthetic fixture, deterministic\"}";

const QT_CHOICE: u32 = 0;
const QT_SCORE: u32 = 1;
const QT_NOUL: u32 = 2;

const Case = struct {
    name: []const u8,
    type_name: []const u8,
    qtype: u32,
    instructions: []const u8,
    opts: []const []const u8,
};

/// render_options applied by hand: choice renders "key: desc", score "level i: c", noul a
/// fixed false/true pair. Same strings laya.common.render_options returns.
const CASES = [_]Case{
    .{
        .name = "choice",
        .type_name = "choice",
        .qtype = QT_CHOICE,
        .instructions = "What setting does the image show?",
        .opts = &.{
            "indoor: Indoors",
            "outdoor: Outdoors",
            "unknown: Unclear",
            "abstract: An abstract pattern",
        },
    },
    .{
        .name = "noul",
        .type_name = "noul",
        .qtype = QT_NOUL,
        .instructions = "Is there a bright rectangular block in the image?",
        .opts = &.{
            "false: no, the statement does not hold",
            "true: yes, the statement holds",
        },
    },
    .{
        .name = "score",
        .type_name = "score",
        .qtype = QT_SCORE,
        .instructions = "How much readable text is in the image?",
        .opts = &.{ "level 0: None", "level 1: A few words", "level 2: Many words" },
    },
};

const Sequence = struct {
    ids: []u32,
    markers: []i32,
};

fn readFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
}

fn replaceAll(alloc: std.mem.Allocator, s: []const u8, from: []const u8, to: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var i: usize = 0;
    while (i < s.len) {
        if (from.len > 0 and std.mem.startsWith(u8, s[i..], from)) {
            try out.appendSlice(alloc, to);
            i += from.len;
        } else {
            try out.append(alloc, s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(alloc);
}

/// build_vlm_inputs, specialised to this project's config: one image, identity option order.
fn buildSequence(alloc: std.mem.Allocator, tok: *const tokenizer.Tokenizer, kase: Case) !Sequence {
    const enc = struct {
        fn f(t: *const tokenizer.Tokenizer, a: std.mem.Allocator, s: []const u8) ![]u32 {
            return t.encode(a, s);
        }
    }.f;

    // image expansion: <fake><global-img><image> x image_seq_len <fake>
    var prefix_buf: std.ArrayList(u8) = .empty;
    defer prefix_buf.deinit(alloc);
    try prefix_buf.appendSlice(alloc, PREFIX_TEXT);
    try prefix_buf.appendSlice(alloc, FAKE);
    try prefix_buf.appendSlice(alloc, GLOB);
    for (0..IMAGE_SEQ_LEN) |_| try prefix_buf.appendSlice(alloc, IMAGE);
    try prefix_buf.appendSlice(alloc, FAKE);
    const prefix_ids = try enc(tok, alloc, prefix_buf.items);
    defer alloc.free(prefix_ids);

    const end_ids = try enc(tok, alloc, OPTION_END);
    defer alloc.free(end_ids);
    if (end_ids.len != 1) {
        std.debug.print("option terminator is {d} tokens, expected 1\n", .{end_ids.len});
        return error.BadTerminator;
    }
    const end_id = end_ids[0];

    // option lines, each truncated to 48 tokens
    const opt_ids = try alloc.alloc([]u32, kase.opts.len);
    defer alloc.free(opt_ids);
    for (kase.opts, 0..) |o, i| {
        const cleaned = try replaceAll(alloc, o, OPTION_END, " ");
        defer alloc.free(cleaned);
        const with_bullet = try std.fmt.allocPrint(alloc, "{s}{s}", .{ OPTION_BULLET, cleaned });
        defer alloc.free(with_bullet);
        const full = try enc(tok, alloc, with_bullet);
        defer alloc.free(full);
        opt_ids[i] = try alloc.dupe(u32, full[0..@min(full.len, 48)]);
    }
    defer for (opt_ids) |o| alloc.free(o);

    // the question text, with any literal end-of-utterance stripped so it cannot forge a boundary
    const ins_clean = try replaceAll(alloc, kase.instructions, "<end_of_utterance>", " ");
    defer alloc.free(ins_clean);
    const qtext = try std.fmt.allocPrint(alloc, QUESTION_TEXT, .{ kase.type_name, ins_clean });
    defer alloc.free(qtext);
    var head_ids = try enc(tok, alloc, qtext);
    defer alloc.free(head_ids);

    // budget: the options plus the question have to fit head_max_len
    var opt_budget: i64 = @intCast(HEAD_MAX_LEN);
    for (opt_ids) |o| opt_budget -= @as(i64, @intCast(o.len)) + 1;
    if (opt_budget < 16) {
        const per_i: usize = @intCast(@max(4, @divTrunc(@as(i64, HEAD_MAX_LEN) - 16, @as(i64, @intCast(opt_ids.len))) - 1));
        for (opt_ids) |*o| {
            if (o.*.len > per_i) o.* = o.*[0..per_i];
        }
        opt_budget = @intCast(HEAD_MAX_LEN);
        for (opt_ids) |o| opt_budget -= @as(i64, @intCast(o.len)) + 1;
    }
    const keep: usize = @intCast(@max(8, opt_budget));
    if (head_ids.len > keep) {
        // keep the tail -- it carries the "Assistant: Options:" cue
        const head_half = keep / 2;
        @memcpy(head_ids[0..head_half], head_ids[0..head_half]);
        const tail_part = head_ids[head_ids.len - (keep - head_half) ..];
        @memcpy(head_ids[head_half .. head_half + (keep - head_half)], tail_part);
        head_ids = head_ids[0..keep];
    }

    const state_ids = try enc(tok, alloc, STATE_TEXT);
    defer alloc.free(state_ids);

    var tail_len: usize = head_ids.len;
    for (opt_ids) |o| tail_len += o.len + 1;
    const room: usize = if (MAX_LEN > prefix_ids.len + tail_len) MAX_LEN - prefix_ids.len - tail_len else 0;
    const st_len = @min(room, state_ids.len);

    const ids = try alloc.alloc(u32, prefix_ids.len + st_len + tail_len);
    const markers = try alloc.alloc(i32, opt_ids.len);
    @memcpy(ids[0..prefix_ids.len], prefix_ids);
    @memcpy(ids[prefix_ids.len .. prefix_ids.len + st_len], state_ids[0..st_len]);
    var w: usize = prefix_ids.len + st_len;
    @memcpy(ids[w .. w + head_ids.len], head_ids);
    w += head_ids.len;
    for (opt_ids, 0..) |o, i| {
        @memcpy(ids[w .. w + o.len], o);
        w += o.len;
        ids[w] = end_id;
        markers[i] = @intCast(w);
        w += 1;
    }
    return .{ .ids = ids, .markers = markers };
}

// -- one case ---------------------------------------------------------------------------------

const Result = struct { probs: []f32, act: [2]f32 };

fn runCase(
    alloc: std.mem.Allocator,
    tok: *const tokenizer.Tokenizer,
    text_gguf: []const u8,
    head_blobs: []const u8,
    img_feats: []const f32,
    kase: Case,
    backend: []const u8,
    base: []const u8,
) !Result {
    const seq = try buildSequence(alloc, tok, kase);
    defer alloc.free(seq.ids);
    defer alloc.free(seq.markers);

    // check the sequence against the ids the real model ran on before blaming anything else
    const ids_path = try std.fmt.allocPrint(alloc, "{s}/{s}/input_ids.i32", .{ base, kase.name });
    defer alloc.free(ids_path);
    if (readFile(alloc, ids_path)) |ib| {
        defer alloc.free(ib);
        const nref = ib.len / 4;
        var nmis: usize = 0;
        var firstmis: ?usize = null;
        for (0..@min(nref, seq.ids.len)) |i| {
            const r = std.mem.readInt(i32, ib[i * 4 ..][0..4], .little);
            if (r != @as(i32, @intCast(seq.ids[i]))) {
                nmis += 1;
                if (firstmis == null) firstmis = i;
            }
        }
        std.debug.print("  {s}: ids got {d} ref {d} mismatched {d}", .{ kase.name, seq.ids.len, nref, nmis });
        if (firstmis) |i| {
            std.debug.print("  first at {d}: got {d} want {d}", .{ i, seq.ids[i], std.mem.readInt(i32, ib[i * 4 ..][0..4], .little) });
        }
        std.debug.print("\n", .{});
    } else |_| {}

    const pos_path = try std.fmt.allocPrint(alloc, "{s}/{s}/marker_pos.i32", .{ base, kase.name });
    defer alloc.free(pos_path);
    if (readFile(alloc, pos_path)) |pb| {
        defer alloc.free(pb);
        const nref = pb.len / 4;
        var nmis: usize = 0;
        for (0..@min(nref, seq.markers.len)) |i| {
            if (std.mem.readInt(i32, pb[i * 4 ..][0..4], .little) != seq.markers[i]) nmis += 1;
        }
        std.debug.print("  {s}: markers got {any} ref ", .{ kase.name, seq.markers });
        for (0..nref) |i| std.debug.print("{d} ", .{std.mem.readInt(i32, pb[i * 4 ..][0..4], .little)});
        std.debug.print(" (mismatched {d})\n", .{nmis});
    } else |_| {}

    const ids_i32 = try alloc.alloc(i32, seq.ids.len);
    defer alloc.free(ids_i32);
    for (seq.ids, 0..) |v, i| ids_i32[i] = @intCast(v);

    const enc_h = try encoder.encodeText(alloc, io_g, text_gguf, ids_i32, img_feats, backend);
    defer alloc.free(enc_h);

    const d = try head_mod.decide(alloc, io_g, head_blobs, enc_h, @intCast(ids_i32.len), seq.markers, kase.qtype, backend);
    defer alloc.free(d.logits);
    return .{ .probs = d.probs, .act = d.act.probs };
}

fn readF32(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return readFile(alloc, path);
}

/// Run every selected case over one set of image features and compare against the oracle.
fn runAll(
    alloc: std.mem.Allocator,
    model_dir: []const u8,
    text_gguf: []const u8,
    head_blobs: []const u8,
    pixels: []const f32,
    backend: []const u8,
    only: ?[]const u8,
    base: []const u8,
) !void {
    // the vision tower runs once; every question reuses its features, as the Python does
    const img_feats = try vision.encodeImages(alloc, io_g, mmproj_path_g, pixels, backend);
    defer alloc.free(img_feats);
    std.debug.print("laya: img_feats {d} values\n", .{img_feats.len});

    const tok_path = try std.fmt.allocPrint(alloc, "{s}/processor/tokenizer.json", .{model_dir});
    defer alloc.free(tok_path);
    var tok = try tokenizer.Tokenizer.load(alloc, tok_path);
    defer tok.deinit();

    var n_pass: usize = 0;
    var n_run: usize = 0;
    for (CASES) |kase| {
        if (only) |o| {
            if (!std.mem.eql(u8, o, kase.name)) continue;
        }
        n_run += 1;
        const res = try runCase(alloc, &tok, text_gguf, head_blobs, img_feats, kase, backend, base);
        defer alloc.free(res.probs);

        const ref_path = try std.fmt.allocPrint(alloc, "{s}/{s}/probs.f32", .{ base, kase.name });
        defer alloc.free(ref_path);
        const ref_bytes = readFile(alloc, ref_path) catch {
            std.debug.print("  {s}: no oracle at {s}, printing only\n", .{ kase.name, ref_path });
            std.debug.print("  probs = ", .{});
            for (res.probs) |v| std.debug.print("{d:.6} ", .{v});
            std.debug.print("  act = {d:.6} {d:.6}\n", .{ res.act[0], res.act[1] });
            continue;
        };
        defer alloc.free(ref_bytes);

        var max_dp: f32 = 0;
        std.debug.print("  {s}: ", .{kase.name});
        for (res.probs, 0..) |v, ci| {
            const r: f32 = @bitCast(std.mem.readInt(u32, ref_bytes[ci * 4 ..][0..4], .little));
            const dd = @abs(v - r);
            if (dd > max_dp) max_dp = dd;
            std.debug.print("{d:.4} ", .{v});
        }
        const ap_path = try std.fmt.allocPrint(alloc, "{s}/{s}/act_probs.f32", .{ base, kase.name });
        defer alloc.free(ap_path);
        var max_da: f32 = 0;
        if (readF32(alloc, ap_path)) |ap_bytes| {
            defer alloc.free(ap_bytes);
            for (0..2) |bi| {
                const r: f32 = @bitCast(std.mem.readInt(u32, ap_bytes[bi * 4 ..][0..4], .little));
                const dd = @abs(res.act[bi] - r);
                if (dd > max_da) max_da = dd;
            }
        } else |_| {}
        std.debug.print(" max|dprob|={e} max|dact|={e}\n", .{ max_dp, max_da });
        // The graph chain itself is exact to ~3e-7 (run with LAYA_ORACLE_PIXELS=1 to see it).
        // What is left is the image path: this preprocessing agrees with the HuggingFace
        // processor to within one grey level on 0.5% of pixels, and the model is sensitive
        // enough that a sub-level image difference lands here. A control experiment -- the
        // reference pixels perturbed by one level on the same pixels -- moves the output by
        // about the same amount, so this is the floor for any implementation that is not
        // bit-identical to PIL's resample.
        if (max_dp < 1e-3 and max_da < 1e-3) n_pass += 1;
    }

    if (n_run == n_pass) {
        std.debug.print("RESULT: PASS ({d} cases)\n", .{n_pass});
    } else {
        std.debug.print("RESULT: FAIL ({d} of {d} matched)\n", .{ n_pass, n_run });
        return error.Mismatch;
    }
}

/// The mmproj path, stashed for runAll so its signature stays about the data, not the argv.
var mmproj_path_g: []const u8 = "";

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 7 or args.len > 8) {
        std.debug.print("usage: laya <model_dir> <text.gguf> <mmproj.gguf> <head_blobs> <image.png> <cpu|vulkan> [case]\n", .{});
        return error.BadArgs;
    }
    const model_dir = args[1];
    const text_gguf = args[2];
    mmproj_path_g = args[3];
    const head_blobs = args[4];
    const png = args[5];
    const backend: []const u8 = args[6];
    const only: ?[]const u8 = if (args.len == 8) args[7] else null;

    // 1. the image. Normally from the PNG, through the same resize and normalisation the Python
    // uses; LAYA_ORACLE_PIXELS swaps in the reference tensor so the rest of the chain can be
    // judged on its own.
    const base = std.fs.path.dirname(png) orelse ".";
    const pixels = try alloc.alloc(f32, 3 * IMAGE_SIZE * IMAGE_SIZE);
    defer alloc.free(pixels);

    const use_oracle = if (init.environ_map.get("LAYA_ORACLE_PIXELS")) |v| v.len > 0 else false;
    if (use_oracle) {
        var pbuf2: [512]u8 = undefined;
        const pb = try readFile(alloc, try std.fmt.bufPrint(&pbuf2, "{s}/pixels.f32", .{base}));
        defer alloc.free(pb);
        if (pb.len != pixels.len * 4) return error.BadReference;
        @memcpy(std.mem.sliceAsBytes(pixels), pb);
        std.debug.print("laya: using ORACLE pixels from {s}/pixels.f32\n", .{base});
    } else {
        const img = try preprocess.decodePng(alloc, io_g, png);
        defer alloc.free(img.rgb);
        var prep = try preprocess.Prep.init(alloc, IMAGE_SIZE, img.h, img.w);
        defer prep.deinit();
        try prep.apply(alloc, img, pixels);
        std.debug.print("laya: image {d}x{d} -> {d}x{d}, backend={s}\n", .{ img.w, img.h, IMAGE_SIZE, IMAGE_SIZE, backend });
    }

    try runAll(alloc, model_dir, text_gguf, head_blobs, pixels, backend, only, base);
}
