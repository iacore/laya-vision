//! Image preprocessing: PNG -> the pixel tensor the vision tower eats.
//!
//! Ported from `laya/preprocess.py`, which is the spec. The checkpoint records no
//! `image_size`/`preprocess` keys, so `ImagePrep.from_config(..., default_backend="processor")`
//! selects image_size 512, backend "processor", interpolation "processor" -- confirmed by
//! running the Python, not inferred.
//!
//! The "processor" path is the HuggingFace Idefics3 processor's two LANCZOS hops:
//!
//!     210x160 --(size.longest_edge 2048)--> 2048x1560 --(max_image_size 512)--> 512x512
//!
//! The first hop upscales to 3.2 megapixels only for the second to throw it away. Both hops
//! are linear and separable, so the pair composes into one matrix per axis:
//!
//!     out[c,i,j] = sum_p sum_q Wh[i,p] * in[c,p,q] * Ww[j,q]
//!
//! Two details are the whole of the accuracy here, and both are easy to drop:
//!
//!   * `_axis_weights` builds its weights in float64 (torch's antialiased resample weights).
//!   * `_resize` rounds the float result back to **uint8** and clamps, and `pixel_values`
//!     then normalises the *rounded* values -- (v - 127.5) / 127.5. The output is therefore
//!     quantised to whole levels, and skipping the round puts every pixel up to half a level
//!     out. torch's `.round()` is round-half-to-even, not half-away-from-zero.
//!
//! What the composition cannot reproduce is the processor clamping the 2048-pixel intermediate
//! back to uint8: clamping is not linear, so LANCZOS overshoot at a hard edge survives here
//! where the processor cut it off. `laya/preprocess.py` says so and measures it at a handful of
//! pixels per frame.

const std = @import("std");

const STAGE1: usize = 2048;
const RESCALE_MEAN: f32 = 127.5; // image_mean / rescale_factor, so (v - 127.5) / 127.5

extern "c" fn stbi_load_from_memory(
    buffer: [*]const u8,
    len: c_int,
    x: *c_int,
    y: *c_int,
    channels_in_file: *c_int,
    desired_channels: c_int,
) ?[*]u8;
extern "c" fn stbi_image_free(retval_from_stbi_load: ?*anyopaque) void;

var io_g: std.Io = undefined;

pub const Image = struct { w: usize, h: usize, rgb: []u8 };

/// Decode a PNG to interleaved RGB, row-major. The caller owns `rgb`.
pub fn decodePng(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Image {
    io_g = io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
    defer alloc.free(bytes);

    var w: c_int = 0;
    var h: c_int = 0;
    var n: c_int = 0;
    const px = stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &w, &h, &n, 3) orelse
        return error.BadPng;
    defer stbi_image_free(px);
    if (w <= 0 or h <= 0) return error.BadPng;

    const len = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 3;
    const out = try alloc.alloc(u8, len);
    errdefer alloc.free(out);
    @memcpy(out, px[0..len]);
    return .{ .w = @intCast(w), .h = @intCast(h), .rgb = out };
}

// -- the resample weights -----------------------------------------------------------------------

/// Lanczos-3, the filter behind PILImageResampling.LANCZOS. Zero outside |x| < 3.
fn lanczos(x: f64) f64 {
    const ax = @abs(x);
    if (ax < 1e-12) return 1.0;
    if (ax >= 3.0) return 0.0;
    const p = std.math.pi * ax;
    return (@sin(p) / p) * (@sin(p / 3.0) / (p / 3.0));
}

/// torch's antialiased resample along one axis, `[n_out, n_in]`, built in float64.
///
/// An output pixel's kernel is centred on (i + 0.5) * scale, stretched by max(1, scale) when
/// downsampling, clipped at the borders and renormalised -- ATen's upsample antialias path.
fn axisWeights(alloc: std.mem.Allocator, n_in: usize, n_out: usize) ![]f64 {
    const out = try alloc.alloc(f64, n_out * n_in);
    errdefer alloc.free(out);
    @memset(out, 0);

    if (n_in == n_out) {
        for (0..n_out) |i| out[i * n_in + i] = 1.0;
        return out;
    }

    const ni: f64 = @floatFromInt(n_in);
    const no: f64 = @floatFromInt(n_out);
    const scale = ni / no;
    const stretch = @max(1.0, scale);
    const support = 3.0 * stretch;
    const span: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 2;

    for (0..n_out) |i| {
        const centre = (@as(f64, @floatFromInt(i)) + 0.5) * scale;
        // the leftmost tap, and the destination column it lands in
        const lo_f = @floor(centre - support + 0.5);
        const lo: i64 = if (lo_f < 0) 0 else @intFromFloat(lo_f);
        var row_sum: f64 = 0;
        for (0..span) |j| {
            const idx: i64 = lo + @as(i64, @intCast(j));
            const t = (@as(f64, @floatFromInt(idx)) + 0.5 - centre) / stretch;
            var w = lanczos(t);
            // taps past the far edge are kept out of the sum, but their slot still exists
            if (idx >= @as(i64, @intCast(n_in))) w = 0;
            const dst: usize = @intCast(@min(@max(idx, 0), @as(i64, @intCast(n_in - 1))));
            out[i * n_in + dst] += w;
            row_sum += w;
        }
        for (0..n_in) |k| out[i * n_in + k] /= row_sum;
    }
    return out;
}

/// PIL accumulates in double, and so does this: with float32 taps a sub-level difference
/// crosses a rounding boundary often enough to show up in the pixels, which is measurable.
fn weightsF64(alloc: std.mem.Allocator, n_in: usize, n_out: usize) ![]f64 {
    return axisWeights(alloc, n_in, n_out);
}

/// The processor's first hop: longest edge to `longest_edge`, the other rounded up to even.
/// Returns (height, width) -- note the branches swap which one gets the fixed value.
pub fn stage1Size(height_in: usize, width_in: usize) struct { h: usize, w: usize } {
    var height = height_in;
    var width = width_in;
    if (width_in >= height_in) {
        height = (STAGE1 * height_in) / width_in;
        height += height % 2;
        width = STAGE1;
    } else {
        width = (STAGE1 * width_in) / height_in;
        width += width % 2;
        height = STAGE1;
    }
    return .{ .h = @max(height, 1), .w = @max(width, 1) };
}

/// PIL's resample rounding, which is what the reference pixels came from:
/// `(UINT8)(CLIP(ss) + 0.5)` -- clamp first, then half-up, then truncate. Not half-even.
fn quantizeLikePil(x: f64) u8 {
    const c0 = std.math.clamp(x, 0.0, 255.0);
    return @intFromFloat(c0 + 0.5);
}

pub const Prep = struct {
    alloc: std.mem.Allocator,
    size: usize,
    in_h: usize,
    in_w: usize,
    mid_h: usize,
    mid_w: usize,
    /// the two hops kept separately, not composed
    wh1: []f64, // [mid_h, in_h]
    ww1: []f64, // [mid_w, in_w]
    wh2: []f64, // [size,  mid_h]
    ww2: []f64, // [size,  mid_w]

    pub fn init(alloc: std.mem.Allocator, image_size: usize, in_h: usize, in_w: usize) !Prep {
        const mid = stage1Size(in_h, in_w);
        const wh1 = try weightsF64(alloc, in_h, mid.h);
        errdefer alloc.free(wh1);
        const ww1 = try weightsF64(alloc, in_w, mid.w);
        errdefer alloc.free(ww1);
        const wh2 = try weightsF64(alloc, mid.h, image_size);
        errdefer alloc.free(wh2);
        const ww2 = try weightsF64(alloc, mid.w, image_size);
        errdefer alloc.free(ww2);
        return .{
            .alloc = alloc,
            .size = image_size,
            .in_h = in_h,
            .in_w = in_w,
            .mid_h = mid.h,
            .mid_w = mid.w,
            .wh1 = wh1,
            .ww1 = ww1,
            .wh2 = wh2,
            .ww2 = ww2,
        };
    }

    pub fn deinit(self: *Prep) void {
        self.alloc.free(self.wh1);
        self.alloc.free(self.ww1);
        self.alloc.free(self.wh2);
        self.alloc.free(self.ww2);
    }

    /// `img` -> `out`, channels-first `[3, size, size]`, normalised to [-1, 1].
    ///
    /// Two hops, not one composition, and that choice is the whole point. `resize_operator` in
    /// the Python composes them precisely *because* it is a different, faster path -- and it
    /// says in its own docstring that the composition cannot reproduce the processor's rounding
    /// and clamping of the 2048-pixel intermediate, so it lands a fraction of a level off with
    /// outliers of ~19 levels at hard edges. The runtime the model actually sees is
    /// `vlm_prefix`, which for backend "processor" calls the HuggingFace processor: two real
    /// resizes with a uint8 round-and-clamp between them. This is that path.
    pub fn apply(self: *const Prep, alloc: std.mem.Allocator, img: Image, out: []f32) !void {
        const s = self.size;
        const mh = self.mid_h;
        const mw = self.mid_w;
        std.debug.assert(out.len == 3 * s * s);
        if (img.h != self.in_h or img.w != self.in_w) return error.SizeMismatch;

        // hop 1: in -> the intermediate, uint8 in and uint8 out as PIL does
        const t1 = try alloc.alloc(f64, 3 * self.in_h * mw);
        defer alloc.free(t1);
        const mid = try alloc.alloc(u8, 3 * mh * mw);
        defer alloc.free(mid);
        for (0..3) |c| {
            for (0..self.in_h) |p0| {
                for (0..mw) |q| {
                    var acc: f64 = 0;
                    for (0..self.in_w) |q0| {
                        acc += @as(f64, @floatFromInt(img.rgb[(p0 * self.in_w + q0) * 3 + c])) * self.ww1[q * self.in_w + q0];
                    }
                    t1[(c * self.in_h + p0) * mw + q] = acc;
                }
            }
            // PIL resamples 8-bit images a row at a time into an 8-bit intermediate, so the
            // horizontal pass rounds before the vertical one ever sees it
            for (t1[c * self.in_h * mw ..][0 .. self.in_h * mw]) |*v| {
                v.* = @floatFromInt(quantizeLikePil(@floatCast(v.*)));
            }
            for (0..mh) |p| {
                for (0..mw) |q| {
                    var acc: f64 = 0;
                    for (0..self.in_h) |p0| acc += self.wh1[p * self.in_h + p0] * t1[(c * self.in_h + p0) * mw + q];
                    mid[(c * mh + p) * mw + q] = quantizeLikePil(@floatCast(acc));
                }
            }
        }

        // hop 2: intermediate -> the target square
        const t2 = try alloc.alloc(f64, 3 * mh * s);
        defer alloc.free(t2);
        for (0..3) |c| {
            for (0..mh) |p| {
                for (0..s) |j| {
                    var acc: f64 = 0;
                    for (0..mw) |q| {
                        acc += @as(f64, @floatFromInt(mid[(c * mh + p) * mw + q])) * self.ww2[j * mw + q];
                    }
                    t2[(c * mh + p) * s + j] = acc;
                }
            }
            for (t2[c * mh * s ..][0 .. mh * s]) |*v| {
                v.* = @floatFromInt(quantizeLikePil(@floatCast(v.*)));
            }
            for (0..s) |i| {
                for (0..s) |j| {
                    var acc: f64 = 0;
                    for (0..mh) |p| acc += self.wh2[i * mh + p] * t2[(c * mh + p) * s + j];
                    // a whole level, clamped, and *then* normalised: the reference is the
                    // quantised value, not the float that produced it
                    const q: f32 = @floatFromInt(quantizeLikePil(@floatCast(acc)));
                    out[(c * s + i) * s + j] = (q - RESCALE_MEAN) / RESCALE_MEAN;
                }
            }
        }
    }
};

// -- verification -------------------------------------------------------------------------------

fn readFile(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io_g, path, alloc, std.Io.Limit.limited(1 << 30));
}

pub fn main(init: std.process.Init) !void {
    io_g = init.io;
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: preprocess <fixture.png> <oracle_dir> [out_pixels.f32]\n", .{});
        return error.BadArgs;
    }
    const png = args[1];
    const cdir = args[2];

    const img = try decodePng(alloc, io_g, png);
    defer alloc.free(img.rgb);
    std.debug.print("preprocess: png {d}x{d}\n", .{ img.w, img.h });

    const mid = stage1Size(img.h, img.w);
    std.debug.print("  stage1 {d}x{d} -> {d}x{d} -> 512x512\n", .{ img.h, img.w, mid.h, mid.w });

    var prep = try Prep.init(alloc, 512, img.h, img.w);
    defer prep.deinit();
    const pixels = try alloc.alloc(f32, 3 * 512 * 512);
    defer alloc.free(pixels);
    try prep.apply(alloc, img, pixels);

    var pbuf: [512]u8 = undefined;
    const ref_bytes = try readFile(alloc, try std.fmt.bufPrint(&pbuf, "{s}/pixels.f32", .{cdir}));
    defer alloc.free(ref_bytes);
    const ref: [*]const f32 = @ptrCast(@alignCast(ref_bytes.ptr));
    if (ref_bytes.len != pixels.len * 4) {
        std.debug.print("  reference holds {d} floats, expected {d}\n", .{ ref_bytes.len / 4, pixels.len });
        return error.BadReference;
    }

    var n_diff: usize = 0;
    var n_half: usize = 0;
    var n_one: usize = 0;
    var n_five: usize = 0;
    var max_abs: f32 = 0;
    var sum_abs: f64 = 0;
    for (pixels, 0..) |v, i| {
        const d = @abs(v - ref[i]);
        const lv = d * RESCALE_MEAN;
        if (d > 0) n_diff += 1;
        if (lv > 0.5) n_half += 1;
        if (lv > 1.0) n_one += 1;
        if (lv > 5.0) n_five += 1;
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }

    std.debug.print("  pixels: {d} values\n", .{pixels.len});
    std.debug.print("  first 4 got: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ pixels[0], pixels[1], pixels[2], pixels[3] });
    std.debug.print("  first 4 ref: {d:.6} {d:.6} {d:.6} {d:.6}\n", .{ ref[0], ref[1], ref[2], ref[3] });
    std.debug.print("  differing: {d}   max {d:.2} levels   mean {d:.3} levels\n", .{ n_diff, max_abs * RESCALE_MEAN, sum_abs / @as(f64, @floatFromInt(pixels.len)) * RESCALE_MEAN });
    std.debug.print("  over 0.5 levels: {d}   over 1: {d}   over 5: {d}\n", .{ n_half, n_one, n_five });

    if (args.len == 4) {
        const f = try std.Io.Dir.cwd().createFile(io_g, args[3], .{});
        defer f.close(io_g);
        var wbuf: [4096]u8 = undefined;
        var fw = f.writer(io_g, &wbuf);
        try fw.interface.writeAll(std.mem.sliceAsBytes(pixels));
        try fw.interface.flush();
        std.debug.print("  wrote pixels -> {s}\n", .{args[3]});
    }

    const mean_lv = sum_abs / @as(f64, @floatFromInt(pixels.len)) * RESCALE_MEAN;
    if (mean_lv < 0.1) {
        std.debug.print("  RESULT: PASS\n", .{});
    } else {
        std.debug.print("  RESULT: FAIL\n", .{});
        return error.Mismatch;
    }
}
