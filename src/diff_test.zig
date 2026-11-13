//! Differential tests: both SIMD kernels against a naive GF(2) Gaussian elimination.
const std = @import("std");
const morph = @import("elimination.zig");
const fused = @import("prefetch.zig");

fn naiveRank(m: []u64, rows: usize, row_words: usize) usize {
    var rank: usize = 0;
    var col: usize = 0;
    while (col < row_words * 64 and rank < rows) : (col += 1) {
        const w = col / 64;
        const bit = @as(u64, 1) << @intCast(col % 64);
        var piv: ?usize = null;
        for (rank..rows) |r| if (m[r * row_words + w] & bit != 0) {
            piv = r;
            break;
        };
        const p = piv orelse continue;
        for (0..row_words) |k| std.mem.swap(u64, &m[p * row_words + k], &m[rank * row_words + k]);
        for (0..rows) |r| if (r != rank and m[r * row_words + w] & bit != 0) {
            for (0..row_words) |k| m[r * row_words + k] ^= m[rank * row_words + k];
        };
        rank += 1;
    }
    return rank;
}

fn check(comptime which: enum { morph, fused }) !void {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const a = std.testing.allocator;
    const dummy = [_]f32{0} ** 64;
    for (0..400) |trial| {
        const rows = 1 + rnd.uintLessThan(usize, 80);
        const row_words = 1 + rnd.uintLessThan(usize, 12);
        const m = try a.alloc(u64, rows * row_words);
        defer a.free(m);
        const sparse = trial % 3 == 0;
        for (m) |*x| x.* = if (sparse) rnd.int(u64) & rnd.int(u64) & rnd.int(u64) else rnd.int(u64);
        // force dependent rows sometimes
        if (rows > 2 and trial % 2 == 0) for (0..row_words) |k| {
            m[(rows - 1) * row_words + k] = m[0 * row_words + k] ^ m[1 * row_words + k];
        };
        const ref = try a.dupe(u64, m);
        defer a.free(ref);
        const want = naiveRank(ref, rows, row_words);
        const got = switch (which) {
            .morph => morph.reduceSNF(m, rows, row_words),
            .fused => fused.fusedReduceSNF(&dummy, m, rows, row_words),
        };
        std.testing.expectEqual(want, got) catch |e| {
            std.debug.print("trial {d} rows={d} words={d}\n", .{ trial, rows, row_words });
            return e;
        };
    }
}

test "reduceSNF matches naive GF(2) rank on 400 random matrices" {
    try check(.morph);
}
test "fusedReduceSNF matches naive GF(2) rank on 400 random matrices" {
    try check(.fused);
}
