//! qsr-zig: rank of a bit-packed matrix over GF(2), with SIMD row elimination.
//!
//! A matrix is `rows` rows of `row_words` u64 words each, row-major, bit c of
//! a row stored at word c/64, bit c%64. Elimination is in place.
const std = @import("std");

pub const elimination = @import("elimination.zig");
pub const prefetch = @import("prefetch.zig");

/// Rank over GF(2). Destroys `matrix` (it is left in reduced row form).
pub fn rankGF2(matrix: []u64, rows: usize, row_words: usize) usize {
    std.debug.assert(matrix.len >= rows * row_words);
    return elimination.reduceSNF(matrix, rows, row_words);
}

test {
    _ = @import("diff_test.zig");
}

test "identity has full rank, duplicate row drops it" {
    var m = [_]u64{ 1, 2, 4, 8 };
    try std.testing.expectEqual(@as(usize, 4), rankGF2(&m, 4, 1));
    var d = [_]u64{ 3, 3, 5 };
    try std.testing.expectEqual(@as(usize, 2), rankGF2(&d, 3, 1));
}
