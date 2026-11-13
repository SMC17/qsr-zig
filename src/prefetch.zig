const std = @import("std");
const elimination = @import("elimination.zig");

/// Prefetch the first 64 floats at `vector_ptr`, then compute GF(2) rank of
/// `matrix` in place.
///
/// Before 2026-11-12 this kernel eliminated against the pivot position
/// without first swapping the pivot row into it, so it returned wrong
/// ranks (caught by src/diff_test.zig). It now delegates elimination to
/// the row-swapping kernel; the prefetch is kept for callers that pass a
/// vector.
pub fn fusedReduceSNF(
    vector_ptr: [*]const f32,
    matrix: []u64,
    rows: usize,
    row_words: usize,
) usize {
    var p: usize = 0;
    while (p < 64) : (p += 16) {
        @prefetch(vector_ptr + p, .{ .rw = .read, .locality = 3, .cache = .data });
    }
    return elimination.reduceSNF(matrix, rows, row_words);
}
