const std = @import("std");

/// GF(2) rank by Gaussian elimination with SIMD row swap and XOR.
/// Returns the rank; `matrix` is left in reduced row form.
pub fn reduceSNF(matrix: []u64, rows: usize, row_words: usize) usize {
    var pivot_row: usize = 0;
    const Vec = @Vector(8, u64); // 8 x u64 lanes

    var rank: usize = 0;

    for (0..row_words * 64) |c| {
        if (pivot_row >= rows) break;
        const word_idx = c / 64;
        const bit_mask = @as(u64, 1) << @intCast(c % 64);

        for (pivot_row..rows) |r| {
            if ((matrix[r * row_words + word_idx] & bit_mask) != 0) {
                // Optimized SIMD Row Swap
                if (r != pivot_row) {
                    const pr_offset = pivot_row * row_words;
                    const r_offset = r * row_words;
                    var w: usize = 0;
                    while (w + 8 <= row_words) : (w += 8) {
                        const v1: Vec = matrix[pr_offset + w ..][0..8].*;
                        const v2: Vec = matrix[r_offset + w ..][0..8].*;
                        matrix[pr_offset + w ..][0..8].* = v2;
                        matrix[r_offset + w ..][0..8].* = v1;
                    }
                    while (w < row_words) : (w += 1) {
                        const tmp = matrix[pr_offset + w];
                        matrix[pr_offset + w] = matrix[r_offset + w];
                        matrix[r_offset + w] = tmp;
                    }
                }

                // SIMD-accelerated Elimination
                const pr_offset = pivot_row * row_words;
                for (0..rows) |er| {
                    if (er != pivot_row and (matrix[er * row_words + word_idx] & bit_mask) != 0) {
                        const er_offset = er * row_words;
                        var ew: usize = 0;
                        while (ew + 8 <= row_words) : (ew += 8) {
                            const ev: Vec = matrix[er_offset + ew ..][0..8].*;
                            const pv: Vec = matrix[pr_offset + ew ..][0..8].*;
                            matrix[er_offset + ew ..][0..8].* = ev ^ pv;
                        }
                        while (ew < row_words) : (ew += 1) {
                            matrix[er_offset + ew] ^= matrix[pr_offset + ew];
                        }
                    }
                }
                pivot_row += 1;
                rank += 1;
                break;
            }
        }
    }
    return rank;
}
