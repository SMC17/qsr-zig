# qsr-zig

Rank of a bit-packed matrix over GF(2) in Zig 0.16, with SIMD row elimination. Apache-2.0.

```zig
const qsr = @import("qsr_zig");
var m = [_]u64{ 0b011, 0b011, 0b101 }; // 3 rows, 1 word each
const r = qsr.rankGF2(&m, 3, 1); // 2
```

```sh
zig fetch --save git+https://github.com/SMC17/qsr-zig
zig build test --summary all   # 4/4, includes 800 differential trials
```

## Status

**Unit-tested.** `src/diff_test.zig` checks both kernels against a naive GF(2)
Gaussian elimination on 400 random matrices each (1-80 rows, 1-12 words per
row, dense, sparse, and with forced dependent rows).

That test found a bug: `fusedReduceSNF` skipped the pivot-row swap and returned
wrong ranks. It now delegates to the row-swapping kernel. Earlier versions also
exported `extractBettiNumbers`, which derived "Betti numbers" from the digits
of a rank. That is not homology, so it was removed (see IDEAS.md).

No performance claim is made. There is no benchmark yet.

## Layout

| File | What |
|---|---|
| `src/root.zig` | `rankGF2` entry point |
| `src/snf_morphogenetic.zig` | elimination with 8-lane `@Vector(8, u64)` row swap and XOR |
| `src/snf_fused.zig` | prefetch, then the same elimination |
| `src/diff_test.zig` | differential tests against a naive reference |
