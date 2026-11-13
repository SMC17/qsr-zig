# Ideas carried forward

## Topological features for vector search (from faiss-zig, removed 2026-11-12)

faiss-zig had `src/topological_vector.zig`: hash a vector into (b0, b1, b2)
"Betti numbers" and filter candidates by popcount distance before L2. The
implementation never built a simplicial complex from the vector, and
`topologicalSearch` discarded its arguments, so nothing was measured.

A real version needs: a Vietoris-Rips or witness complex over a neighbourhood,
boundary matrices per dimension, and ranks over GF(2) (this library), giving
b_k = dim C_k - rank d_k - rank d_{k+1}. Then a recall/QPS comparison against
plain HNSW on a public dataset before any claim. Source of the old module is in
faiss-zig history before the commit that removed it.
