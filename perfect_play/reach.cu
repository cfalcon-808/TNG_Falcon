// Forward reachability from the opening (compiled after kernels.cu, whose helpers it uses).
//
// Placement layers: one byte per position, 1 = reachable. Layer "15" for tigers to move is laid out like
// placement_offsets(15): the movement blocks k = 15 - c for c = 0..5, as in solve_placement_dtm.py.
// Movement phase: uint16 per position = fewest plies from the opening, 0xFFFF = not reachable.

#define UNREACHED 0xFFFFu

// Goats to move with n placed -> tigers to move with n + 1 placed (place a goat on any empty square).
extern "C" __global__ void fw_pl_goat(int n, i64 total, const i64 *offG, int nbG, const i64 *offT,
                                      const u8 *rG, u8 *rT) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || !rG[i]) return;
    int c = block_of(i, offG, nbG);
    int k = n - c;
    u32 tm, gm; unrank_board(i - offG[c], k, tm, gm);
    u32 occ = tm | gm;
    for (int e = 0; e < 23; e++)
        if (!((occ >> e) & 1u)) rT[offT[c] + rank_board(tm, gm | (1u << e), k + 1)] = 1;
}

// Tigers to move with n placed -> goats to move with n placed (slide, or capture into block c + 1).
// The 6th capture ends the game, so it leads to no position.
extern "C" __global__ void fw_pl_tiger(int n, i64 total, const i64 *off, int nb, const u8 *rT, u8 *rG) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total || !rT[i]) return;
    int c = block_of(i, off, nb);
    int k = n - c;
    u32 tm, gm; unrank_board(i - off[c], k, tm, gm);
    u32 occ = tm | gm;
    for (int p = 0; p < 23; p++) if ((tm >> p) & 1u)
        for (int d = 1; d <= 4; d++) {
            int q = NB[p * 5 + d];
            if (q < 0) continue;
            if (!((occ >> q) & 1u)) rG[off[c] + rank_board(tm ^ (1u << p) ^ (1u << q), gm, k)] = 1;
            else if ((gm >> q) & 1u) {
                int l = LD[p * 5 + d];
                if (l < 0 || ((occ >> l) & 1u) || c + 1 >= 6) continue;
                rG[off[c + 1] + rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1)] = 1;
            }
        }
}

// Breadth-first search in the movement phase: expand every position first reached at ply r.
// Every writer in a round writes the same value r + 1, so the races are benign.
extern "C" __global__ void fw_mv_goat(int k, i64 B, u32 r, const u16 *dG, u16 *dT, unsigned long long *added) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B || dG[i] != r) return;
    u32 tm, gm; unrank_board(i, k, tm, gm);
    u32 occ = tm | gm;
    for (int p = 0; p < 23; p++) if ((gm >> p) & 1u)
        for (int d = 1; d <= 4; d++) {
            int q = NB[p * 5 + d];
            if (q < 0 || ((occ >> q) & 1u)) continue;
            i64 j = rank_board(tm, gm ^ (1u << p) ^ (1u << q), k);
            if (dT[j] == UNREACHED) { dT[j] = (u16)(r + 1); atomicAdd(added, 1ull); }
        }
}

extern "C" __global__ void fw_mv_tiger(int k, i64 B, u32 r, const u16 *dT, u16 *dG, u16 *dGdown,
                                       unsigned long long *added) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B || dT[i] != r) return;
    u32 tm, gm; unrank_board(i, k, tm, gm);
    u32 occ = tm | gm;
    for (int p = 0; p < 23; p++) if ((tm >> p) & 1u)
        for (int d = 1; d <= 4; d++) {
            int q = NB[p * 5 + d];
            if (q < 0) continue;
            if (!((occ >> q) & 1u)) {
                i64 j = rank_board(tm ^ (1u << p) ^ (1u << q), gm, k);
                if (dG[j] == UNREACHED) { dG[j] = (u16)(r + 1); atomicAdd(added, 1ull); }
            } else if ((gm >> q) & 1u) {
                int l = LD[p * 5 + d];
                if (l < 0 || ((occ >> l) & 1u) || k <= 10) continue;   // k = 10: the 6th capture wins
                i64 j = rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1);
                if (dGdown[j] == UNREACHED) { dGdown[j] = (u16)(r + 1); atomicAdd(added, 1ull); }
            }
        }
}
