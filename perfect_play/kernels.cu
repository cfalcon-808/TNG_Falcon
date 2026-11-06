// Tigers & Goats retrograde solver kernels (compiled at run time by CuPy/NVRTC).
//
// Movement-phase value code (uint16, from the mover's point of view):
//   0          unresolved / draw
//   2*d        mover WINS in d plies (d >= 1)
//   2*d + 1    mover LOSES in d plies (d >= 0; d = 0: no legal move)
// Placement-phase value (uint8): 0 draw, 1 mover wins, 2 mover loses.

typedef long long i64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

__constant__ int NB[23 * 5];   // neighbour of p in direction d, -1 if none
__constant__ int LD[23 * 5];   // capture landing of a tiger on p jumping in direction d, -1 if not allowed
__constant__ int ADJ[23 * 4];  // squares adjacent to q (undirected graph), -1 padded
__constant__ i64 BIN[24 * 24]; // binomial coefficients

__device__ __forceinline__ i64 binom(int n, int k) { return BIN[n * 24 + k]; }

__device__ __forceinline__ i64 rank_board(u32 tm, u32 gm, int k) {
    i64 rt = 0, rg = 0;
    int ti = 0, gi = 0, j = 0;
    for (int p = 0; p < 23; p++) {
        if ((tm >> p) & 1u) { ti++; rt += binom(p, ti); }
        else { if ((gm >> p) & 1u) { gi++; rg += binom(j, gi); } j++; }
    }
    return rt * binom(20, k) + rg;
}

__device__ __forceinline__ void unrank_board(i64 index, int k, u32 &tm, u32 &gm) {
    i64 bk = binom(20, k);
    i64 rt = index / bk, rg = index - rt * bk;
    tm = 0; int x = 22;
    for (int i = 3; i >= 1; i--) { while (binom(x, i) > rt) x--; tm |= 1u << x; rt -= binom(x, i); x--; }
    u32 slots = 0; x = 19;
    for (int i = k; i >= 1; i--) { while (binom(x, i) > rg) x--; slots |= 1u << x; rg -= binom(x, i); x--; }
    gm = 0; int j = 0;
    for (int p = 0; p < 23; p++) {
        if ((tm >> p) & 1u) continue;
        if ((slots >> j) & 1u) gm |= 1u << p;
        j++;
    }
}

// ── Movement phase ──────────────────────────────────────────────────────────────────────────────────
// mv_init: terminal positions, successor counters, and captures into the already-solved layer k-1.
extern "C" __global__ void mv_init(int k, i64 B, u16 *vG, u16 *vT, u8 *cG, u8 *cT,
                                   u16 *extW, u16 *extM, const u16 *lowG) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    u32 tm, gm; unrank_board(i, k, tm, gm);
    u32 occ = tm | gm;
    int c = 15 - k;
    // goats to move: in-layer slides only
    int ng = 0;
    for (int p = 0; p < 23; p++) if ((gm >> p) & 1u)
        for (int d = 1; d <= 4; d++) { int q = NB[p * 5 + d]; if (q >= 0 && !((occ >> q) & 1u)) ng++; }
    if (ng == 0) vG[i] = 1; else cG[i] = (u8)ng;
    // tigers to move: slides stay in the layer; captures go to layer k-1 (or win at the 6th capture)
    int total = 0, open = 0; u32 ew = 0, em = 0;
    for (int p = 0; p < 23; p++) if ((tm >> p) & 1u)
        for (int d = 1; d <= 4; d++) {
            int q = NB[p * 5 + d];
            if (q < 0) continue;
            if (!((occ >> q) & 1u)) { total++; open++; }
            else if ((gm >> q) & 1u) {
                int l = LD[p * 5 + d];
                if (l < 0 || ((occ >> l) & 1u)) continue;
                total++;
                if (c + 1 >= 6) { ew = 1; continue; }
                u16 v = lowG[rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1)];
                if (v == 0) open++;                                              // capture into a draw
                else if (v & 1u) { u32 dd = (v >> 1) + 1; if (ew == 0 || dd < ew) ew = dd; } // goat lost there
                else { u32 dd = v >> 1; if (dd > em) em = dd; }                  // goat wins there
            }
        }
    extW[i] = (u16)ew; extM[i] = (u16)em;
    if (total == 0) vT[i] = 1;
    else {
        cT[i] = (u8)open;
        if (open == 0 && ew == 0) vT[i] = (u16)(2 * (1 + em) + 1);            // every move loses
    }
}

// mv_level: expand every position resolved at distance L to its in-layer predecessors.
extern "C" __global__ void mv_level(int k, i64 B, u32 L, u16 *vG, u16 *vT, u32 *cG32, u32 *cT32,
                                    const u16 *extW, const u16 *extM) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    u16 a = vG[i], b = vT[i];
    bool fg = a != 0 && (u32)(a >> 1) == L;
    bool ft = b != 0 && (u32)(b >> 1) == L;
    if (!fg && !ft) return;
    u32 tm, gm; unrank_board(i, k, tm, gm);
    u32 occ = tm | gm;
    if (fg) {   // goats to move here; predecessors: tigers to move, one tiger slide back
        bool goat_lost = a & 1u;
        for (int q = 0; q < 23; q++) if ((tm >> q) & 1u)
            for (int e = 0; e < 4; e++) {
                int p = ADJ[q * 4 + e];
                if (p < 0 || ((occ >> p) & 1u)) continue;
                i64 j = rank_board(tm ^ (1u << q) ^ (1u << p), gm, k);
                if (goat_lost) atomicCAS(vT + j, (u16)0, (u16)(2 * (L + 1)));
                else {
                    u32 sh = (u32)(j & 3) * 8u;
                    u32 old = atomicSub(cT32 + (j >> 2), 1u << sh);
                    if (((old >> sh) & 0xffu) == 1u && extW[j] == 0) {
                        u32 dd = 1 + (L > extM[j] ? L : extM[j]);
                        atomicCAS(vT + j, (u16)0, (u16)(2 * dd + 1));
                    }
                }
            }
    }
    if (ft) {   // tigers to move here; predecessors: goats to move, one goat slide back
        bool tiger_lost = b & 1u;
        for (int q = 0; q < 23; q++) if ((gm >> q) & 1u)
            for (int e = 0; e < 4; e++) {
                int p = ADJ[q * 4 + e];
                if (p < 0 || ((occ >> p) & 1u)) continue;
                i64 j = rank_board(tm, gm ^ (1u << q) ^ (1u << p), k);
                if (tiger_lost) atomicCAS(vG + j, (u16)0, (u16)(2 * (L + 1)));
                else {
                    u32 sh = (u32)(j & 3) * 8u;
                    u32 old = atomicSub(cG32 + (j >> 2), 1u << sh);
                    if (((old >> sh) & 0xffu) == 1u) atomicCAS(vG + j, (u16)0, (u16)(2 * (L + 1) + 1));
                }
            }
    }
}

// mv_ext: tiger wins by a capture into the solved layer, at distance L + 1.
extern "C" __global__ void mv_ext(i64 B, u32 L, u16 *vT, const u16 *extW) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    if ((u32)extW[i] == L + 1) atomicCAS(vT + i, (u16)0, (u16)(2 * (L + 1)));
}

// mv_check: verify the minimax recursion for every position of a solved movement layer.
// Writes 1 into bad[i] (bit 0: goats-to-move entry, bit 1: tigers-to-move entry) on any mismatch.
extern "C" __global__ void mv_check(int k, i64 B, const u16 *vG, const u16 *vT, const u16 *lowG, u8 *bad) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    u32 tm, gm; unrank_board(i, k, tm, gm);
    u32 occ = tm | gm;
    int c = 15 - k;
    u8 flag = 0;
    {   // goats to move
        int moves = 0, draws = 0; u32 bestw = 0xffffffffu, worstl = 0; bool win = false;
        for (int p = 0; p < 23; p++) if ((gm >> p) & 1u)
            for (int d = 1; d <= 4; d++) {
                int q = NB[p * 5 + d];
                if (q < 0 || ((occ >> q) & 1u)) continue;
                moves++;
                u16 v = vT[rank_board(tm, gm ^ (1u << p) ^ (1u << q), k)];
                if (v == 0) draws++;
                else if (v & 1u) { win = true; u32 dd = (v >> 1) + 1; if (dd < bestw) bestw = dd; }
                else { u32 dd = (v >> 1) + 1; if (dd > worstl) worstl = dd; }
            }
        u16 expect;
        if (moves == 0) expect = 1;
        else if (win) expect = (u16)(2 * bestw);
        else if (draws) expect = 0;
        else expect = (u16)(2 * worstl + 1);
        if (expect != vG[i]) flag |= 1;
    }
    {   // tigers to move
        int moves = 0, draws = 0; u32 bestw = 0xffffffffu, worstl = 0; bool win = false;
        for (int p = 0; p < 23; p++) if ((tm >> p) & 1u)
            for (int d = 1; d <= 4; d++) {
                int q = NB[p * 5 + d];
                if (q < 0) continue;
                u16 v;
                if (!((occ >> q) & 1u)) v = vG[rank_board(tm ^ (1u << p) ^ (1u << q), gm, k)];
                else if ((gm >> q) & 1u) {
                    int l = LD[p * 5 + d];
                    if (l < 0 || ((occ >> l) & 1u)) continue;
                    if (c + 1 >= 6) v = 1;   // goats "lose in 0": the capture wins at once
                    else v = lowG[rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1)];
                } else continue;
                moves++;
                if (v == 0) draws++;
                else if (v & 1u) { win = true; u32 dd = (v >> 1) + 1; if (dd < bestw) bestw = dd; }
                else { u32 dd = (v >> 1) + 1; if (dd > worstl) worstl = dd; }
            }
        u16 expect;
        if (moves == 0) expect = 1;
        else if (win) expect = (u16)(2 * bestw);
        else if (draws) expect = 0;
        else expect = (u16)(2 * worstl + 1);
        if (expect != vT[i]) flag |= 2;
    }
    bad[i] = flag;
}

// mv_to_u8: movement code -> placement value (0 draw, 1 mover wins, 2 mover loses), into a combined array.
extern "C" __global__ void mv_to_u8(i64 B, const u16 *v, u8 *out) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    u16 x = v[i];
    out[i] = x == 0 ? 0 : ((x & 1u) ? 2 : 1);
}

// ── Placement phase (acyclic) ───────────────────────────────────────────────────────────────────────
__device__ __forceinline__ int block_of(i64 i, const i64 *off, int nblocks) {
    int c = 0;
    while (c + 1 < nblocks && i >= off[c + 1]) c++;
    return c;
}

// Goats to move with n placed: place on any empty square -> tigers to move with n + 1 placed.
extern "C" __global__ void pl_goat(int n, i64 total, const i64 *offG, int nbG, const i64 *offT,
                                   const u8 *inT, u8 *outG) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = block_of(i, offG, nbG);
    int k = n - c;
    u32 tm, gm; unrank_board(i - offG[c], k, tm, gm);
    u32 occ = tm | gm;
    bool win = false, draw = false;
    for (int e = 0; e < 23 && !win; e++) {
        if ((occ >> e) & 1u) continue;
        u8 v = inT[offT[c] + rank_board(tm, gm | (1u << e), k + 1)];
        if (v == 2) win = true; else if (v == 0) draw = true;
    }
    outG[i] = win ? 1 : (draw ? 0 : 2);
}

// Tigers to move with n placed: slide (same block) or capture (block c + 1) -> goats to move, n placed.
extern "C" __global__ void pl_tiger(int n, i64 total, const i64 *off, int nb, const u8 *inG, u8 *outT) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = block_of(i, off, nb);
    int k = n - c;
    u32 tm, gm; unrank_board(i - off[c], k, tm, gm);
    u32 occ = tm | gm;
    bool any = false, win = false, draw = false;
    for (int p = 0; p < 23 && !win; p++) if ((tm >> p) & 1u)
        for (int d = 1; d <= 4 && !win; d++) {
            int q = NB[p * 5 + d];
            if (q < 0) continue;
            u8 v;
            if (!((occ >> q) & 1u)) v = inG[off[c] + rank_board(tm ^ (1u << p) ^ (1u << q), gm, k)];
            else if ((gm >> q) & 1u) {
                int l = LD[p * 5 + d];
                if (l < 0 || ((occ >> l) & 1u)) continue;
                if (c + 1 >= 6) v = 2;
                else v = inG[off[c + 1] + rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1)];
            } else continue;
            any = true;
            if (v == 2) win = true; else if (v == 0) draw = true;
        }
    outT[i] = !any ? 2 : (win ? 1 : (draw ? 0 : 2));
}

// pack4: four 2-bit values per byte (value of position 4j + r in bits 2r..2r+1).
extern "C" __global__ void pack4(i64 total, const u8 *in, u8 *out) {
    i64 j = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    i64 base = 4 * j;
    if (base >= total) return;
    u8 b = 0;
    for (int r = 0; r < 4; r++) if (base + r < total) b |= (u8)((in[base + r] & 3u) << (2 * r));
    out[j] = b;
}

// ── Placement phase with ply counts (solve_placement_dtm.py) ────────────────────────────────────────
// Ply code (uint8, mover's point of view): 0 draw, otherwise d + 1 where d = plies to the end with
// optimal play (winner fastest, loser slowest). The winner makes the last move, so d is odd when the
// mover wins and even when the mover loses.

// mv_to_dtm8: movement code (0, 2d, 2d+1) -> ply code; counts entries whose parity breaks the rule above.
extern "C" __global__ void mv_to_dtm8(i64 B, const u16 *v, u8 *out, unsigned long long *parity_bad) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    u16 x = v[i];
    if (x == 0) { out[i] = 0; return; }
    u32 d = x >> 1;
    bool win = (x & 1u) == 0;
    if (win != ((d & 1u) == 1u)) atomicAdd(parity_bad, 1ull);
    out[i] = (u8)(d + 1);
}

// One successor seen from the mover: s is the successor's ply code from the opponent's point of view.
struct Best {
    u32 win, loss; bool draw, any;
    __device__ Best() : win(0xffffffffu), loss(0), draw(false), any(false) {}
    __device__ void add(u32 s) {
        any = true;
        if (s == 0) { draw = true; return; }
        u32 d = s - 1;
        if ((d & 1u) == 0) { if (d + 1 < win) win = d + 1; }   // opponent loses in d: we win in d + 1
        else if (d + 1 > loss) loss = d + 1;                   // opponent wins in d: we lose in d + 1
    }
    __device__ u32 code() const {
        if (!any) return 1;                                     // no legal move: lose in 0
        if (win != 0xffffffffu) return win + 1;
        if (draw) return 0;
        return loss + 1;
    }
};

__device__ __forceinline__ u8 store8(u32 code, u32 *max_code) {
    atomicMax(max_code, code);
    return (u8)(code > 255u ? 255u : code);
}

// Goats to move with n placed: place on any empty square -> tigers to move with n + 1 placed.
extern "C" __global__ void pl_goat_dtm(int n, i64 total, const i64 *offG, int nbG, const i64 *offT,
                                       const u8 *inT, u8 *outG, u32 *max_code) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = block_of(i, offG, nbG);
    int k = n - c;
    u32 tm, gm; unrank_board(i - offG[c], k, tm, gm);
    u32 occ = tm | gm;
    Best b;
    for (int e = 0; e < 23; e++) {
        if ((occ >> e) & 1u) continue;
        b.add(inT[offT[c] + rank_board(tm, gm | (1u << e), k + 1)]);
    }
    outG[i] = store8(b.code(), max_code);
}

// Tigers to move with n placed: slide (same block) or capture (block c + 1, or a win at the 6th capture).
extern "C" __global__ void pl_tiger_dtm(int n, i64 total, const i64 *off, int nb, const u8 *inG, u8 *outT,
                                        u32 *max_code) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = block_of(i, off, nb);
    int k = n - c;
    u32 tm, gm; unrank_board(i - off[c], k, tm, gm);
    u32 occ = tm | gm;
    Best b;
    for (int p = 0; p < 23; p++) if ((tm >> p) & 1u)
        for (int d = 1; d <= 4; d++) {
            int q = NB[p * 5 + d];
            if (q < 0) continue;
            if (!((occ >> q) & 1u)) b.add(inG[off[c] + rank_board(tm ^ (1u << p) ^ (1u << q), gm, k)]);
            else if ((gm >> q) & 1u) {
                int l = LD[p * 5 + d];
                if (l < 0 || ((occ >> l) & 1u)) continue;
                if (c + 1 >= 6) b.add(1u);                  // goats lose in 0: the 6th capture wins at once
                else b.add(inG[off[c + 1] + rank_board(tm ^ (1u << p) ^ (1u << l), gm ^ (1u << q), k - 1)]);
            }
        }
    outT[i] = store8(b.code(), max_code);
}

// cmp_u2: does the win/draw/loss implied by a ply code equal the 2-bit table entry?
extern "C" __global__ void cmp_u2(i64 total, const u8 *dtm, const u8 *packed, unsigned long long *bad) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    u32 s = dtm[i];
    u32 v = s == 0 ? 0u : (((s - 1) & 1u) ? 1u : 2u);
    u32 old = (packed[i >> 2] >> (2 * (i & 3))) & 3u;
    if (v != old) atomicAdd(bad, 1ull);
}
