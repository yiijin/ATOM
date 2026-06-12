# SPDX-License-Identifier: Apache-2.0
"""FlyDSL port of the prefill causal-conv1d "tile" kernel.

This mirrors ``_causal_conv1d_fwd_kernel_tile`` from ``causal_conv1d.py`` (the
2D Triton tile kernel) but is authored in the FlyDSL Python DSL and compiled
through the Fly/ROCDL MLIR stack.

Programming model
-----------------
Triton lets the compiler map a ``[BLOCK_N channels x BLOCK_M tokens]`` register
tile onto the wave/lane grid implicitly. FlyDSL's low-level ``buffer_ops`` API
requires an explicit thread mapping, so we use:

    1 thread  ==  1 (token, channel) output element
    block     ==  (BLOCK_M tokens) x (CH_PER_BLK channels)  flattened to 1D
    grid      ==  (num_programs, ceil(dim / CH_PER_BLK))

``num_programs`` is the flattened ``(sequence, chunk)`` list (axis 0 of the
Triton grid), built from ``query_start_loc`` exactly like the Triton wrapper.
A wave (64 consecutive lanes) covers 64 consecutive *tokens* of one channel.
The benchmark feeds a token-contiguous ``x`` (``x.stride(1) == 1``), so this
mapping gives fully coalesced global loads of ``x``.

Each thread independently loads its causal window ``x[p], x[p-1], ... x[p-(K-1)]``
(KERNEL_WIDTH taps), blending in the conv-state columns at the sequence start,
accumulates the depthwise conv in fp32, applies SiLU, and scatters the result
into the per-token Q / K / V output tensors (the input channels are the
concatenation ``[Q | K | V]``).

Only the common prefill configuration is ported (``IS_APC_ENABLED=False``,
``USE_PAD_SLOT`` implicitly handled by an exact-size grid). KERNEL_WIDTH in
{2, 3, 4} is supported.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, rocdl
    from flydsl.expr.arith import CmpIPredicate
    from flydsl.expr.typing import T, Int32
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import scf
    from flydsl.expr import buffer_ops
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    _FLYDSL_AVAILABLE = True
except Exception:  # pragma: no cover - flydsl optional
    _FLYDSL_AVAILABLE = False


PAD_SLOT_ID = -1
_LOG2E = 1.4426950408889634


def is_flydsl_available() -> bool:
    return _FLYDSL_AVAILABLE


def build_causal_conv1d_flydsl_module(
    width: int,
    has_bias: bool,
    silu: bool,
    block_m: int = 64,
    block_dim: int = 256,
    dtype_str: str = "bf16",
):
    """Build + return a ``@flyc.jit`` launcher specialized on compile-time params.

    Parameters
    ----------
    width       : convolution kernel width (taps), 2..4
    has_bias    : whether a per-channel bias is added
    silu        : whether SiLU activation is applied
    block_m     : tokens processed per program along the sequence (must match
                  the chunking used to build ``batch_ptr`` / metadata; 64)
    block_dim   : threads per block (multiple of block_m); CH_PER_BLK channels
                  per block = block_dim // block_m
    dtype_str   : element dtype of x / weight / outputs ("bf16" or "fp16")
    """
    assert _FLYDSL_AVAILABLE, "flydsl is not installed"
    assert width in (2, 3, 4), f"unsupported width={width}"
    assert block_dim % block_m == 0, "block_dim must be a multiple of block_m"
    assert (block_m & (block_m - 1)) == 0, "block_m must be a power of two"

    W = width
    STATE_LEN = W - 1
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)
    BLOCK_M = block_m
    LOG2_BM = int(block_m).bit_length() - 1  # block_m == 1 << LOG2_BM
    CH_PER_BLK = block_dim // block_m

    @flyc.kernel
    def conv1d_kernel(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,           # conv_states
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,          # query_start_loc
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,    # token_chunk_offset_ptr
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,                    # k_dim_size  (Q channel count)
        vd: Int32,                    # v_dim_size  (unused directly; kept for parity)
        sx0: Int32,                   # x.stride(0)  (per-channel)
        sx1: Int32,                   # x.stride(1)  (per-token)
        sw0: Int32,                   # weight.stride(0)
        sw1: Int32,                   # weight.stride(1)
        scs0: Int32,                  # conv_states.stride(0) (slot)
        scs1: Int32,                  # conv_states.stride(1) (channel)
        scs2: Int32,                  # conv_states.stride(2) (state token)
        sci: Int32,                   # cache_indices.stride(0)
        qs0: Int32,                   # query.stride(0) (token)
        qs1: Int32,                   # query.stride(1) (channel)
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
    ):
        i32 = T.i32
        elem_dtype = T.bf16 if dtype_str == "bf16" else T.f16

        def _v(x):
            return x.ir_value() if hasattr(x, "ir_value") else x

        # materialize scalar (Int32) kernel params as raw i32 ir.Values
        dim = _v(dim)
        kd = _v(kd)
        vd = _v(vd)
        sx0 = _v(sx0)
        sx1 = _v(sx1)
        sw0 = _v(sw0)
        sw1 = _v(sw1)
        scs0 = _v(scs0)
        scs1 = _v(scs1)
        scs2 = _v(scs2)
        sci = _v(sci)
        qs0 = _v(qs0)
        qs1 = _v(qs1)
        ks0 = _v(ks0)
        ks1 = _v(ks1)
        vs0 = _v(vs0)
        vs1 = _v(vs1)

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        x_r = _rsrc(x_ptr)
        w_r = _rsrc(w_ptr)
        b_r = _rsrc(bias_ptr)
        cs_r = _rsrc(cs_ptr)
        ci_r = _rsrc(cache_idx_ptr)
        hi_r = _rsrc(has_init_ptr)
        qsl_r = _rsrc(qsl_ptr)
        batch_r = _rsrc(batch_ptr)
        choff_r = _rsrc(chunk_off_ptr)
        q_r = _rsrc(q_ptr)
        k_r = _rsrc(k_ptr)
        v_r = _rsrc(v_ptr)

        def c32(v):
            return arith.constant(int(v), type=i32)

        def cf(v):
            return arith.constant(float(v), type=T.f32)

        def to_i32(v):
            return arith.index_cast(i32, v)

        def mul(a, b):
            return arith.muli(a, b)

        def add(a, b):
            return arith.addi(a, b)

        def sub(a, b):
            return arith.subi(a, b)

        # --- thread / block decomposition ---
        tid = to_i32(fx.thread_idx.x)
        pid_x = to_i32(fx.block_idx.x)   # program (sequence-chunk)
        pid_y = to_i32(fx.block_idx.y)   # channel tile

        token_local = arith.andi(tid, c32(BLOCK_M - 1))      # tid % BLOCK_M
        ch_local = arith.shrui(tid, c32(LOG2_BM))            # tid // BLOCK_M
        channel = add(mul(pid_y, c32(CH_PER_BLK)), ch_local)

        # --- program -> sequence / chunk ---
        idx_seq = buffer_ops.buffer_load(batch_r, pid_x, vec_width=1, dtype=i32)
        chunk_off = buffer_ops.buffer_load(choff_r, pid_x, vec_width=1, dtype=i32)

        seq_start = buffer_ops.buffer_load(qsl_r, idx_seq, vec_width=1, dtype=i32)
        seq_end = buffer_ops.buffer_load(qsl_r, add(idx_seq, c32(1)), vec_width=1, dtype=i32)
        seqlen = sub(seq_end, seq_start)

        token_offset = mul(c32(BLOCK_M), chunk_off)
        p = add(token_offset, token_local)                  # within-seq position
        valid_token = arith.cmpi(CmpIPredicate.slt, p, seqlen)

        is_chunk0 = arith.cmpi(CmpIPredicate.eq, chunk_off, c32(0))

        # in/out conv_state slot (current_first/last index == 0 when APC off)
        in_coord = buffer_ops.buffer_load(
            ci_r, mul(idx_seq, sci), vec_width=1, dtype=i32
        )

        # has_initial_state[idx_seq] (bool, 1 byte)
        hi_i8 = buffer_ops.buffer_load(hi_r, idx_seq, vec_width=1, dtype=T.i8)
        hi_nz = arith.cmpi(CmpIPredicate.ne, hi_i8, arith.constant(0, type=T.i8))
        has_prior = arith.andi(is_chunk0, hi_nz)

        # --- load weights (fp32) ---
        w_base = mul(channel, sw0)
        w_taps = []
        for j in fx.range_constexpr(W):
            wj = buffer_ops.buffer_load(
                w_r, add(w_base, mul(c32(j), sw1)), vec_width=1, dtype=elem_dtype
            )
            w_taps.append(arith.extf(T.f32, wj))

        # --- bias -> acc ---
        if fx.const_expr(HAS_BIAS):
            b_bf = buffer_ops.buffer_load(b_r, channel, vec_width=1, dtype=elem_dtype)
            acc = arith.extf(T.f32, b_bf)
        else:
            acc = cf(0.0)

        ch_sx = mul(channel, sx0)

        def in_seq(within_pos):
            return arith.andi(
                arith.cmpi(CmpIPredicate.sge, within_pos, c32(0)),
                arith.cmpi(CmpIPredicate.slt, within_pos, seqlen),
            )

        def load_x_at(within_pos):
            """Load x[channel, seq_start + within_pos] as fp32 (0 outside [0,seqlen)).

            The upper bound matters: threads past the sequence end must not read
            beyond x's allocation (can fault on page-aligned tensors).
            """
            in_range = in_seq(within_pos)
            cu = add(seq_start, within_pos)
            off = add(ch_sx, mul(cu, sx1))
            safe_off = arith.select(in_range, off, c32(0))
            val = buffer_ops.buffer_load(x_r, safe_off, vec_width=1, dtype=elem_dtype)
            val = arith.extf(T.f32, val)
            return arith.select(in_range, val, cf(0.0)), in_range

        # window values arr[d] = x[p - d], d = 0..W-1
        #
        # A wave (64 lanes) covers 64 contiguous tokens of one channel
        # (lane == token_local), so x[p-d] for lane L is just lane (L-d)'s x[p].
        # We load x[p] once (coalesced) and obtain the shifted taps via
        # ``ds_bpermute`` within the wave -- avoiding (W-1)x redundant DRAM loads.
        # Only the first ``d`` lanes of each wave (whose source lane falls outside
        # the wave) take a scalar global / conv-state fallback.
        xt_f32, _ = load_x_at(p)
        xt_i32 = arith.bitcast(i32, xt_f32)
        arr = [xt_f32]
        for d in fx.range_constexpr(1, W):
            # cross-lane shift-down by d (source lane = token_local - d)
            src_lane = sub(token_local, c32(d))
            shuf_i = rocdl.ds_bpermute(i32, mul(src_lane, c32(4)), xt_i32)
            shuf_f = arith.bitcast(T.f32, shuf_i)

            use_shuf = arith.cmpi(CmpIPredicate.sge, token_local, c32(d))
            need_glob = arith.cmpi(CmpIPredicate.slt, token_local, c32(d))
            pos = sub(p, c32(d))
            in_range = in_seq(pos)
            cu = add(seq_start, pos)
            real_off = add(ch_sx, mul(cu, sx1))
            safe_off = arith.select(arith.andi(need_glob, in_range), real_off, c32(0))
            gv = arith.extf(
                T.f32,
                buffer_ops.buffer_load(x_r, safe_off, vec_width=1, dtype=elem_dtype),
            )
            gv = arith.select(in_range, gv, cf(0.0))
            # blend conv_state for the pre-sequence region (pos < 0)
            need_state = arith.andi(
                arith.cmpi(CmpIPredicate.slt, pos, c32(0)), has_prior
            )
            st_idx = add(c32(STATE_LEN), pos)
            cs_off = add(
                add(mul(in_coord, scs0), mul(channel, scs1)), mul(st_idx, scs2)
            )
            safe_cs = arith.select(need_state, cs_off, c32(0))
            cs_v = arith.extf(
                T.f32, buffer_ops.buffer_load(cs_r, safe_cs, vec_width=1, dtype=elem_dtype)
            )
            gv = arith.select(need_state, cs_v, gv)

            arr.append(arith.select(use_shuf, shuf_f, gv))

        # --- depthwise conv: acc += sum_j w[j] * x[p - (W-1-j)] ---
        for j in fx.range_constexpr(W):
            acc = arith.addf(acc, arith.mulf(w_taps[j], arr[W - 1 - j]))

        # --- SiLU: acc * 1/(1 + exp2(acc * -log2e)) ---
        if fx.const_expr(SILU):
            neg = arith.mulf(acc, cf(-_LOG2E))
            e = rocdl.exp2(T.f32, neg)
            den = arith.addf(cf(1.0), e)
            r = rocdl.rcp(T.f32, den)
            acc = arith.mulf(acc, r)

        acc_out = arith.truncf(elem_dtype, acc)

        # --- scatter to Q / K / V (each channel falls in exactly one block) ---
        cu_tok = add(seq_start, p)
        is_q = arith.cmpi(CmpIPredicate.slt, channel, kd)
        vstart = mul(kd, c32(2))
        is_k = arith.andi(
            arith.cmpi(CmpIPredicate.sge, channel, kd),
            arith.cmpi(CmpIPredicate.slt, channel, vstart),
        )
        is_v = arith.andi(
            arith.cmpi(CmpIPredicate.sge, channel, vstart),
            arith.cmpi(CmpIPredicate.slt, channel, dim),
        )

        def guarded_store(cond, base_ptr, tok_stride, dim_stride, feat):
            off = add(mul(cu_tok, tok_stride), mul(feat, dim_stride))
            _if = scf.IfOp(arith.andi(valid_token, cond))
            with ir.InsertionPoint(_if.then_block):
                buffer_ops.buffer_store(acc_out, base_ptr, off)
                scf.YieldOp([])

        guarded_store(is_q, q_r, qs0, qs1, channel)
        guarded_store(is_k, k_r, ks0, ks1, sub(channel, kd))
        guarded_store(is_v, v_r, vs0, vs1, sub(channel, vstart))

        # --- conv_state writeback (chunk 0 only) ---
        # New state holds the last STATE_LEN input values of the whole sequence.
        # Slot t (0..STATE_LEN-1) maps to within-seq position pos_x = seqlen-STATE_LEN+t
        # (unified for both seqlen>=STATE_LEN and seqlen<STATE_LEN). When pos_x<0 the
        # value comes from the prior conv_state (shift-left) or zero.
        # out_coord == in_coord when APC off / current_last_index == 0.
        if fx.const_expr(STATE_LEN > 0):
            zero_e = arith.constant(0.0, type=elem_dtype)
            pos_x = add(sub(seqlen, c32(STATE_LEN)), token_local)
            x_in_range = arith.andi(
                arith.cmpi(CmpIPredicate.sge, pos_x, c32(0)),
                arith.cmpi(CmpIPredicate.slt, pos_x, seqlen),
            )
            cu_src = add(seq_start, pos_x)
            safe_x = arith.select(x_in_range, add(ch_sx, mul(cu_src, sx1)), c32(0))
            val_x = buffer_ops.buffer_load(x_r, safe_x, vec_width=1, dtype=elem_dtype)
            # prior (shift-left) source for the seqlen<STATE_LEN region
            need_prior = arith.andi(
                arith.cmpi(CmpIPredicate.slt, pos_x, c32(0)), has_prior
            )
            st_in = add(token_local, seqlen)
            safe_pr = arith.select(
                need_prior,
                add(add(mul(in_coord, scs0), mul(channel, scs1)), mul(st_in, scs2)),
                c32(0),
            )
            val_pr = buffer_ops.buffer_load(cs_r, safe_pr, vec_width=1, dtype=elem_dtype)
            wb_val = arith.select(
                x_in_range, val_x, arith.select(need_prior, val_pr, zero_e)
            )
            cs_wr = add(
                add(mul(in_coord, scs0), mul(channel, scs1)),
                mul(token_local, scs2),
            )
            should_write = arith.andi(
                arith.andi(
                    is_chunk0, arith.cmpi(CmpIPredicate.slt, token_local, c32(STATE_LEN))
                ),
                arith.cmpi(CmpIPredicate.slt, channel, dim),
            )
            # all sources read above; barrier so reads finish before overwrite
            # (in_coord == out_coord aliases the prior-state reads).
            fx.gpu.barrier()
            _ifw = scf.IfOp(should_write)
            with ir.InsertionPoint(_ifw.then_block):
                buffer_ops.buffer_store(wb_val, cs_r, cs_wr)
                scf.YieldOp([])

    grid_y = lambda d: (d + CH_PER_BLK - 1) // CH_PER_BLK  # noqa: E731

    @flyc.jit
    def launch(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
        num_programs: Int32,
        grid_y_dim: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = arith.index_cast(T.index, num_programs)
        gy = arith.index_cast(T.index, grid_y_dim)
        conv1d_kernel(
            x_ptr, w_ptr, bias_ptr, cs_ptr, cache_idx_ptr, has_init_ptr,
            qsl_ptr, batch_ptr, chunk_off_ptr, q_ptr, k_ptr, v_ptr,
            dim, kd, vd, sx0, sx1, sw0, sw1, scs0, scs1, scs2, sci,
            qs0, qs1, ks0, ks1, vs0, vs1,
        ).launch(
            grid=(gx, gy, 1),
            block=(BLOCK_M * CH_PER_BLK, 1, 1),
            stream=stream,
        )

    launch._grid_y = grid_y
    launch._ch_per_blk = CH_PER_BLK
    launch._block_m = BLOCK_M
    return launch


def build_causal_conv1d_flydsl_lds_module(
    width: int,
    has_bias: bool,
    silu: bool,
    tm: int = 64,
    tn: int = 64,
    block_threads: int = 256,
    dtype_str: str = "bf16",
):
    """LDS-staged 2D tile kernel (transposed load/store for full coalescing).

    The input ``x`` is token-contiguous while the q/k/v outputs are
    feature-contiguous. A 1D thread map can only coalesce one of the two. This
    kernel stages a ``[TN channels x (TM + STATE_LEN) tokens]`` tile of ``x`` in
    LDS:

        Phase 1 (load):   threads walk the tile token-major -> coalesced global
                          loads of x, written into LDS (fp32).
        Phase 2 (store):  threads are re-mapped so a wave spans TN *channels* of
                          one token -> the depthwise conv reads its window from
                          LDS and the q/k/v stores are coalesced along feature.

    This mirrors what the Triton tile kernel gets implicitly from its register
    tile, and is the configuration the benchmark exercises (TM == BLOCK_M == 64,
    q/k/v channel boundaries are multiples of 64 so each tile targets exactly one
    output).
    """
    assert _FLYDSL_AVAILABLE, "flydsl is not installed"
    assert width in (2, 3, 4)
    assert block_threads % tn == 0, "block_threads must be a multiple of tn"
    assert tm % (block_threads // tn) == 0

    W = width
    SL = W - 1
    ROW = tm + SL                     # LDS columns per channel (odd-ish -> few bank conflicts)
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)
    TM, TN, BT = tm, tn, block_threads
    TOK_STEP = BT // TN               # tokens advanced per compute iteration
    NTOK = TM // TOK_STEP             # compute iterations per thread
    NLDS = TN * ROW
    NLOAD = (NLDS + BT - 1) // BT     # load iterations per thread
    LDS_BYTES = NLDS * 4              # fp32 staging
    # Coalesced load decomposition (shift-based, no integer divide):
    #   tid -> (f_base = tid >> log2(TM), t_const = tid & (TM-1))
    # A wave reads consecutive tokens of one feature (fully coalesced); each
    # thread loads ELEMS body tokens stepping FSTEP features apart.
    assert (TM & (TM - 1)) == 0, "TM must be a power of two for shift decomposition"
    assert TN == TM, "halo load decomposition assumes TN == TM"
    TM_LOG2 = TM.bit_length() - 1
    FSTEP = BT // TM                  # feature stride between a thread's loads
    ELEMS = TN // FSTEP               # body elements loaded per thread

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"causal_conv1d_lds_w{W}_tm{TM}_tn{TN}_{dtype_str}",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_BYTES

    @flyc.kernel
    def conv1d_lds_kernel(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
    ):
        i32 = T.i32
        elem_dtype = T.bf16 if dtype_str == "bf16" else T.f16

        def _v(x):
            return x.ir_value() if hasattr(x, "ir_value") else x

        dim = _v(dim); kd = _v(kd); vd = _v(vd)
        sx0 = _v(sx0); sx1 = _v(sx1); sw0 = _v(sw0); sw1 = _v(sw1)
        scs0 = _v(scs0); scs1 = _v(scs1); scs2 = _v(scs2); sci = _v(sci)
        qs0 = _v(qs0); qs1 = _v(qs1); ks0 = _v(ks0); ks1 = _v(ks1)
        vs0 = _v(vs0); vs1 = _v(vs1)

        def c32(v):
            return arith.constant(int(v), type=i32)

        def cf(v):
            return arith.constant(float(v), type=T.f32)

        def to_i32(v):
            return arith.index_cast(i32, v)

        def mul(a, b):
            return arith.muli(a, b)

        def add(a, b):
            return arith.addi(a, b)

        def sub(a, b):
            return arith.subi(a, b)

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        x_r = _rsrc(x_ptr)
        w_r = _rsrc(w_ptr)
        b_r = _rsrc(bias_ptr)
        cs_r = _rsrc(cs_ptr)
        ci_r = _rsrc(cache_idx_ptr)
        hi_r = _rsrc(has_init_ptr)
        qsl_r = _rsrc(qsl_ptr)
        batch_r = _rsrc(batch_ptr)
        choff_r = _rsrc(chunk_off_ptr)
        q_r = _rsrc(q_ptr)
        k_r = _rsrc(k_ptr)
        v_r = _rsrc(v_ptr)

        lds = SmemPtr(allocator.get_base(), lds_off, T.f32, shape=(NLDS,))
        lds.get()  # materialize the memref view in the entry block (dominates all uses)

        tid = to_i32(fx.thread_idx.x)
        pid_x = to_i32(fx.block_idx.x)
        pid_y = to_i32(fx.block_idx.y)
        ch_base = mul(pid_y, c32(TN))

        idx_seq = buffer_ops.buffer_load(batch_r, pid_x, vec_width=1, dtype=i32)
        chunk_off = buffer_ops.buffer_load(choff_r, pid_x, vec_width=1, dtype=i32)
        seq_start = buffer_ops.buffer_load(qsl_r, idx_seq, vec_width=1, dtype=i32)
        seq_end = buffer_ops.buffer_load(
            qsl_r, add(idx_seq, c32(1)), vec_width=1, dtype=i32
        )
        seqlen = sub(seq_end, seq_start)
        token_offset = mul(c32(TM), chunk_off)
        is_chunk0 = arith.cmpi(CmpIPredicate.eq, chunk_off, c32(0))

        # NOTE: conv_state metadata (cache_indices / has_initial_state) is loaded
        # lazily inside the is_chunk0 branches below, since only chunk-0 blocks
        # ever touch conv_state. This keeps those scalar VMEM loads (and their
        # ~600-cycle latency on the kernel-entry critical path) off the other
        # chunks entirely.

        def in_seq(pos):
            return arith.andi(
                arith.cmpi(CmpIPredicate.sge, pos, c32(0)),
                arith.cmpi(CmpIPredicate.slt, pos, seqlen),
            )

        def load_in_coord():
            return buffer_ops.buffer_load(
                ci_r, mul(idx_seq, sci), vec_width=1, dtype=i32
            )

        def load_has_prior():
            hi8 = buffer_ops.buffer_load(hi_r, idx_seq, vec_width=1, dtype=T.i8)
            return arith.cmpi(CmpIPredicate.ne, hi8, arith.constant(0, type=T.i8))

        # ---- Phase 1: cooperative load of the tile into LDS ----
        # Shift-based decomposition (no integer divide): a wave reads consecutive
        # tokens of one feature (coalesced); addressing is incremental.
        t_const = arith.andi(tid, c32(TM - 1))       # token within tile, 0..TM-1
        f_base = arith.shrui(tid, c32(TM_LOG2))       # 0..FSTEP-1

        # Body region (LDS columns SL..ROW-1): tile tokens, within_pos >= 0 so
        # only the upper seq bound + feature bound matter.
        pos_body = add(token_offset, t_const)
        pos_ok = arith.cmpi(CmpIPredicate.slt, pos_body, seqlen)
        cu_body = add(seq_start, pos_body)
        cu_term = mul(cu_body, sx1)
        lds_col_body = add(c32(SL), t_const)
        base_addr = add(mul(add(ch_base, f_base), sx0), cu_term)
        fstep_addr = mul(c32(FSTEP), sx0)
        # Issue all body loads first (in-flight together) then consume, so the
        # scalar loads overlap instead of serializing load -> use -> next load.
        # (A block-level fast/slow split was tried here and regressed: the extra
        # control flow + code duplication outweighed the "free" address selects
        # that already hide under VMEM-wait in this memory-bound kernel.)
        # Strength-reduced addressing: hoist the feature-base products and walk
        # the per-element stride with running adds instead of recomputing
        # flocal*sx0 / flocal*ROW each iteration (kills the unrolled v_mul_lo /
        # v_add_lshl chain that showed up as ~16% "other" in the ATT trace).
        chf = add(ch_base, f_base)                         # feature base for this thread
        lds_base = add(mul(f_base, c32(ROW)), lds_col_body)  # LDS index for j=0
        raws = []
        valids = []
        cur_off = base_addr
        for j in fx.range_constexpr(ELEMS):
            valid = arith.andi(
                pos_ok,
                arith.cmpi(CmpIPredicate.slt, add(chf, c32(j * FSTEP)), dim),
            )
            off = arith.select(valid, cur_off, c32(0))
            raws.append(buffer_ops.buffer_load(x_r, off, vec_width=1, dtype=elem_dtype))
            valids.append(valid)
            if fx.const_expr(j + 1 < ELEMS):
                cur_off = add(cur_off, fstep_addr)
        for j in fx.range_constexpr(ELEMS):
            xv = arith.select(valids[j], arith.extf(T.f32, raws[j]), cf(0.0))
            lds_idx = lds_base if j == 0 else add(lds_base, c32(j * FSTEP * ROW))
            lds.store(xv, [lds_idx])

        # Halo region (LDS columns 0..SL-1): the KW-1 prior tokens. For chunk>0
        # these come straight from x (within_halo >= 0); only chunk-0 may need the
        # conv_state blend, so that path (and its metadata loads) is gated.
        if fx.const_expr(SL > 0):
            halo_feat = add(ch_base, t_const)
            within_halo = add(sub(token_offset, c32(SL)), f_base)
            rng_h = in_seq(within_halo)
            cu_h = add(seq_start, within_halo)
            safe_h = arith.select(rng_h, add(mul(halo_feat, sx0), mul(cu_h, sx1)), c32(0))
            xv_h = arith.select(
                rng_h,
                arith.extf(
                    T.f32, buffer_ops.buffer_load(x_r, safe_h, vec_width=1, dtype=elem_dtype)
                ),
                cf(0.0),
            )
            _ifcs = scf.IfOp(is_chunk0, [T.f32], has_else=True)
            with ir.InsertionPoint(_ifcs.then_block):
                need = arith.andi(
                    arith.cmpi(CmpIPredicate.slt, within_halo, c32(0)), load_has_prior()
                )
                st_idx = add(c32(SL), within_halo)
                cs_off = add(
                    add(mul(load_in_coord(), scs0), mul(halo_feat, scs1)),
                    mul(st_idx, scs2),
                )
                safe_cs = arith.select(need, cs_off, c32(0))
                cs_v = arith.extf(
                    T.f32,
                    buffer_ops.buffer_load(cs_r, safe_cs, vec_width=1, dtype=elem_dtype),
                )
                scf.YieldOp([arith.select(need, cs_v, xv_h)])
            with ir.InsertionPoint(_ifcs.else_block):
                scf.YieldOp([xv_h])
            val_h = _ifcs.results[0]
            do_h = arith.andi(
                arith.cmpi(CmpIPredicate.slt, f_base, c32(SL)),
                arith.cmpi(CmpIPredicate.slt, halo_feat, dim),
            )
            _ifh = scf.IfOp(do_h)
            with ir.InsertionPoint(_ifh.then_block):
                lds.store(val_h, [add(mul(t_const, c32(ROW)), f_base)])
                scf.YieldOp([])

        fx.gpu.barrier()

        # ---- Phase 2: compute + feature-coalesced store ----
        ch_local = arith.remui(tid, c32(TN))
        tok_group = arith.divui(tid, c32(TN))
        channel = add(ch_base, ch_local)

        w_base = mul(channel, sw0)
        w_taps = []
        for j in fx.range_constexpr(W):
            w_taps.append(
                arith.extf(
                    T.f32,
                    buffer_ops.buffer_load(
                        w_r, add(w_base, mul(c32(j), sw1)), vec_width=1, dtype=elem_dtype
                    ),
                )
            )
        if fx.const_expr(HAS_BIAS):
            bias_f = arith.extf(
                T.f32, buffer_ops.buffer_load(b_r, channel, vec_width=1, dtype=elem_dtype)
            )
        else:
            bias_f = cf(0.0)

        vstart = mul(kd, c32(2))
        ch_row = mul(ch_local, c32(ROW))

        # Compute all NTOK outputs into registers first.
        acc_outs = []
        cu_toks = []
        valids = []
        for i in fx.range_constexpr(NTOK):
            tj = add(tok_group, c32(i * TOK_STEP))
            p = add(token_offset, tj)
            valids.append(arith.cmpi(CmpIPredicate.slt, p, seqlen))
            cu_toks.append(add(seq_start, p))
            acc = bias_f
            for j in fx.range_constexpr(W):
                xw = lds.load([add(ch_row, add(tj, c32(j)))])
                acc = arith.addf(acc, arith.mulf(w_taps[j], xw))
            if fx.const_expr(SILU):
                e_ = rocdl.exp2(T.f32, arith.mulf(acc, cf(-_LOG2E)))
                acc = arith.mulf(acc, rocdl.rcp(T.f32, arith.addf(cf(1.0), e_)))
            acc_outs.append(arith.truncf(elem_dtype, acc))

        # Store via one block-uniform output branch (q/k/v boundaries are
        # multiples of TN, so a tile targets exactly one output). all_tok fast
        # path drops the per-token bound check for fully-interior tiles.
        all_tok = arith.cmpi(
            CmpIPredicate.sle, add(token_offset, c32(TM)), seqlen
        )

        def emit_store(block_cond, base_r, ts, ds, feat):
            _ifb = scf.IfOp(block_cond)
            with ir.InsertionPoint(_ifb.then_block):
                feat_term = mul(feat, ds)
                # Strength-reduce the per-token output offset: cu_toks[i] differs
                # from cu_toks[0] by i*TOK_STEP, so off_i = base_off + i*(TOK_STEP*ts).
                # Hoist base_off / step_ts and walk with adds (no per-token mul).
                base_off = add(mul(cu_toks[0], ts), feat_term)
                step_ts = mul(c32(TOK_STEP), ts)
                _ift = scf.IfOp(all_tok, has_else=True)
                with ir.InsertionPoint(_ift.then_block):
                    cur = base_off
                    for i in fx.range_constexpr(NTOK):
                        buffer_ops.buffer_store(acc_outs[i], base_r, cur)
                        if fx.const_expr(i + 1 < NTOK):
                            cur = add(cur, step_ts)
                    scf.YieldOp([])
                with ir.InsertionPoint(_ift.else_block):
                    cur = base_off
                    for i in fx.range_constexpr(NTOK):
                        _ifv = scf.IfOp(valids[i])
                        with ir.InsertionPoint(_ifv.then_block):
                            buffer_ops.buffer_store(acc_outs[i], base_r, cur)
                            scf.YieldOp([])
                        if fx.const_expr(i + 1 < NTOK):
                            cur = add(cur, step_ts)
                    scf.YieldOp([])
                scf.YieldOp([])

        blk_q = arith.cmpi(CmpIPredicate.sle, add(ch_base, c32(TN)), kd)
        blk_k = arith.andi(
            arith.cmpi(CmpIPredicate.sge, ch_base, kd),
            arith.cmpi(CmpIPredicate.sle, add(ch_base, c32(TN)), vstart),
        )
        blk_v = arith.cmpi(CmpIPredicate.sge, ch_base, vstart)
        emit_store(blk_q, q_r, qs0, qs1, channel)
        emit_store(blk_k, k_r, ks0, ks1, sub(channel, kd))
        emit_store(blk_v, v_r, vs0, vs1, sub(channel, vstart))

        # ---- conv_state writeback (chunk 0): slot t = within-seq seqlen-SL+t ----
        # Only chunk-0 blocks write conv_state, so gate the whole thing (loads +
        # barrier + store) on is_chunk0 (block-uniform) to skip it entirely on
        # the other chunks (the vast majority for long sequences).
        if fx.const_expr(SL > 0):
            _ifc0 = scf.IfOp(is_chunk0)
            with ir.InsertionPoint(_ifc0.then_block):
                zero_e = arith.constant(0.0, type=elem_dtype)
                in_coord = load_in_coord()
                t_slot = tok_group  # 0..TOK_STEP-1 ; only < SL used
                pos_x = add(sub(seqlen, c32(SL)), t_slot)
                x_in = in_seq(pos_x)
                ch_sx = mul(channel, sx0)
                cu_src = add(seq_start, pos_x)
                safe_x = arith.select(x_in, add(ch_sx, mul(cu_src, sx1)), c32(0))
                val_x = buffer_ops.buffer_load(x_r, safe_x, vec_width=1, dtype=elem_dtype)
                need_pr = arith.andi(
                    arith.cmpi(CmpIPredicate.slt, pos_x, c32(0)), load_has_prior()
                )
                st_in = add(t_slot, seqlen)
                safe_pr = arith.select(
                    need_pr,
                    add(add(mul(in_coord, scs0), mul(channel, scs1)), mul(st_in, scs2)),
                    c32(0),
                )
                val_pr = buffer_ops.buffer_load(cs_r, safe_pr, vec_width=1, dtype=elem_dtype)
                wb_val = arith.select(x_in, val_x, arith.select(need_pr, val_pr, zero_e))
                cs_wr = add(
                    add(mul(in_coord, scs0), mul(channel, scs1)), mul(t_slot, scs2)
                )
                should_write = arith.andi(
                    arith.cmpi(CmpIPredicate.slt, t_slot, c32(SL)),
                    arith.cmpi(CmpIPredicate.slt, channel, dim),
                )
                fx.gpu.barrier()
                _ifw = scf.IfOp(should_write)
                with ir.InsertionPoint(_ifw.then_block):
                    buffer_ops.buffer_store(wb_val, cs_r, cs_wr)
                    scf.YieldOp([])
                scf.YieldOp([])

    @flyc.jit
    def launch(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
        num_programs: Int32,
        grid_y_dim: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx = arith.index_cast(T.index, num_programs)
        gy = arith.index_cast(T.index, grid_y_dim)
        conv1d_lds_kernel(
            x_ptr, w_ptr, bias_ptr, cs_ptr, cache_idx_ptr, has_init_ptr,
            qsl_ptr, batch_ptr, chunk_off_ptr, q_ptr, k_ptr, v_ptr,
            dim, kd, vd, sx0, sx1, sw0, sw1, scs0, scs1, scs2, sci,
            qs0, qs1, ks0, ks1, vs0, vs1,
        ).launch(grid=(gx, gy, 1), block=(BT, 1, 1), stream=stream)

    launch._tn = TN
    launch._tm = TM
    return launch


def build_causal_conv1d_flydsl_v2_module(
    width: int,
    has_bias: bool,
    silu: bool,
    tm: int = 64,
    tn: int = 64,
    block_threads: int = 256,
    dtype_str: str = "bf16",
):
    """FlyDSL port whose logic is byte-for-byte identical to the hand-written
    HIP ``conv1d_v11_t`` kernel, so the only variable is the toolchain
    (Fly/ROCDL MLIR vs hipcc). Differences vs the ``_lds`` module above:

      * LDS stages ``x`` as **bf16** (half the LDS of the fp32 ``_lds`` kernel);
        the bf16->f32 conversion is deferred to the conv inner loop.
      * Explicit **fast/slow load split**: a fully-interior tile takes a
        bounds-free coalesced path; boundary tiles take a sequence-relative
        path that blends conv_state at the halo.
      * The store re-stages results through LDS (**transpose**) so the
        compute thread-map (feat_local=tid>>2, tok_group=tid&3) still yields a
        feature-coalesced global store.
      * conv_state writeback (chunk 0) reads the sequence tail from x; the load
        barrier already orders the halo conv_state reads before this write.
    """
    assert _FLYDSL_AVAILABLE, "flydsl is not installed"
    assert width in (2, 3, 4)
    assert tm == 64 and tn == 64 and block_threads == 256, \
        "flydslv2 mirrors v11's fixed TM=TN=64, 256-thread tile"

    W = width
    KW = W
    SL = W - 1
    TM, TN, BT = tm, tn, block_threads
    LDS_PAD = TM + KW          # halo(KW-1) + body(TM) + pad(1)
    EPT = TM // 4              # outputs per thread (4 token groups)
    FG = BT // TM              # feat-base groups in cooperative load (=4)
    ELEMS = TN * TM // BT      # body features loaded per thread (=16)
    LOG2_TM = TM.bit_length() - 1   # =6
    NLDS = TN * LDS_PAD
    STORE_PAD = TN + 1
    LDS_BYTES = NLDS * 2       # bf16 staging
    HAS_BIAS = bool(has_bias)
    SILU = bool(silu)

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=f"causal_conv1d_v2_w{W}_tm{TM}_{dtype_str}",
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + LDS_BYTES

    @flyc.kernel
    def conv1d_v2_kernel(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
    ):
        i32 = T.i32
        elem_dtype = T.bf16 if dtype_str == "bf16" else T.f16

        def _v(x):
            return x.ir_value() if hasattr(x, "ir_value") else x

        dim = _v(dim); kd = _v(kd); vd = _v(vd)
        sx0 = _v(sx0); sx1 = _v(sx1); sw0 = _v(sw0); sw1 = _v(sw1)
        scs0 = _v(scs0); scs1 = _v(scs1); scs2 = _v(scs2); sci = _v(sci)
        qs0 = _v(qs0); qs1 = _v(qs1); ks0 = _v(ks0); ks1 = _v(ks1)
        vs0 = _v(vs0); vs1 = _v(vs1)

        def c32(v):
            return arith.constant(int(v), type=i32)

        def cf(v):
            return arith.constant(float(v), type=T.f32)

        def to_i32(v):
            return arith.index_cast(i32, v)

        def mul(a, b):
            return arith.muli(a, b)

        def add(a, b):
            return arith.addi(a, b)

        def sub(a, b):
            return arith.subi(a, b)

        def f32(bf):
            return arith.extf(T.f32, bf)

        def _rsrc(ptr):
            return buffer_ops.create_buffer_resource(ptr, max_size=True)

        x_r = _rsrc(x_ptr); w_r = _rsrc(w_ptr); b_r = _rsrc(bias_ptr)
        cs_r = _rsrc(cs_ptr); ci_r = _rsrc(cache_idx_ptr); hi_r = _rsrc(has_init_ptr)
        qsl_r = _rsrc(qsl_ptr); batch_r = _rsrc(batch_ptr); choff_r = _rsrc(chunk_off_ptr)
        q_r = _rsrc(q_ptr); k_r = _rsrc(k_ptr); v_r = _rsrc(v_ptr)

        lds = SmemPtr(allocator.get_base(), lds_off, elem_dtype, shape=(NLDS,))
        lds.get()

        def lds_st(val, idx):
            lds.store(val, [idx])

        def lds_ld(idx):
            return lds.load([idx])

        tid = to_i32(fx.thread_idx.x)
        pid_x = to_i32(fx.block_idx.x)
        pid_y = to_i32(fx.block_idx.y)

        seq_idx = buffer_ops.buffer_load(batch_r, pid_x, vec_width=1, dtype=i32)
        chunk_idx = buffer_ops.buffer_load(choff_r, pid_x, vec_width=1, dtype=i32)
        seq_start = buffer_ops.buffer_load(qsl_r, seq_idx, vec_width=1, dtype=i32)
        seq_end = buffer_ops.buffer_load(qsl_r, add(seq_idx, c32(1)), vec_width=1, dtype=i32)
        seqlen = sub(seq_end, seq_start)

        feat_start = mul(pid_y, c32(TN))
        tok_start = mul(chunk_idx, c32(TM))
        is_chunk0 = arith.cmpi(CmpIPredicate.eq, chunk_idx, c32(0))

        feat_local = arith.shrui(tid, c32(2))
        tok_group = arith.andi(tid, c32(3))
        tok_base = mul(tok_group, c32(EPT))
        gfeat = add(feat_start, feat_local)
        feat_valid = arith.cmpi(CmpIPredicate.slt, gfeat, dim)

        # ── weights + bias (fp32) issued early ──
        w_base = mul(gfeat, sw0)
        w_taps = []
        for j in fx.range_constexpr(W):
            w_taps.append(f32(buffer_ops.buffer_load(
                w_r, add(w_base, mul(c32(j), sw1)), vec_width=1, dtype=elem_dtype)))
        if fx.const_expr(HAS_BIAS):
            bias_f = f32(buffer_ops.buffer_load(b_r, gfeat, vec_width=1, dtype=elem_dtype))
        else:
            bias_f = cf(0.0)

        # ── cooperative load into bf16 LDS ──
        t_const = arith.andi(tid, c32(TM - 1))
        f_base = arith.shrui(tid, c32(LOG2_TM))
        hc = arith.shrui(tid, c32(6))
        hf = arith.andi(tid, c32(63))
        tok_gbase = add(sub(add(seq_start, tok_start), c32(KW - 1)), c32(0))  # seq_start+tok_start-(KW-1)
        gt1 = add(tok_gbase, add(t_const, c32(KW - 1)))                      # seq_start+tok_start+t_const

        all_feat = arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), dim)
        all_tok1 = arith.cmpi(CmpIPredicate.slt, add(tok_start, c32(TM - 1)), seqlen)
        all_tok2 = arith.cmpi(CmpIPredicate.sge, tok_start, c32(KW - 1))
        fast = arith.andi(arith.andi(all_feat, all_tok1), all_tok2)

        _ifld = scf.IfOp(fast, has_else=True)
        with ir.InsertionPoint(_ifld.then_block):
            # fast path: fully interior, coalesced, no bounds/state
            cur = add(mul(add(feat_start, f_base), sx0), gt1)
            fstep = mul(c32(FG), sx0)
            raws = []
            for j in fx.range_constexpr(ELEMS):
                raws.append(buffer_ops.buffer_load(x_r, cur, vec_width=1, dtype=elem_dtype))
                if fx.const_expr(j + 1 < ELEMS):
                    cur = add(cur, fstep)
            # issue halo load early (before LDS body stores) to overlap latency
            do_halo = arith.cmpi(CmpIPredicate.slt, hc, c32(KW - 1))
            prefix_off = arith.select(
                do_halo, add(mul(add(feat_start, hf), sx0), add(tok_gbase, hc)), c32(0))
            prefix_v = buffer_ops.buffer_load(x_r, prefix_off, vec_width=1, dtype=elem_dtype)
            lds_idx = add(mul(f_base, c32(LDS_PAD)), add(t_const, c32(KW - 1)))
            for j in fx.range_constexpr(ELEMS):
                cur_idx = lds_idx if j == 0 else add(lds_idx, c32(j * FG * LDS_PAD))
                lds_st(raws[j], cur_idx)
            _ifh = scf.IfOp(do_halo)
            with ir.InsertionPoint(_ifh.then_block):
                lds_st(prefix_v, add(mul(hf, c32(LDS_PAD)), hc))
                scf.YieldOp([])
            scf.YieldOp([])
        with ir.InsertionPoint(_ifld.else_block):
            # slow path: sequence-relative bounds (still coalesced)
            zero_e = arith.constant(0.0, type=elem_dtype)
            body_wp = add(tok_start, t_const)
            body_ok = arith.cmpi(CmpIPredicate.slt, body_wp, seqlen)
            sl_m1 = arith.select(
                arith.cmpi(CmpIPredicate.sgt, seqlen, c32(0)), sub(seqlen, c32(1)), c32(0))
            body_gt = add(seq_start, arith.select(body_ok, body_wp, sl_m1))
            for j in fx.range_constexpr(ELEMS):
                gf = add(add(feat_start, f_base), c32(j * FG))
                gf_ok = arith.cmpi(CmpIPredicate.slt, gf, dim)
                safe_gf = arith.select(gf_ok, gf, c32(0))
                raw = buffer_ops.buffer_load(
                    x_r, add(mul(safe_gf, sx0), body_gt), vec_width=1, dtype=elem_dtype)
                val = arith.select(arith.andi(body_ok, gf_ok), raw, zero_e)
                lds_st(val, add(mul(add(f_base, c32(j * FG)), c32(LDS_PAD)),
                                add(t_const, c32(KW - 1))))
            # halo column with conv_state blend at chunk0
            do_halo = arith.cmpi(CmpIPredicate.slt, hc, c32(KW - 1))
            _ifh = scf.IfOp(do_halo)
            with ir.InsertionPoint(_ifh.then_block):
                gf = add(feat_start, hf)
                gf_ok = arith.cmpi(CmpIPredicate.slt, gf, dim)
                wp = sub(add(tok_start, hc), c32(KW - 1))
                wp_in = arith.andi(
                    arith.cmpi(CmpIPredicate.sge, wp, c32(0)),
                    arith.cmpi(CmpIPredicate.slt, wp, seqlen))
                # in-seq source from x
                safe_xoff = arith.select(
                    arith.andi(wp_in, gf_ok), add(mul(gf, sx0), add(seq_start, wp)), c32(0))
                xv = arith.select(
                    arith.andi(wp_in, gf_ok),
                    buffer_ops.buffer_load(x_r, safe_xoff, vec_width=1, dtype=elem_dtype),
                    zero_e)
                # pre-seq source from conv_state (chunk0 + has_init)
                hi8 = buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                hi_nz = arith.cmpi(CmpIPredicate.ne, hi8, arith.constant(0, type=T.i8))
                need_cs = arith.andi(
                    arith.andi(arith.cmpi(CmpIPredicate.slt, wp, c32(0)), is_chunk0),
                    arith.andi(hi_nz, gf_ok))
                in_coord = buffer_ops.buffer_load(ci_r, mul(seq_idx, sci), vec_width=1, dtype=i32)
                slot = add(c32(KW - 1), wp)
                cs_off = arith.select(
                    need_cs,
                    add(add(mul(in_coord, scs0), mul(gf, scs1)), mul(slot, scs2)), c32(0))
                csv = buffer_ops.buffer_load(cs_r, cs_off, vec_width=1, dtype=elem_dtype)
                hv = arith.select(need_cs, csv, xv)
                lds_st(hv, add(mul(hf, c32(LDS_PAD)), hc))
                scf.YieldOp([])
            scf.YieldOp([])

        fx.gpu.barrier()

        # ── compute: acc[e] = bias + sum_k w[k] * x[tok_base+e .. +KW-1] ──
        # The EPT outputs of a thread share their conv window: across e=0..EPT-1
        # and k=0..W-1 only the contiguous LDS span [row_base .. row_base+EPT+W-2]
        # is touched. Load that span ONCE into registers (running offset, so the
        # constant column delta is hoisted out of the address math) and MAC from
        # registers — this avoids the EPT*W scalar memref.loads whose per-tap
        # address recomputation is what inflated flydslv2's VALU count vs v11.
        row_base = add(mul(feat_local, c32(LDS_PAD)), tok_base)
        NSPAN = EPT + W - 1
        xw = []
        for i in fx.range_constexpr(NSPAN):
            idx = row_base if i == 0 else add(row_base, c32(i))
            xw.append(f32(lds_ld(idx)))
        acc = []
        for e in fx.range_constexpr(EPT):
            a = bias_f
            for kk in fx.range_constexpr(W):
                a = arith.addf(a, arith.mulf(w_taps[kk], xw[e + kk]))
            if fx.const_expr(SILU):
                ex = rocdl.exp2(T.f32, arith.mulf(a, cf(-_LOG2E)))
                a = arith.mulf(a, rocdl.rcp(T.f32, arith.addf(cf(1.0), ex)))
            acc.append(a)

        # ── store: transpose through LDS (fast) or direct (slow) ──
        store_fast = arith.andi(
            arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), dim),
            arith.cmpi(CmpIPredicate.slt, add(tok_start, c32(TM - 1)), seqlen))
        vstart = mul(kd, c32(2))
        blk_q = arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), kd)
        blk_k = arith.andi(
            arith.cmpi(CmpIPredicate.sge, feat_start, kd),
            arith.cmpi(CmpIPredicate.sle, add(feat_start, c32(TN)), vstart))
        blk_v = arith.cmpi(CmpIPredicate.sge, feat_start, vstart)

        _ifst = scf.IfOp(store_fast, has_else=True)
        with ir.InsertionPoint(_ifst.then_block):
            fx.gpu.barrier()
            for e in fx.range_constexpr(EPT):
                lds_st(arith.truncf(elem_dtype, acc[e]),
                       add(mul(add(tok_base, c32(e)), c32(STORE_PAD)), feat_local))
            fx.gpu.barrier()
            sf = arith.andi(tid, c32(TN - 1))
            tg = arith.shrui(tid, c32(6))
            tg_ept = mul(tg, c32(EPT))
            tok0 = add(add(seq_start, tok_start), tg_ept)

            def emit_fast(cond, res, ts, ds, fo):
                _ifb = scf.IfOp(cond)
                with ir.InsertionPoint(_ifb.then_block):
                    of = sub(add(feat_start, sf), fo)
                    base_off = add(mul(tok0, ts), mul(of, ds))
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        val = lds_ld(add(mul(add(tg_ept, c32(e)), c32(STORE_PAD)), sf))
                        buffer_ops.buffer_store(val, res, cur)
                        if fx.const_expr(e + 1 < EPT):
                            cur = add(cur, ts)
                    scf.YieldOp([])

            emit_fast(blk_q, q_r, qs0, qs1, c32(0))
            emit_fast(blk_k, k_r, ks0, ks1, kd)
            emit_fast(blk_v, v_r, vs0, vs1, vstart)
            scf.YieldOp([])
        with ir.InsertionPoint(_ifst.else_block):
            def emit_slow(cond, res, ts, ds, fo):
                _ifb = scf.IfOp(arith.andi(cond, feat_valid))
                with ir.InsertionPoint(_ifb.then_block):
                    of = sub(gfeat, fo)
                    base_off = add(mul(add(add(seq_start, tok_start), tok_base), ts),
                                   mul(of, ds))
                    cur = base_off
                    for e in fx.range_constexpr(EPT):
                        tok_ok = arith.cmpi(
                            CmpIPredicate.slt,
                            add(add(tok_start, tok_base), c32(e)), seqlen)
                        _ifv = scf.IfOp(tok_ok)
                        with ir.InsertionPoint(_ifv.then_block):
                            buffer_ops.buffer_store(
                                arith.truncf(elem_dtype, acc[e]), res, cur)
                            scf.YieldOp([])
                        if fx.const_expr(e + 1 < EPT):
                            cur = add(cur, ts)
                    scf.YieldOp([])

            emit_slow(blk_q, q_r, qs0, qs1, c32(0))
            emit_slow(blk_k, k_r, ks0, ks1, kd)
            emit_slow(blk_v, v_r, vs0, vs1, vstart)
            scf.YieldOp([])

        # ── conv_state writeback (chunk 0) ──
        if fx.const_expr(SL > 0):
            _ifc0 = scf.IfOp(is_chunk0)
            with ir.InsertionPoint(_ifc0.then_block):
                zero_e = arith.constant(0.0, type=elem_dtype)
                slot = tok_group
                should = arith.andi(
                    arith.cmpi(CmpIPredicate.slt, slot, c32(KW - 1)),
                    arith.cmpi(CmpIPredicate.slt, gfeat, dim))
                _ifw = scf.IfOp(should)
                with ir.InsertionPoint(_ifw.then_block):
                    in_coord = buffer_ops.buffer_load(
                        ci_r, mul(seq_idx, sci), vec_width=1, dtype=i32)
                    pos_x = add(sub(seqlen, c32(KW - 1)), slot)
                    x_in = arith.cmpi(CmpIPredicate.sge, pos_x, c32(0))
                    safe_x = arith.select(
                        x_in, add(mul(gfeat, sx0), add(seq_start, pos_x)), c32(0))
                    val_x = buffer_ops.buffer_load(x_r, safe_x, vec_width=1, dtype=elem_dtype)
                    hi8 = buffer_ops.buffer_load(hi_r, seq_idx, vec_width=1, dtype=T.i8)
                    hi_nz = arith.cmpi(CmpIPredicate.ne, hi8, arith.constant(0, type=T.i8))
                    need_pr = arith.andi(
                        arith.cmpi(CmpIPredicate.slt, pos_x, c32(0)), hi_nz)
                    src = add(slot, seqlen)
                    safe_pr = arith.select(
                        need_pr,
                        add(add(mul(in_coord, scs0), mul(gfeat, scs1)), mul(src, scs2)),
                        c32(0))
                    val_pr = buffer_ops.buffer_load(cs_r, safe_pr, vec_width=1, dtype=elem_dtype)
                    wb_val = arith.select(x_in, val_x, arith.select(need_pr, val_pr, zero_e))
                    cs_wr = add(add(mul(in_coord, scs0), mul(gfeat, scs1)), mul(slot, scs2))
                    buffer_ops.buffer_store(wb_val, cs_r, cs_wr)
                    scf.YieldOp([])
                scf.YieldOp([])

    @flyc.jit
    def launch(
        x_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        bias_ptr: fx.Pointer,
        cs_ptr: fx.Pointer,
        cache_idx_ptr: fx.Pointer,
        has_init_ptr: fx.Pointer,
        qsl_ptr: fx.Pointer,
        batch_ptr: fx.Pointer,
        chunk_off_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        k_ptr: fx.Pointer,
        v_ptr: fx.Pointer,
        dim: Int32,
        kd: Int32,
        vd: Int32,
        sx0: Int32,
        sx1: Int32,
        sw0: Int32,
        sw1: Int32,
        scs0: Int32,
        scs1: Int32,
        scs2: Int32,
        sci: Int32,
        qs0: Int32,
        qs1: Int32,
        ks0: Int32,
        ks1: Int32,
        vs0: Int32,
        vs1: Int32,
        num_programs: Int32,
        grid_y_dim: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = arith.index_cast(T.index, num_programs)
        gy = arith.index_cast(T.index, grid_y_dim)
        conv1d_v2_kernel(
            x_ptr, w_ptr, bias_ptr, cs_ptr, cache_idx_ptr, has_init_ptr,
            qsl_ptr, batch_ptr, chunk_off_ptr, q_ptr, k_ptr, v_ptr,
            dim, kd, vd, sx0, sx1, sw0, sw1, scs0, scs1, scs2, sci,
            qs0, qs1, ks0, ks1, vs0, vs1,
        ).launch(grid=(gx, gy, 1), block=(BT, 1, 1), stream=stream)

    launch._tn = TN
    launch._tm = TM
    return launch


@functools.lru_cache(maxsize=None)
def _get_compiled_v2(width, has_bias, silu, tm, tn, block_threads, dtype_str):
    return build_causal_conv1d_flydsl_v2_module(
        width, has_bias, silu, tm, tn, block_threads, dtype_str
    )


@functools.lru_cache(maxsize=None)
def _get_compiled(width, has_bias, silu, block_m, block_dim, dtype_str):
    return build_causal_conv1d_flydsl_module(
        width, has_bias, silu, block_m, block_dim, dtype_str
    )


@functools.lru_cache(maxsize=None)
def _get_compiled_lds(width, has_bias, silu, tm, tn, block_threads, dtype_str):
    return build_causal_conv1d_flydsl_lds_module(
        width, has_bias, silu, tm, tn, block_threads, dtype_str
    )


def _build_chunk_metadata(query_start_loc: torch.Tensor, block_m: int):
    """Build (num_programs, batch_ptr, token_chunk_offset_ptr) like the Triton wrapper."""
    device = query_start_loc.device
    seqlens = query_start_loc.diff().to("cpu")
    nums = -(-seqlens // block_m)  # ceil
    tot = int(nums.sum().item())
    mlist = np.repeat(np.arange(len(nums)), nums.numpy())
    offsetlist = []
    for num in nums.tolist():
        offsetlist.extend(range(int(num)))
    batch_ptr = torch.from_numpy(np.asarray(mlist, dtype=np.int32)).to(device)
    token_chunk_offset_ptr = torch.tensor(
        offsetlist, dtype=torch.int32, device=device
    )
    return tot, batch_ptr, token_chunk_offset_ptr


def causal_conv1d_flydsl_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    k_dim_size: int,
    v_dim_size: int,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    block_m: int = 64,
    block_dim: int = 256,
    impl: str = "lds",
    **kwargs,
):
    """FlyDSL prefill causal conv1d. Returns (query, key, value).

    Signature mirrors ``_causal_conv1d_fn_tile`` (extra kwargs are ignored) so it
    can be dropped into the same benchmark harness.

    ``impl``: "lds" (default, LDS-staged 2D tile, coalesced load+store) or
    "simple" (1 thread per output element).
    """
    if x.dtype != conv_states.dtype:  # avoid no-op .to() dispatch on the hot path
        x = x.to(conv_states.dtype)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    silu = activation in ("silu", "swish")

    if cache_indices is None:
        cache_indices = torch.arange(
            query_start_loc.numel() - 1, dtype=torch.int32, device=x.device
        )
    if has_initial_state is None:
        has_initial_state = torch.zeros(
            query_start_loc.numel() - 1, dtype=torch.bool, device=x.device
        )

    # chunk schedule (reuse harness metadata when provided)
    if metadata is not None and hasattr(metadata, "nums_dict") and block_m in metadata.nums_dict:
        entry = metadata.nums_dict[block_m]
        tot = int(entry["tot"])
        batch_ptr = entry["batch_ptr"]
        chunk_off_ptr = entry["token_chunk_offset_ptr"]
        if batch_ptr.device != x.device:  # avoid no-op .to() dispatch on the hot path
            batch_ptr = batch_ptr.to(x.device)
            chunk_off_ptr = chunk_off_ptr.to(x.device)
    else:
        tot, batch_ptr, chunk_off_ptr = _build_chunk_metadata(query_start_loc, block_m)

    query = torch.empty([cu_seqlen, k_dim_size], dtype=x.dtype, device=x.device)
    key = torch.empty([cu_seqlen, k_dim_size], dtype=x.dtype, device=x.device)
    value = torch.empty([cu_seqlen, v_dim_size], dtype=x.dtype, device=x.device)

    if tot == 0:
        return query, key, value

    dtype_str = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    if impl == "v2":
        launcher = _get_compiled_v2(
            int(width), bias is not None, bool(silu), int(block_m), 64, 256, dtype_str
        )
        tn = launcher._tn
        grid_y_dim = (dim + tn - 1) // tn
    elif impl == "lds":
        launcher = _get_compiled_lds(
            int(width), bias is not None, bool(silu), int(block_m), 64, 256, dtype_str
        )
        tn = launcher._tn
        grid_y_dim = (dim + tn - 1) // tn
    else:
        launcher = _get_compiled(
            int(width), bias is not None, bool(silu), int(block_m), int(block_dim),
            dtype_str,
        )
        ch_per_blk = launcher._ch_per_blk
        grid_y_dim = (dim + ch_per_blk - 1) // ch_per_blk

    bias_arg = bias if bias is not None else x  # dummy ptr when no bias (HAS_BIAS=False)

    launch_args = (
        x,
        weight,
        bias_arg,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        chunk_off_ptr,
        query,
        key,
        value,
        int(dim),
        int(k_dim_size),
        int(v_dim_size),
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        int(conv_states.stride(0)),
        int(conv_states.stride(1)),
        int(conv_states.stride(2)),
        int(cache_indices.stride(0)),
        int(query.stride(0)),
        int(query.stride(1)),
        int(key.stride(0)),
        int(key.stride(1)),
        int(value.stride(0)),
        int(value.stride(1)),
        int(tot),
        int(grid_y_dim),
        torch.cuda.current_stream(),
    )

    # Fast-dispatch path: the per-call @flyc.jit entry re-runs sig.bind + builds
    # a cache key over all ~30 args (~40us host overhead, which dominates small-T
    # cases where the GPU kernel is only a few us). flyc.compile() bakes the
    # constexpr layout once and returns a CompiledFunction whose __call__ is just
    # the CallState fast path (~5us); only data pointers / scalars / stream vary
    # between calls, which is exactly our situation. Cache it on the launcher.
    compiled = getattr(launcher, "_fast_compiled", None)
    if compiled is None:
        try:
            compiled = flyc.compile(launcher, *launch_args)
            launcher._fast_compiled = compiled
        except Exception:
            launcher._fast_compiled = False  # fall back permanently
            compiled = False
    if compiled is not False:
        compiled(*launch_args)
    else:
        launcher(*launch_args)
    return query, key, value
