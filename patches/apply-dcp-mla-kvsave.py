#!/usr/bin/env python3
"""
Apply DCP with KV savings to SGLang MLA backend.

Changes:
1. forward_batch_info: out_cache_loc // dcp_size + dcp_kv_mask
2. model_runner_kv_cache_mixin: page_size *= dcp_size (KV pool stays same size, virtual pages)
3. MLA decode: DCP indices (with division) + Q all-gather + LSE correction + masked KV write
4. MLA prefill: DCP indices (with division) + Q all-gather + LSE correction + masked KV write
5. Prefill updater: DCP params
"""
import sys


def patch():
    # =========================================================================
    # 1. forward_batch_info: enable out_cache_loc division
    # =========================================================================
    path = '/opt/sglang/python/sglang/srt/model_executor/forward_batch_info.py'
    with open(path, encoding="utf-8") as f:
        src = f.read()
    src = src.replace(
        '        # DCP: no infrastructure changes - handled in attention backend',
        '        # For DCP\n'
        '        if model_runner.dcp_size > 1:\n'
        '            dcp_size = model_runner.dcp_size\n'
        '            dcp_rank = model_runner.dcp_rank\n'
        '            ret.dcp_kv_mask = ret.out_cache_loc % dcp_size == dcp_rank\n'
        '            ret.out_cache_loc = ret.out_cache_loc // dcp_size',
    )
    with open(path, 'w', encoding="utf-8") as f:
        f.write(src)
    print('1. forward_batch_info: out_cache_loc // dcp_size')

    # =========================================================================
    # 2. model_runner_kv_cache_mixin: inflate page_size
    # =========================================================================
    path2 = '/opt/sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py'
    with open(path2, encoding="utf-8") as f:
        src2 = f.read()
    src2 = src2.replace(
        '                    if self.page_size == 1:',
        '                    if self.page_size == 1 and self.dcp_size == 1:',
    )
    src2 = src2.replace(
        '                            self.max_total_num_tokens,\n                            page_size=self.page_size,',
        '                            self.max_total_num_tokens,\n                            page_size=self.page_size * self.dcp_size,',
    )
    with open(path2, 'w', encoding="utf-8") as f:
        f.write(src2)
    print('2. model_runner_kv_cache_mixin: page_size *= dcp_size')

    # =========================================================================
    # 3. MLA backend: DCP for BOTH decode AND prefill
    # =========================================================================
    path3 = '/opt/sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py'
    with open(path3, encoding="utf-8") as f:
        src3 = f.read()

    # --- 3a. Fix imports: use dividing DCP kernel ---
    src3 = src3.replace(
        'create_flashinfer_kv_indices_dcp_nodiv_triton,\n    create_flashinfer_kv_indices_dcp_nodiv_triton,',
        'create_flashinfer_kv_indices_for_dcp_triton,',
    )
    print('3a. Fixed imports')

    # --- 3b. Decode indices: already uses DCP kernel, fix to use dividing version ---
    src3 = src3.replace(
        'create_flashinfer_kv_indices_dcp_nodiv_triton[(bs,)](',
        'create_flashinfer_kv_indices_for_dcp_triton[(bs,)](',
    )
    print('3b. Decode indices: dividing DCP kernel')

    # --- 3c. Decode KV write: use masked Triton kernel ---
    old_decode_kv = (
        "                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                        layer, cache_loc, k, k_rope,\n"
        "                    )"
    )
    new_decode_kv = (
        "                    if self.dcp_size > 1 and forward_batch.dcp_kv_mask is not None:\n"
        "                        _lid = layer.layer_id\n"
        "                        _pool = forward_batch.token_to_kv_pool\n"
        "                        if hasattr(_pool, 'full_kv_pool'):\n"
        "                            _pool = _pool.full_kv_pool\n"
        "                        _buf = _pool.kv_buffer[_lid - _pool.start_layer]\n"
        "                        set_mla_kv_buffer_masked_triton(\n"
        "                            _buf, cache_loc, k, k_rope, forward_batch.dcp_kv_mask,\n"
        "                        )\n"
        "                    else:\n"
        "                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                            layer, cache_loc, k, k_rope,\n"
        "                        )"
    )
    src3 = src3.replace(old_decode_kv, new_decode_kv, 1)
    print('3c. Decode KV write: masked')

    # --- 3d. Prefill (extend) KV write: also masked ---
    old_extend_kv = (
        "                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                        layer, cache_loc, k, k_rope\n"
        "                    )"
    )
    # This is inside the dcp_kv_mask check block from earlier patch
    # Check if already masked
    new_extend_kv = (
        "                    if self.dcp_size > 1 and forward_batch.dcp_kv_mask is not None:\n"
        "                        _lid = layer.layer_id\n"
        "                        _pool = forward_batch.token_to_kv_pool\n"
        "                        if hasattr(_pool, 'full_kv_pool'):\n"
        "                            _pool = _pool.full_kv_pool\n"
        "                        _buf = _pool.kv_buffer[_lid - _pool.start_layer]\n"
        "                        set_mla_kv_buffer_masked_triton(\n"
        "                            _buf, cache_loc, k, k_rope, forward_batch.dcp_kv_mask,\n"
        "                        )\n"
        "                    else:\n"
        "                        forward_batch.token_to_kv_pool.set_mla_kv_buffer(\n"
        "                            layer, cache_loc, k, k_rope\n"
        "                        )"
    )
    if 'set_mla_kv_buffer_masked_triton' in src3 and old_extend_kv in src3:
        # Already has masked decode, just need to handle extend
        src3 = src3.replace(old_extend_kv, new_extend_kv, 1)
        print('3d. Extend KV write: masked')
    elif old_extend_kv in src3:
        # Simple extend KV without any DCP
        src3 = src3.replace(old_extend_kv, new_extend_kv, 1)
        print('3d. Extend KV write: masked (simple path)')
    else:
        print('3d. SKIP: extend KV write already handled')

    # --- 3e. Prefill (extend) paged attention: add DCP Q all-gather + LSE correction ---
    old_extend_paged = (
        "            o = q.new_empty(q.shape)\n"
        "            o = prefill_wrapper_paged.run(\n"
        "                q,\n"
        "                q_rope,\n"
        "                k_buf[:, :, : layer.v_head_dim],\n"
        "                k_buf[:, :, layer.v_head_dim :],\n"
        "                out=o,\n"
        "            )\n"
        "\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)"
    )
    new_extend_paged = (
        "            if self.dcp_size > 1:\n"
        "                # DCP: All-gather Q, partial attention, LSE merge\n"
        "                q_ag = get_dcp_group().all_gather(q.contiguous(), dim=1)\n"
        "                qr_ag = get_dcp_group().all_gather(q_rope.contiguous(), dim=1)\n"
        "                o = q_ag.new_empty(q_ag.shape)\n"
        "                o, s = prefill_wrapper_paged.run(\n"
        "                    q_ag, qr_ag,\n"
        "                    k_buf[:, :, : layer.v_head_dim],\n"
        "                    k_buf[:, :, layer.v_head_dim :],\n"
        "                    out=o, return_lse=True,\n"
        "                )\n"
        "                o = cp_lse_ag_out_rs(o, s, get_dcp_group())\n"
        "            else:\n"
        "                o = q.new_empty(q.shape)\n"
        "                o = prefill_wrapper_paged.run(\n"
        "                    q, q_rope,\n"
        "                    k_buf[:, :, : layer.v_head_dim],\n"
        "                    k_buf[:, :, layer.v_head_dim :],\n"
        "                    out=o,\n"
        "                )\n"
        "\n"
        "        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)"
    )
    if old_extend_paged in src3:
        src3 = src3.replace(old_extend_paged, new_extend_paged)
        print('3e. Extend paged attention: DCP Q all-gather + LSE')
    else:
        print('3e. WARN: extend paged attention pattern not found')

    # --- 3f. Prefill updater: add DCP params ---
    old_prefill_update = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "    ):\n"
        "        if use_ragged:\n"
        "            paged_kernel_lens = prefix_lens"
    )
    new_prefill_update = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        dcp_size: Optional[int] = None,\n"
        "        dcp_rank: Optional[int] = None,\n"
        "    ):\n"
        "        if use_ragged:\n"
        "            paged_kernel_lens = prefix_lens"
    )
    src3 = src3.replace(old_prefill_update, new_prefill_update, 1)

    # Pass dcp params to call_begin_forward in prefill update
    old_prefill_call = (
        "            spec_info,\n"
        "        )\n"
        "\n"
        "    def call_begin_forward(\n"
        "        self,\n"
        "        wrapper_ragged"
    )
    new_prefill_call = (
        "            spec_info,\n"
        "            dcp_size=dcp_size,\n"
        "            dcp_rank=dcp_rank,\n"
        "        )\n"
        "\n"
        "    def call_begin_forward(\n"
        "        self,\n"
        "        wrapper_ragged"
    )
    src3 = src3.replace(old_prefill_call, new_prefill_call, 1)

    # Add dcp params to prefill call_begin_forward signature
    old_prefill_cbf = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "    ):\n"
        "        bs = len(seq_lens)\n"
        "        sm_scale = self.scaling"
    )
    new_prefill_cbf = (
        "        spec_info: Optional[SpecInput] = None,\n"
        "        dcp_size: Optional[int] = None,\n"
        "        dcp_rank: Optional[int] = None,\n"
        "    ):\n"
        "        if dcp_size is None:\n"
        "            dcp_size = 1\n"
        "            dcp_rank = 0\n"
        "        bs = len(seq_lens)\n"
        "        sm_scale = self.scaling"
    )
    src3 = src3.replace(old_prefill_cbf, new_prefill_cbf, 1)
    print('3f. Prefill updater: DCP params')

    # --- 3g. Prefill call_begin_forward: DCP indices ---
    old_prefill_indices = (
        "            create_flashinfer_kv_indices_triton[(bs,)](\n"
        "                self.req_to_token,\n"
        "                req_pool_indices,\n"
        "                paged_kernel_lens,\n"
        "                kv_indptr,\n"
        "                None,\n"
        "                kv_indices,\n"
        "                self.req_to_token.shape[1],\n"
        "            )\n"
        "            qo_indptr"
    )
    new_prefill_indices = (
        "            if dcp_size > 1:\n"
        "                paged_kernel_lens = paged_kernel_lens // dcp_size + (\n"
        "                    dcp_rank < paged_kernel_lens % dcp_size\n"
        "                )\n"
        "                paged_kernel_lens_sum = paged_kernel_lens.sum().item()\n"
        "                kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)\n"
        "                kv_indptr = kv_indptr[: bs + 1]\n"
        "                kv_indices = torch.empty(\n"
        "                    paged_kernel_lens_sum, dtype=torch.int32,\n"
        "                    device=req_pool_indices.device,\n"
        "                )\n"
        "            create_flashinfer_kv_indices_for_dcp_triton[(bs,)](\n"
        "                self.req_to_token,\n"
        "                req_pool_indices,\n"
        "                paged_kernel_lens,\n"
        "                kv_indptr,\n"
        "                None,\n"
        "                kv_indices,\n"
        "                self.req_to_token.shape[1],\n"
        "                dcp_size,\n"
        "                dcp_rank,\n"
        "            )\n"
        "            qo_indptr"
    )
    src3 = src3.replace(old_prefill_indices, new_prefill_indices, 1)
    print('3g. Prefill indices: DCP kernel')

    # --- 3h. Prefill plan: num_local_heads * dcp_size (fix the stale reference) ---
    # The plan already has `self.num_local_heads * dcp_size` from earlier patch
    # but dcp_size might not be defined. Make sure it is.
    # Already handled by adding dcp_size param to call_begin_forward

    # --- 3i. Pass dcp params from init_forward_metadata to prefill updater ---
    # Find where indices_updater_prefill.update is called
    old_prefill_updcall = "self.indices_updater_prefill.update("
    idx = src3.find(old_prefill_updcall)
    if idx > 0:
        end = src3.find(")", idx)
        call = src3[idx:end+1]
        if "dcp_size" not in call:
            new_call = call.rstrip(")") + "\n                dcp_size=self.dcp_size,\n                dcp_rank=self.dcp_rank,\n            )"
            src3 = src3.replace(call, new_call, 1)
            print('3i. Passed dcp to prefill updater call')

    with open(path3, 'w', encoding="utf-8") as f:
        f.write(src3)

    print('\nDCP KV savings patch complete!')


if __name__ == '__main__':
    patch()
