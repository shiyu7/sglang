**摘要**
- 目标：在保持最小改动的前提下，将 Context Parallel（CP）从仅 NSA 扩展到 FlashInfer MLA 路径，并为 DeepSeek V3 与 Kimi-K2.5 模型接入通用 CP 流程；通过公共 CP utils 解耦 NSA 与 CP 判断，在不同 backend 中实现对应行为。
- 范围：不重构现有 NSA-CP 路径；新增/复用通用 CP 元数据与 gating，补齐 FlashInfer MLA 的 CP 前向分支，并在 DeepSeek/Kimi 的 MLA 路径上启用通用 CP。
- 已确认决策：
  - FlashInfer 路径需要同时覆盖 `in-seq-split` 与 `round-robin-split`。
  - CLI/内部逻辑采用“通用 helper/flag 优先，旧 NSA 参数保留兼容”的策略。

**现状分析**
- 通用 CP 基础：`cp_utils.py` 已提供 `ContextParallelMetadata`、`can_cp_split()`、`cp_attn_forward_extend()` 等通用能力，且 `ForwardBatch` 同时支持 `attn_cp_metadata` 与 `nsa_cp_metadata` 两套元数据。
  - 参考：[cp_utils.py](file:///Users/bytedance/Repos/sglang/python/sglang/srt/layers/utils/cp_utils.py)
  - 参考：ForwardBatch 字段定义 [forward_batch_info.py:L419-L421](file:///Users/bytedance/Repos/sglang/python/sglang/srt/model_executor/forward_batch_info.py#L419-L421)
- NSA 专用路径：`nsa/utils.py` 与 `communicator_nsa_cp.py` 持有 NSA-CP 专用逻辑与通讯器，`DeepseekV2` 在 NSA 分支中使用它们；CP gating 广泛使用 `is_nsa_enable_prefill_cp()`。
  - 参考：[nsa/utils.py](file:///Users/bytedance/Repos/sglang/python/sglang/srt/layers/attention/nsa/utils.py)
  - 参考：[communicator_nsa_cp.py](file:///Users/bytedance/Repos/sglang/python/sglang/srt/layers/communicator_nsa_cp.py)
- 已有通用模型接入样例：`qwen3_moe.py` 在 forward 中基于通用 CP utils 设置 `forward_batch.attn_cp_metadata`，由后端在前向时消费。
  - 参考：[qwen3_moe.py:L982-L989](file:///Users/bytedance/Repos/sglang/python/sglang/srt/models/qwen3_moe.py#L982-L989)
- Attention Backend 状态：
  - FlashAttention 后端中已有对 `attn_cp_metadata` 的处理（prefill 阶段分块+聚合 KV）。参考：[flashattention_backend.py:L674-L697](file:///Users/bytedance/Repos/sglang/python/sglang/srt/layers/attention/flashattention_backend.py#L674-L697)
  - FlashInfer MLA 系列（`flashinfer_mla_backend.py`、`flashmla_backend.py`、`trtllm_mla_backend.py`）目前未接入 `attn_cp_metadata` 分支。
  - 后端注册：MLA 模型使用 `flashinfer` 时，会被路由到 `FlashInferMLAAttnBackend`。[attention_registry.py:L41-L46](file:///Users/bytedance/Repos/sglang/python/sglang/srt/layers/attention/attention_registry.py#L41-L46)
- DeepSeek/Kimi 接入点：
  - DeepSeek V3 系列在 `deepseek_v2.py` 中。NSA 分支已实现 NSA-CP；非 NSA 的 MLA 分支尚未接入通用 CP。[deepseek_v2.py:L1510-L1616](file:///Users/bytedance/Repos/sglang/python/sglang/srt/models/deepseek_v2.py#L1510-L1616)
  - Kimi-K2.5 架构同样在 `deepseek_v2` 路径下由 `model_arch` 分支控制，需在 MLA 分支复用通用 CP。
- ServerArgs：既有 `enable_nsa_prefill_context_parallel`（legacy）与通用 `enable_prefill_context_parallel` 并存，但通用 gating 使用较少；CLI 已暴露 `--attn-cp-size` 与通用 CP 标志位。[server_args.py:L680-L683](file:///Users/bytedance/Repos/sglang/python/sglang/srt/server_args.py#L680-L683)

**改动方案（最小化）**
1. 通用 CP 开关与模式兜底（通用优先，兼容旧 NSA 参数）
   - 扩展 `cp_utils.py` 中的 gating：
     - `is_prefill_context_parallel_enabled()` 返回 `enable_prefill_context_parallel or enable_nsa_prefill_context_parallel`（保持向后兼容）。
     - 新增 `get_prefill_cp_mode()`：优先返回 `prefill_cp_mode`，否则回退到 `nsa_prefill_cp_mode`。
     - 新增 `cp_use_prefill(forward_batch)`：判断 forward 为 CP 场景且 `forward_batch.attn_cp_metadata` 存在。
   - 在 `server_args.py` 中补充轻量 helper（若当前不存在）：`prefill_cp_enabled()` 与 `get_prefill_cp_mode()`，供新代码统一读取；旧字段保留不删。
   - 对基础设施层仅做必要的 helper 替换，不做大面积重构。

2. 通用 CP 元数据同时覆盖 `in-seq-split` 与 `round-robin-split`
   - 将 `nsa/utils.py` 中与 split 相关、但不依赖 NSA indexer 的逻辑下沉/复用到 `cp_utils.py`：
     - `is_prefill_cp_in_seq_split()` 保持；
     - 新增 `is_prefill_cp_round_robin_split()`；
     - 新增通用 `can_prefill_cp_round_robin_split(forward_batch)`；
     - 新增通用 `cp_round_robin_split_data()`、`cp_round_robin_split_q_seqs()`；
     - 扩展 `prepare_context_parallel_metadata()`，使其在 `round-robin-split` 下返回适配 metadata，或新增 `prepare_round_robin_context_parallel_metadata()` 供通用 CP 使用。
   - 原则：能复用 NSA 现有 split 代码就复用，但接口收敛到 `cp_utils.py`；NSA 文件只保留 NSA index/cache/indexer 专属内容。

3. FlashInfer MLA 接入通用 CP（改 MLA 基类 backend，覆盖 `flashinfer` / `flashmla` / `trtllm_mla`）
   - 在 `FlashInferMLAAttnBackend` 的 prefill 前向路径中：
     - 若 `forward_batch.forward_mode.is_context_parallel_extend()` 且 `forward_batch.attn_cp_metadata is not None` 且 `attn_cp_size > 1`，则：
       - `in-seq-split`：通过 `cp_attn_forward_extend()` 将 `q` 分为前/后两段，构造闭包 `attn_fn(q_chunk, cu_seqlens_q, cache_seqlens, max_seqlen_q)`，内部调用现有 MLA ragged/paged wrapper 做一次前向；将结果拼接为完整输出。
       - `round-robin-split`：在进入 wrapper 前对 `q/k/v/positions/seq lens` 使用通用 CP utils 做 token 级 round-robin 重排，前向后再按通用 CP utils 做 all-gather + rerange，行为对齐 NSA 现有 round-robin 语义。
       - 若保存 KV：调用 `cp_allgather_and_save_kv_cache()` 聚集并写回全量 KV 到 KV 池，保证后续层与 MoE 的全局可见性一致。
     - 非 CP 路径维持原逻辑不变。
   - 由于 `flashmla_backend.py` 与 `trtllm_mla_backend.py` 继承 `FlashInferMLAAttnBackend`，优先将 CP 实现在基类中，避免分别改三套 backend。

4. 模型侧（DeepSeek/Kimi）在 MLA 分支启用通用 CP 元数据
   - 在 `DeepseekV2ForCausalLM.forward()`（非 NSA 分支）中，在进入 `self.model(...)` 之前：
     - 若 `is_prefill_context_parallel_enabled()`，调用 `can_cp_split(len(input_ids), attn_cp_size, forward_batch)`，成功则设置 `forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(...)`。
     - 若 `get_prefill_cp_mode()` 为 `round-robin-split`，则改用通用 round-robin metadata/seq-split 准备函数。
   - 初始化时，当满足“通用 CP 启用且 MLA 分支”时，将 `self.layer_communicator` 置为 `NSACPLayerCommunicator` 以沿用“在 round-robin 模式下 MLP 前聚合至 FULL”的成熟路径（行为与 NSA-CP 一致）。
     - 为最小改动，复用 `NSACPLayerCommunicator`；后续若需要完全去 NSA 命名，可在未来重命名。当前仅在 gating 处做兼容。

5. NSA utils 的轻量兼容（不侵入语义）
   - 修改 `nsa/utils.py` 中的 `nsa_use_prefill_cp(forward_batch)`：当通用 CP 启用且 `forward_batch.attn_cp_metadata` 存在时，也返回 True。这样 `NSACPLayerCommunicator` 可被用于通用 CP 场景，避免在短期内重写一套 CP 通讯器。
   - 从 `nsa/utils.py` 迁移/复用所有“与 split 规则相关但不依赖 NSA indexer”的公共逻辑到 `cp_utils.py`，`nsa/utils.py` 仅保留 NSA 特有逻辑与向后兼容包装。

6. Kimi-K2.5
   - Kimi-K2.5 归属于 `DeepseekV2` 分支（`model_arch` 判定）。上述在 DeepSeek MLA 分支的接入即可覆盖 Kimi-K2.5 MLA 路径；无需在 `kimi_k25.py` 另加重复逻辑。

**变更清单（按文件）**
- `python/sglang/srt/layers/utils/cp_utils.py`
  - 扩展/新增 gating 方法：`is_prefill_context_parallel_enabled()` 兼容 NSA；新增 `get_prefill_cp_mode()`、`cp_use_prefill(forward_batch)`。
  - 吸收/复用 round-robin 相关公共 split 与 metadata 逻辑。
  - 尽量保持现有 `ContextParallelMetadata`、`cp_attn_forward_extend()` 接口稳定。
- `python/sglang/srt/server_args.py`
  - 新增/补齐通用 helper：`prefill_cp_enabled()`、`get_prefill_cp_mode()`。
  - 在 DeepSeek/Kimi 的 CP 约束校验处，统一基于通用 helper，同时兼容 legacy `enable_nsa_prefill_context_parallel`/`nsa_prefill_cp_mode`。
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`
  - 在 prefill 路径集成通用 CP 分支：检测 `attn_cp_metadata`，按 `in-seq` 或 `round-robin` 走不同的 CP 前向/重排；在需要时聚合并保存 KV。
  - 不改 decode 路径与已存在的 ragged/paged 选择逻辑。
- `python/sglang/srt/models/deepseek_v2.py`
  - 在非 NSA 的 MLA 分支 forward 前设置 `forward_batch.attn_cp_metadata`（参照 `qwen3_moe.py`）。
  - 在层初始化时，当通用 CP 启用时选择 `NSACPLayerCommunicator` 作为 `layer_communicator`，以复用 ALL-GATHER 到 FULL 的 MoE 前通讯行为。
- `python/sglang/srt/layers/attention/nsa/utils.py`
  - 放宽 `nsa_use_prefill_cp(forward_batch)` 的判定：`forward_batch.attn_cp_metadata` 存在时也返回 True（仅作为短期复用 NSACP 通讯器的 gating）。

**不在本次范围**
- 不重构 `communicator.py` 中大量 `is_nsa_enable_prefill_cp()` 判定点；不触碰 DP/Graph runner 的 require_gathered_buffer 判定。
- 不在 FlashInfer 非 MLA 路径（`flashinfer_backend.py`）新增 CP 逻辑（已有 FA/FA3 的 CP 分支可用）。
- 不改变 NSA indexer、NSA cache、DSA 专属路径的语义。

**验证与验收**
1. 单机（Hopper/Blackwell）功能验证：
   - DeepSeek-V3（非 NSA 架构）+ FlashInfer MLA + 通用 CP in-seq-split：
     - 启动：`--tp 8 --attn-cp-size 4 --enable-prefill-context-parallel --prefill-cp-mode in-seq-split --attention-backend flashinfer`
     - 观察：prefill 阶段触发 CP（日志/断点），MoE 前置 FULL 聚合，结果一致。
   - DeepSeek-V3（非 NSA 架构）+ FlashInfer MLA + 通用 CP round-robin-split：
     - 启动：`--tp 8 --attn-cp-size 8 --enable-prefill-context-parallel --prefill-cp-mode round-robin-split --attention-backend flashinfer`
     - 观察：token 级重排/all-gather/rerange 正确，MoE 路由一致。
   - Kimi-K2.5 + FlashInfer MLA + 通用 CP in-seq-split：同上组合参数验证。
   - Kimi-K2.5 + FlashInfer MLA + 通用 CP round-robin-split：同上组合参数验证。
2. 回归：
   - NSA-CP 路径与现有 UT/CI 保持通过（例如 [test_deepseek_v32_cp_single_node.py](file:///Users/bytedance/Repos/sglang/test/registered/cp/test_deepseek_v32_cp_single_node.py)）。
   - 非 CP 场景无性能/行为回退（对比 baseline 日志/吞吐）。
3. 边界：
   - 仅 prefill 支持 CP；batch=1 限制维持；多机暂不启用（与既有约束一致）。

**风险与缓解**
- 风险：在 `NSACPLayerCommunicator` 复用下，非 NSA 通用 CP 的 gather/fusion 行为需正确 gating。
  - 缓解：gating 仅基于 `forward_batch.attn_cp_metadata` 与 CP enable；decode 路径不受影响；新增逻辑仅在 prefill 触发。
- 风险：FlashInfer MLA 的 wrapper 接口与 FA3/FA2 略有差异。
  - 缓解：采用闭包 `attn_fn` + `cp_attn_forward_extend()`，仅传入 MLA 已有元数据（`qo_indptr/kv_indptr` 等），不改变 wrapper 初始化流程。

**实施步骤**
1. 扩展 `cp_utils.py` 的 gating 与工具函数（兼容 NSA 开关）。
2. 在 `server_args.py` 补 helper，并统一 DeepSeek/Kimi CP 约束校验到通用 helper。
3. 在 `flashinfer_mla_backend.py` 的 prefill 路径接入 `attn_cp_metadata` 分支，分别支持 `in-seq` 与 `round-robin`，并复用 `cp_attn_forward_extend()` / `cp_allgather_and_save_kv_cache()`。
5. 在 `deepseek_v2.py` 的非 NSA MLA 分支：设置 `forward_batch.attn_cp_metadata`；初始化时在通用 CP 启用时选择 `NSACPLayerCommunicator`。
6. 在 `nsa/utils.py` 放宽 `nsa_use_prefill_cp(forward_batch)` 以覆盖通用 CP（短期过渡）。
7. 本地验证与小集成测试；如有必要，补充最小 e2e 回归用例（DeepSeek V3 / Kimi-K2.5 + FlashInfer MLA + `in-seq` / `round-robin`）。

**接受标准**
- DeepSeek V3（非 NSA）与 Kimi-K2.5 在 FlashInfer MLA + 通用 CP（`in-seq-split`、`round-robin-split`）下能正确前向与 MoE 路由，UT/小规模评测通过。
- 现有 NSA-CP 测试用例不回退；非 CP 路径性能不劣化。
