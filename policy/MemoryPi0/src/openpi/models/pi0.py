import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# 获取日志记录器，用于记录Pi0模型的相关信息
logger = logging.getLogger("openpi")


# 时间步嵌入器类，用于将标量时间步嵌入到向量表示中
class TimestepEmbedder(nnx.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, rngs=None):
        # 替换 Sequential 为显式层
        self.mlp_1 = nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs)
        self.mlp_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
        args = jnp.asarray(t[:, None], dtype=jnp.float32) * freqs[None]
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 使用显式层
        return self.mlp_2(nnx.silu(self.mlp_1(t_freq)))


# 交叉变换器块，用于处理查询、键和值的注意力机制
class CrossTransformerBlock(nnx.Module):
    def __init__(self, feature_dim, rngs=None):
        self.q_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.k_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.v_proj = nnx.Linear(feature_dim, feature_dim, rngs=rngs)
        self.attn_norm = nnx.LayerNorm(feature_dim, rngs=rngs)
        # 替换 Sequential 为显式层以避免索引问题
        self.ffn_1 = nnx.Linear(feature_dim, feature_dim * 4, rngs=rngs)
        self.ffn_2 = nnx.Linear(feature_dim * 4, feature_dim, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(feature_dim, rngs=rngs)

    def __call__(self, query, k, v, mask=None):
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        # Reshape for attention: (batch, seq, 1, D)
        q = q[:, :, None, :]
        k = k[:, :, None, :]
        v = v[:, :, None, :]
        # Attention
        head_dim = q.shape[-1]
        attn_logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(head_dim)
        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        attn_out = attn_out[:, :, 0, :]  # (batch, seq_q, D)
        x = self.attn_norm(query + attn_out)
        # 使用显式层
        ffn_out = self.ffn_2(nnx.gelu(self.ffn_1(x)))
        return self.ffn_norm(x + ffn_out)


# 简化的瓶颈SE模块，用于特征压缩（不假设2D结构）
class BottleneckSE(nnx.Module):
    def __init__(self, C_in, C_mid, C_out, rngs=None):
        self.reduce = nnx.Linear(C_in, C_mid, rngs=rngs)
        self.act = nnx.relu
        self.excite_conv1 = nnx.Linear(C_mid, C_mid // 16, rngs=rngs)
        self.excite_conv2 = nnx.Linear(C_mid // 16, C_mid, rngs=rngs)
        self.expand = nnx.Linear(C_mid, C_out, rngs=rngs)

    def __call__(self, x):
        # x: (batch, seq_len, C_in)
        b, n, c = x.shape

        # 降维
        z = self.act(self.reduce(x))  # (b, n, C_mid)

        # 全局平均池化沿着序列维度
        pooled = jnp.mean(z, axis=1, keepdims=True)  # (b, 1, C_mid)

        # 计算注意力权重
        excite_out = self.act(self.excite_conv1(pooled))
        w = nnx.sigmoid(self.excite_conv2(excite_out))  # (b, 1, C_mid)

        # 应用注意力并扩展
        final = self.expand(z * w)  # (b, n, C_out)
        return final


# 门控融合模块，用于融合两个特征向量
class GateFusion(nnx.Module):
    def __init__(self, dim, rngs=None):
        self.proj = nnx.Linear(dim * 2, dim, rngs=rngs)

    def __call__(self, x1, x2):
        scale = nnx.sigmoid(self.proj(jnp.concatenate([x1, x2], axis=-1)))
        return scale * x1 + (1 - scale) * x2


# 认知记忆库类，用于管理认知特征的记忆
class CogMemBank(nnx.Module):
    def __init__(self, dataloader_type, group_size, token_size, mem_length=16, retrieval_layers=2, use_timestep_pe=True, fusion_type='gate', consolidate_type='tome', update_fused=False, max_episodes=1000, rngs=None):
        assert dataloader_type in ('stream', 'group')
        assert fusion_type in ('gate', 'add')
        assert consolidate_type in ('fifo', 'tome')

        self.dataloader_type = dataloader_type
        self.group_size = group_size
        self.token_size = token_size
        self.mem_length = mem_length
        self.retrieval_layers = retrieval_layers
        self.use_timestep_pe = use_timestep_pe
        self.fusion_type = fusion_type
        self.consolidate_type = consolidate_type
        self.update_fused = update_fused
        self.max_episodes = max_episodes
        self.N = 1  # For cognitive memory, N=1

        # 检索块 - 使用字典而不是列表来避免索引键问题
        self.retrieval_blocks = {}
        for i in range(self.retrieval_layers):
            self.retrieval_blocks[f'block_{i}'] = CrossTransformerBlock(self.token_size, rngs=rngs)

        if self.fusion_type == 'gate':
            self.gate_fusion_blocks = GateFusion(self.token_size, rngs=rngs)

        if self.use_timestep_pe:
            self.timestep_encoder = TimestepEmbedder(self.token_size, self.token_size // 4, rngs=rngs)
        else:
            self.timestep_encoder = None

        # 记忆库状态变量，使用数组结构以支持JAX
        self.bank = nnx.Variable({
            'feats': jnp.zeros((self.max_episodes, self.mem_length, self.token_size)),  # N=1 for cog
            'timesteps': jnp.zeros((self.max_episodes, self.mem_length)),
            'counts': jnp.zeros(self.max_episodes, dtype=jnp.int32)
        })
        self.eid_stream = nnx.Variable(None)

    def reset(self):
        # 由于移除了状态变量，这个方法现在为空
        pass

    def _compute_retrieved(self, current_bank, eid, working_mem):
        count = current_bank['counts'][eid]
        # Slice to max_length
        hist_feats = current_bank['feats'][eid]  # (mem_length, token_size) for Cog, (mem_length, N, token_size) for Per
        hist_timesteps = current_bank['timesteps'][eid]  # (mem_length,)
        episode_mem = hist_feats.reshape(-1, self.token_size)[None]  # (1, mem_length*self.N, D)

        if self.use_timestep_pe:
            pe = self.timestep_encoder(hist_timesteps)[None]  # (1, mem_length, D)
            pe = jnp.repeat(pe, self.N, axis=1)  # (1, mem_length*self.N, D)
        else:
            pe = jnp.zeros_like(episode_mem)

        # Create mask for valid tokens
        seq_len_kv = self.mem_length * self.N
        valid_mask = jnp.arange(seq_len_kv) < count * self.N
        mask = valid_mask[None, None, None, :]  # (1, 1, 1, seq_len_kv)

        query = working_mem
        for block_key in self.retrieval_blocks:
            block = self.retrieval_blocks[block_key]
            query = block(query, episode_mem + pe, episode_mem, mask=mask)
        return query

    def _consolidate(self, feats, eid, feat_to_store):
        old = feats[eid]
        new = jnp.concatenate([old[1:], feat_to_store[None]], axis=0)
        return feats.at[eid].set(new)

    def _consolidate_timesteps(self, timesteps, eid, new_t):
        old = timesteps[eid]
        new = jnp.concatenate([old[1:], jnp.array([new_t])], axis=0)
        return timesteps.at[eid].set(new)

    def process_batch(self, tokens, episode_ids, timesteps):
        # 实现记忆机制：检索和融合历史记忆
        B, N, D = tokens.shape
        outputs = []
        current_bank = self.bank.value.copy()

        for i in range(B):
            eid = episode_ids[i]
            working_mem = tokens[i:i+1]  # (1, N, D)
            count = current_bank['counts'][eid].astype(jnp.int32)  # Ensure count is int32 for indexing

            retrieved = jax.lax.cond(
                count > 0,
                lambda: self._compute_retrieved(current_bank, eid, working_mem).astype(working_mem.dtype),
                lambda: working_mem
            )

            # 融合
            if self.fusion_type == 'add':
                fused = (working_mem + retrieved) * 0.5
            elif self.fusion_type == 'gate':
                fused = self.gate_fusion_blocks(working_mem, retrieved)

            outputs.append(fused)

            # 巩固记忆
            feat_to_store = jax.lax.cond(self.update_fused, lambda: fused.squeeze(0), lambda: tokens[i])
            if self.N == 1:
                feat_to_store = feat_to_store[0]
            new_count = jax.lax.cond(
                count < self.mem_length,
                lambda: count + 1,
                lambda: count
            )
            new_feats = jax.lax.cond(
                count < self.mem_length,
                lambda: current_bank['feats'].at[eid, count].set(feat_to_store),
                lambda: self._consolidate(current_bank['feats'], eid, feat_to_store)
            )
            new_timesteps = jax.lax.cond(
                self.use_timestep_pe,
                lambda: jax.lax.cond(
                    count < self.mem_length,
                    lambda: current_bank['timesteps'].at[eid, count].set(timesteps[i]),
                    lambda: self._consolidate_timesteps(current_bank['timesteps'], eid, timesteps[i])
                ),
                lambda: current_bank['timesteps']
            )
            current_bank['feats'] = new_feats
            current_bank['timesteps'] = new_timesteps
            current_bank['counts'] = current_bank['counts'].at[eid].set(new_count)

        # 更新记忆库
        self.bank.value = current_bank
        return jnp.concatenate(outputs, axis=0)


# 感知记忆库类，继承自CogMemBank，用于管理感知特征的记忆
class PerMemBank(CogMemBank):
    def __init__(self, dataloader_type, group_size, token_size, mem_length=16, retrieval_layers=2, use_timestep_pe=True, fusion_type='gate', consolidate_type='tome', update_fused=False, max_episodes=1000, rngs=None):
        # 调用父类构造函数，但重写bank
        super().__init__(dataloader_type, group_size, token_size, mem_length, retrieval_layers, use_timestep_pe, fusion_type, consolidate_type, update_fused, max_episodes, rngs)
        # 重写bank以适应感知特征的形状 (max_episodes, mem_length, 768, token_size)
        self.N = 768  # 固定序列长度
        self.bank = nnx.Variable({
            'feats': jnp.zeros((self.max_episodes, self.mem_length, self.N, self.token_size)),
            'timesteps': jnp.zeros((self.max_episodes, self.mem_length)),
            'counts': jnp.zeros(self.max_episodes, dtype=jnp.int32)
        })


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    # 将mask_ar广播到input_mask的形状
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    # 计算累积和，用于确定注意力范围
    cumsum = jnp.cumsum(mask_ar, axis=1)
    # 创建注意力掩码：当前token可以关注累积mask_ar小于等于自身的token
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # 创建有效掩码：只关注有效的输入token
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    # 返回注意力掩码和有效掩码的逻辑与
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    # 检查embedding_dim是否为偶数
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    # 创建分数数组，用于计算周期
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    # 计算周期，从min_period到max_period
    period = min_period * (max_period / min_period)**fraction
    # 计算正弦余弦输入
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    # 返回正弦和余弦的拼接
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    # 数据类型，默认为bfloat16
    dtype: str = "bfloat16"
    # PaliGemma变体，默认为gemma_2b
    paligemma_variant: _gemma.Variant = "gemma_2b"
    # 动作专家变体，默认为gemma_300m
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 设置模型特定的默认值
    # 动作维度，默认为32
    action_dim: int = 32
    # 动作视野，默认为50
    action_horizon: int = 50
    # 最大token长度，默认为48
    max_token_len: int = 48

    # 记忆相关参数
    dataloader_type: str = "group"
    group_size: int = 16
    per_token_size: int = 256
    mem_length: int = 16
    retrieval_layers: int = 2
    use_timestep_pe: bool = True
    fusion_type: str = 'gate'
    consolidate_type: str = 'tome'
    update_fused: bool = False
    max_episodes: int = 1000

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # 返回模型类型为PI0
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        # 创建Pi0模型实例
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 定义图像规格
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # 禁用类型检查，创建观察规格
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        # 定义动作规格
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    @override
    def create_dummy_inputs(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        # 创建假输入用于测试
        image_shape = (batch_size, *_model.IMAGE_RESOLUTION, 3)
        images = {
            "base_0_rgb": jnp.zeros(image_shape, jnp.float32),
            "left_wrist_0_rgb": jnp.zeros(image_shape, jnp.float32),
            "right_wrist_0_rgb": jnp.zeros(image_shape, jnp.float32),
        }
        image_masks = {
            "base_0_rgb": jnp.ones((batch_size,), jnp.bool_),
            "left_wrist_0_rgb": jnp.ones((batch_size,), jnp.bool_),
            "right_wrist_0_rgb": jnp.ones((batch_size,), jnp.bool_),
        }
        state = jnp.zeros((batch_size, self.action_dim), jnp.float32)
        tokenized_prompt = jnp.zeros((batch_size, self.max_token_len), jnp.int32)
        tokenized_prompt_mask = jnp.ones((batch_size, self.max_token_len), bool)

        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim), jnp.float32)

        return observation, actions

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        # 初始化过滤器列表
        filters = []
        has_lora = False
        # 定义Gemma参数过滤器
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        # 定义动作专家参数过滤器
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        # 检查PaliGemma变体是否包含lora
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter, )
            if "lora" not in self.action_expert_variant:
                # 如果只冻结Gemma参数，排除动作专家参数
                filters.append(nnx.Not(action_expert_params_filter), )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter, )
            has_lora = True

        if has_lora:
            # 如果使用任何lora，排除所有lora参数
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        # 调用父类构造函数
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        # 获取PaliGemma和动作专家配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # 存储配置以供后续使用
        self.action_expert_config = action_expert_config
        # TODO: rewrite gemma in NNX. For now, use bridge.
        # 创建LLM模块，使用桥接
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        # 延迟初始化LLM
        llm.lazy_init(rngs=rngs, method="init")
        # 创建图像模块
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        # 延迟初始化图像模块
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        # 将LLM和图像模块组合成PaliGemma
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 状态投影层
        self.state_proj = nnx.Linear(config.action_dim, self.action_expert_config.width, rngs=rngs)
        # 动作输入投影层
        self.action_in_proj = nnx.Linear(config.action_dim, self.action_expert_config.width, rngs=rngs)
        # 动作时间MLP输入层
        self.action_time_mlp_in = nnx.Linear(2 * self.action_expert_config.width, self.action_expert_config.width, rngs=rngs)
        # 动作时间MLP输出层
        self.action_time_mlp_out = nnx.Linear(self.action_expert_config.width, self.action_expert_config.width, rngs=rngs)
        # 动作输出投影层
        self.action_out_proj = nnx.Linear(self.action_expert_config.width, config.action_dim, rngs=rngs)

        # 计算视觉维度
        self.vision_dim = paligemma_config.width  # 图像编码器输出维度
        self.cog_token_size = 1024  # 认知token大小

        # 感知压缩器
        self.per_compr = BottleneckSE(
            C_in=self.vision_dim,
            C_mid=config.per_token_size * 2,
            C_out=config.per_token_size,
            rngs=rngs,
        )

        # 认知压缩器
        self.extract_cog_tokens = BottleneckSE(
            C_in=1024,
            C_mid=512,
            C_out=self.cog_token_size,
            rngs=rngs,
        )
        self.cog_proj = nnx.Linear(2048, self.cog_token_size, rngs=rngs)

        # 认知记忆库
        self.cog_mem_bank = CogMemBank(
            dataloader_type=config.dataloader_type,
            group_size=config.group_size,
            token_size=self.cog_token_size,
            mem_length=config.mem_length,
            retrieval_layers=config.retrieval_layers,
            use_timestep_pe=config.use_timestep_pe,
            fusion_type=config.fusion_type,
            consolidate_type=config.consolidate_type,
            update_fused=config.update_fused,
            max_episodes=config.max_episodes,
            rngs=rngs,
        )

        # 感知记忆库
        self.per_mem_bank = PerMemBank(
            dataloader_type=config.dataloader_type,
            group_size=config.group_size,
            token_size=config.per_token_size,
            mem_length=config.mem_length,
            retrieval_layers=config.retrieval_layers,
            use_timestep_pe=config.use_timestep_pe,
            fusion_type=config.fusion_type,
            consolidate_type=config.consolidate_type,
            update_fused=config.update_fused,
            max_episodes=config.max_episodes,
            rngs=rngs,
        )

        self.cur_timestep = 0

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # 初始化输入掩码、自回归掩码和token列表
        input_mask = []
        ar_mask = []
        tokens = []
        # 嵌入图像
        for name in obs.images:
            # 获取图像token
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            # 扩展图像掩码到token长度
            input_mask.append(einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            ))
            # 图像token之间可以相互关注
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言输入（即tokenized prompt）
        if obs.tokenized_prompt is not None:
            # 嵌入tokenized输入
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言输入之间完全关注
            ar_mask += [False] * tokenized_inputs.shape[1]
        # 拼接所有token
        tokens = jnp.concatenate(tokens, axis=1)
        # 拼接所有输入掩码
        input_mask = jnp.concatenate(input_mask, axis=1)
        # 转换为数组的自回归掩码
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        # 初始化输入掩码、自回归掩码和token列表
        input_mask = []
        ar_mask = []
        tokens = []
        # 添加单个状态token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # 图像/语言输入不能关注状态或动作
        ar_mask += [True]

        # 使用正弦余弦位置编码嵌入时间步，敏感度范围在[0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # 使用MLP混合时间步 + 动作信息
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/状态输入不能关注动作token
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        # 拼接所有token
        tokens = jnp.concatenate(tokens, axis=1)
        # 拼接所有输入掩码
        input_mask = jnp.concatenate(input_mask, axis=1)
        # 转换为数组的自回归掩码
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     *,
                     train: bool = False,
                     episode_ids: jnp.ndarray = None,
                     timesteps: jnp.ndarray = None) -> at.Float[at.Array, "*b ah"]:
        # 分割随机数生成器
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        # 预处理观察
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 获取批次形状
        batch_shape = actions.shape[:-2]
        # 生成噪声
        noise = jax.random.normal(noise_rng, actions.shape)
        # 生成时间步
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        # 计算x_t和u_t
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 提取视觉特征
        vision_feats = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            vision_feats.append(image_tokens)
        vision_feats = jnp.concatenate(vision_feats, axis=1)
        per_tokens = self.per_compr(vision_feats)

        # 提取认知特征
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        input_mask = jnp.concatenate([prefix_mask, jnp.ones((prefix_mask.shape[0], 1), dtype=jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.array([True])], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, jnp.zeros((prefix_tokens.shape[0], 1, self.action_expert_config.width))], mask=attn_mask, positions=positions)
        cog_tokens = self.extract_cog_tokens(self.cog_proj(prefix_out[:, -1:, :]))  # 最后一个token作为认知特征

        # 记忆库处理
        if episode_ids is None:
            episode_ids = jnp.zeros(actions.shape[0], dtype=jnp.int32)
        if timesteps is None:
            timesteps = jnp.arange(actions.shape[0], dtype=jnp.float32)

        cog_tokens = self.cog_mem_bank.process_batch(cog_tokens, episode_ids, timesteps)
        per_tokens = self.per_mem_bank.process_batch(per_tokens, episode_ids, timesteps)

        # 一次大的前向传递：前缀 + 后缀
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        # 拼接输入掩码和自回归掩码
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        # 创建注意力掩码
        attn_mask = make_attn_mask(input_mask, ar_mask)
        # 计算位置
        positions = jnp.cumsum(input_mask, axis=1) - 1
        # LLM前向传递
        (prefix_out, suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix_tokens],
                                                         mask=attn_mask,
                                                         positions=positions)
        # 计算v_t
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

        # 返回损失：v_t和u_t的平方差的均值
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        episode_ids: jnp.ndarray = None,
        timesteps: jnp.ndarray = None,
    ) -> _model.Actions:
        # 预处理观察
        observation = _model.preprocess_observation(None, observation, train=False)
        # 注意：我们使用扩散文献中更常见的约定，其中t=1是噪声，t=0是目标分布。是的，这与pi0论文相反，我很抱歉。
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # 生成噪声
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 提取视觉特征
        vision_feats = []
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            vision_feats.append(image_tokens)
        vision_feats = jnp.concatenate(vision_feats, axis=1)
        per_tokens = self.per_compr(vision_feats)

        # 提取认知特征
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        input_mask = jnp.concatenate([prefix_mask, jnp.ones((prefix_mask.shape[0], 1), dtype=jnp.bool_)], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, jnp.array([True])], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, jnp.zeros((prefix_tokens.shape[0], 1, self.action_expert_config.width))], mask=attn_mask, positions=positions)
        cog_tokens = self.extract_cog_tokens(self.cog_proj(prefix_out[:, -1:, :]))  # 最后一个token作为认知特征

        # 记忆库处理
        if episode_ids is None:
            episode_ids = jnp.zeros(batch_size, dtype=jnp.int32)
        if timesteps is None:
            timesteps = jnp.arange(batch_size, dtype=jnp.float32)

        cog_tokens = self.cog_mem_bank.process_batch(cog_tokens, episode_ids, timesteps)
        per_tokens = self.per_mem_bank.process_batch(per_tokens, episode_ids, timesteps)

        # 首先用前缀的前向传递填充KV缓存
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            # 执行一步采样
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t,
                                                                           jnp.broadcast_to(time, batch_size))
            # `suffix_attn_mask` 是形状 (b, suffix_len, suffix_len) 的掩码，表示后缀token如何相互关注
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` 是形状 (b, suffix_len, prefix_len) 的掩码，表示后缀token如何关注前缀token
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` 是形状 (b, suffix_len, prefix_len + suffix_len) 的掩码，表示后缀token（生成查询）如何关注完整的前缀 + 后缀序列（生成键和值）
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` 是形状 (b, suffix_len) 的数组，表示后缀token的位置
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
                                                             mask=full_attn_mask,
                                                             positions=positions,
                                                             kv_cache=kv_cache)
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            # 条件函数，检查是否继续循环
            x_t, time = carry
            # 对浮点误差鲁棒
            return time >= -dt / 2

        # 使用while循环进行采样
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
