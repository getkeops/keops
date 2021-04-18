import torch
from torch.nn.functional import linear
from pykeops.torch import LazyTensor
from torch.nn.functional import softmax, dropout

# Plug-in replacements for the PyTorch v1.8
# torch.nn.functional.multi_head_attention_forward
# and torch.nn.MultiheadAttention.

# ======================================================================================
#                                   Implementation
# ======================================================================================


from typing import Callable, List, Optional, Tuple

Tensor = torch.Tensor


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    lazy: bool = True,
    landmarks: int = None,
    landmarks_selection: str = "random",
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - lazy: bool, defaults to True.
          If True, use KeOps LazyTensors instead of the original dense PyTorch tensors.
        - landmarks: int, defaults to None.
          If None, use the original bruteforce self-attention.
          Otherwise, use the Nystroem method with a set number of landmarks (aka. inducing points).
          See "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention",
          Xiong et al. (2021) for reference.
        - landmarks_selection: str, defaults to "random".
          Method to select the Nyström landmarks:
          either "random" or "local average".

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # Straightforward copy-paste from the PyTorch repo: ================================

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    if isinstance(embed_dim, torch.Tensor):
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
        head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (
            key is value or torch.equal(key, value)
        ):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        raise NotImplementedError()  # KeOps: we do not yet support attention masks.

        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
            attn_mask.dtype
        )
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
            )
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            )
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # Real computations start here. ====================================================
    # Following the PyTorch conventions:
    # - L denotes the length of the target sequence (usually, "N" with KeOps)
    # - S denotes the length of the source sequence (usually, "M" with KeOps)
    # - N denotes the batch size (usually, "B" with KeOps)
    # - E denotes the full embedding dimension
    # - H or num_heads denotes the number of attention heads
    # - D = E / H denotes the effective "head" dimension of each query/key vector.
    # - C denotes the number of Nystroem landmarks (= inducing points).
    #
    # At this point:
    # q is (N*H, L, D)
    # k is (N*H, S, D)
    # v is (N*H, S, D)

    # Shall we use the Nystroem method? ================================================
    nystroem = landmarks is not None

    if nystroem:
        # 1. Select the landmarks for the queries and keys -----------------------------
        if landmarks_selection == "random":
            # Draw "landmarks" (= "C") indices without replacements
            # in [0, tgt_len-1] and [0, src_len-1].
            id_q = torch.ones(tgt_len).multinomial(landmarks, replacement=False)
            id_k = torch.ones(src_len).multinomial(landmarks, replacement=False)
            q_landmarks = q[:, id_q, :]  # (N*H, C, D)
            k_landmarks = k[:, id_k, :]  # (N*H, C, D)

        elif landmarks_selection == "local average":
            # This method assumes that the key and query points
            # can be grouped in meaningful contiguous blocks along the
            # "sequence" dimension.
            # This may be sensible for language processing (where tokens
            # are usually words in a sentence), but is unreasonable
            # for e.g. point cloud processing (with token=points being ordered
            # at random).
            q_landmarks = q.reshape(
                bsz * num_heads, landmarks, -1, head_dim
            )  # (N*H, C, L/C, D)
            q_landmarks = q_landmarks.mean(2)  # (N*H, C, D)
            k_landmarks = k.reshape(
                bsz * num_heads, landmarks, -1, head_dim
            )  # (N*H, C, S/C, D)
            k_landmarks = k_landmarks.mean(2)  # (N*H, C, D)

        else:
            raise ValueError(
                f"landmarks_selection method is '{landmarks_selection}', "
                + "but should be either 'random' or 'local average'."
            )

        # 2. Compute the Nystroem correction matrix ------------------------------------
        # 2.a. Compute the Attention matrix for the landmarks:
        A = q_landmarks @ k_landmarks.transpose(
            1, 2
        )  # (N*H, C, D) @ (N*H, D, C) = (N*H, C, C)
        assert list(A.size()) == [bsz * num_heads, landmarks, landmarks]
        A = softmax(A, dim=-1)  # (N*H, C, C) with rows that sum up to 1.

        # 2.b. Compute the pseudo-inverse of A_s using the iterative method of
        #      "A New Iterative Method for Finding Approximate Inverses of Complex Matrices",
        #      Razavi et al. (2014).
        I = torch.eye(landmarks, device=A.device)  # (C, C)

        A_norms = torch.max(torch.sum(A, dim=-2), dim=-1).values  # (N*H,)
        Z = 1 / A_norms[:, None, None] * A.transpose(1, 2)  # (N*H, C, C)

        for _ in range(6):  # The Nyström-former paper advises to use 6 iterations
            AZ = A @ Z  # (N*H, C, C)
            Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))

        # --> Z is now a (N*H, C, C) batch of (C, C) matrices that
        #     are close to the pseudoinverss of A.

        # 3. Nyström approximate matrix-matrix product ---------------------------------
        if attn_mask is not None:
            raise NotImplementedError(
                "KeOps attention layers do not support attention masks."
            )
        if key_padding_mask is not None:
            raise NotImplementedError(
                "KeOps attention layers do not support key padding masks."
            )

        if not lazy:  # Original PyTorch code.
            # See https://github.com/mlpen/Nystromformer/blob/main/code/attention_nystrom.py
            # for the reference implementation.

            # (N*H, L, D) @ (N*H, D, C) = (N*H, L, C), with rows that sum up to 1:
            A_q_kl = softmax(q @ k_landmarks.transpose(1, 2), dim=-1)

            A_ql_kl_inv = Z  # see above: (N*H, C, C).

            # (N*H, C, D) @ (N*H, D, S) = (N*H, C, S), with rows that sum up to 1:
            A_ql_k = softmax(q_landmarks @ k.transpose(1, 2), dim=-1)

            # (N*H, L, C) @ (N*H, C, C) @ (N*H, C, S) @ (N*H, S, D) = (N*H, L, D)
            attn_output = (A_q_kl @ A_ql_kl_inv) @ (A_ql_k @ v)

        else:  # KeOps implementation

            # 3.a: "vv = A_ql_k @ v"  --------------------------------------------------
            bsz_2 = 64  # = "B"

            if bsz * num_heads * landmarks < 1024 * 8:
                # We reshape the problem to use more treads than the number of CUDA cores

                ql_i = LazyTensor(
                    q_landmarks.reshape(
                        bsz * num_heads, 1, landmarks, 1, head_dim
                    ).contiguous()
                )  # (N*H, 1, C, 1, D)
                k_j = LazyTensor(
                    k.reshape(
                        bsz * num_heads, bsz_2, 1, src_len // bsz_2, head_dim
                    ).contiguous()
                )  # (N*H, B, 1, S/B, D)
                v_j = LazyTensor(
                    v.reshape(
                        bsz * num_heads, bsz_2, 1, src_len // bsz_2, head_dim
                    ).contiguous()
                )  # (N*H, B, 1, S/B, D)

                A_qlk_ij = ql_i | k_j  # (N*H, B, C, S/B)
                assert list(A_qlk_ij.shape) == [
                    bsz * num_heads,
                    bsz_2,
                    landmarks,
                    src_len // bsz_2,
                ]

                # (N*H, B, C, S//B) @ (N*H, B, S//B, D) = (N*H, B, C, D)
                vv = A_qlk_ij.reduction(
                    "Max_SumShiftExpWeight", other=LazyTensor(1).concat(v_j), dim=3
                )
                # vv has shape (N*H, B, C, D+2): [max_i, sum exp(att_ij - m_i), sum exp(att_ij - m_i) v_j]
                max_vv = (
                    vv[:, :, :, :1].max(dim=1, keepdim=True).values
                )  # (N*H, 1, C, 1)
                max_vv = vv[:, :, :, :1] - max_vv  # (N*H, B, C, 1)
                weighted = vv[:, :, :, 1:] * max_vv.exp()  # (N*H, B, C, 1+D)
                vv = weighted[:, :, :, 1:].sum(dim=1) / weighted[:, :, :, :1].sum(
                    dim=1
                )  # (N*H, C, D)

            else:
                ql_i = LazyTensor(
                    q_landmarks[:, :, None, :].contiguous()
                )  # (N*H, C, 1, D)
                k_j = LazyTensor(k[:, None, :, :].contiguous())  # (N*H, 1, S, D)
                v_j = LazyTensor(v[:, None, :, :].contiguous())  # (N*H, 1, S, D)

                A_qlk_ij = ql_i | k_j  # (N*H, C, S)
                assert list(A_qlk_ij.shape) == [bsz * num_heads, landmarks, src_len]

                # (N*H, C, S) @ (N*H, S, D) = (N*H, C, D)
                vv = A_qlk_ij.sumsoftmaxweight(v_j, dim=2)

            # 3.b: "vvv = A_ql_kl_inv @ vv" --------------------------------------------
            vvv = Z @ vv  # (N*H, C, C) @ (N*H, C, D) = (N*H, C, D)

            # 3.d: "attn_output = A_q_kl @ vvv" ----------------------------------------
            if bsz * num_heads * landmarks < 1024 * 8:
                q_i = LazyTensor(
                    q.reshape(
                        bsz * num_heads, bsz_2, tgt_len // bsz_2, 1, head_dim
                    ).contiguous()
                )  # (N*H, B, L/B, 1, D)
                kl_j = LazyTensor(
                    k_landmarks[:, None, None, :, :].contiguous()
                )  # (N*H, 1, 1, C, D)
                vvv_j = LazyTensor(
                    vvv[:, None, None, :, :].contiguous()
                )  # (N*H, 1, 1, C, D)

                A_qkl_ij = q_i | kl_j  # (N*H, B, L/B, C)
                assert list(A_qkl_ij.shape) == [
                    bsz * num_heads,
                    bsz_2,
                    tgt_len // bsz_2,
                    landmarks,
                ]

                # (N*H, B, L/B, C) @ (N*H, 1, C, D) = (N*H, B, L/B, D)
                attn_output = A_qkl_ij.sumsoftmaxweight(vvv_j, dim=3)
                attn_output = attn_output.view(
                    bsz * num_heads, tgt_len, head_dim
                )  # (N*H, L, D)

            else:
                q_i = LazyTensor(q[:, :, None, :].contiguous())  # (N*H, L, 1, D)
                kl_j = LazyTensor(
                    k_landmarks[:, None, :, :].contiguous()
                )  # (N*H, 1, C, D)
                vvv_j = LazyTensor(vvv[:, None, :, :].contiguous())  # (N*H, 1, C, D)

                A_qkl_ij = q_i | kl_j  # (N*H, L, C)
                assert list(A_qkl_ij.shape) == [bsz * num_heads, tgt_len, landmarks]

                # (N*H, L, C) @ (N*H, C, D) = (N*H, L, D)
                attn_output = A_qkl_ij.sumsoftmaxweight(vvv_j, dim=2)

    else:  # No Nystroem: vanilla attention ============================================
        # Compute the attention scores -----------------------------------------------------
        if not lazy:  # Original PyTorch code
            attn_output_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_output_weights.size()) == [
                bsz * num_heads,
                tgt_len,
                src_len,
            ]

        else:
            q_i = LazyTensor(q[:, :, None, :].contiguous())  # (N*H, L, 1, D)
            k_j = LazyTensor(k[:, None, :, :].contiguous())  # (N*H, 1, S, D)
            v_j = LazyTensor(v[:, None, :, :].contiguous())  # (N*H, 1, S, D)

            attn_output_weights = q_i | k_j  # (N*H, L, S)
            assert list(attn_output_weights.shape) == [
                bsz * num_heads,
                tgt_len,
                src_len,
            ]

        # Mask on the attention matrix: not supported yet ----------------------------------
        if attn_mask is not None:
            raise NotImplementedError(
                "KeOps attention layers do not support attention masks."
            )

            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        # Mask on the keys: not supported yet ----------------------------------------------
        if key_padding_mask is not None:
            raise NotImplementedError(
                "KeOps attention layers do not support key padding masks."
            )

            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        # Perform the normalized matrix-vector product -------------------------------------
        if not lazy:  # Original PyTorch code
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            attn_output_weights = dropout(
                attn_output_weights, p=dropout_p, training=training
            )

            attn_output = torch.bmm(attn_output_weights, v)

        else:
            if dropout_p > 0.0:
                raise NotImplementedError(
                    "KeOps attentions layers do not support dropout."
                )

            # (N*H, L, S) @ (N*H, S, D) = (N*H, L, D)
            attn_output = attn_output_weights.sumsoftmaxweight(v_j, dim=2)

    assert list(attn_output.shape) == [bsz * num_heads, tgt_len, head_dim]

    # Final post-processing ============================================================
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        raise NotImplementedError(
            "KeOps attentions layers cannot return attention weights."
        )
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads

    else:
        return attn_output, None


# ======================================================================================
#                              Differentiable Module
# ======================================================================================

from torch.nn import Parameter


class MultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(
        self,
        *args,
        lazy: bool = True,
        landmarks: int = None,
        landmarks_selection: str = "random",
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__(*args, **kwargs)
        self.lazy = lazy
        self.landmarks = landmarks
        self.landmarks_selection = landmarks_selection

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                lazy=self.lazy,
                landmarks=self.landmarks,
                landmarks_selection=self.landmarks_selection,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                lazy=self.lazy,
                landmarks=self.landmarks,
                landmarks_selection=self.landmarks_selection,
            )
