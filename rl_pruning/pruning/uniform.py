import torch.nn.utils.prune as prune


def uniform_prune(model, target_compression: float):
    # Verify correct bounds
    assert 0.0 <= target_compression < 1.0

    # ---- Parameter split ----
    # TODO: tweak 70% FFN, 30% attention?
    ffn_fraction = 0.7
    attn_fraction = 0.3

    ffn_prune_ratio = min(target_compression / ffn_fraction, 0.9)
    attn_prune_ratio = min(target_compression / attn_fraction, 0.9)

    layers = model.bert.encoder.layer
    for layer in layers:
        # prune neurons
        _prune_ffn(layer, ffn_prune_ratio)
        # prune attention heads
        _prune_attention(layer, attn_prune_ratio)

    # Verify we didn't blow anything up
    assert model.bert.encoder.layer[0].attention.self.num_attention_heads > 0
    assert model.bert.encoder.layer[0].intermediate.dense.weight.shape[0] > 0


def _prune_ffn(layer, ratio):
    if ratio <= 0:
        return

    intermediate = layer.intermediate.dense
    output = layer.output.dense

    hidden_dim = intermediate.weight.shape[1]
    prune_dim = int(hidden_dim * ratio)

    if prune_dim <= 0:
        return

    # Structured pruning on output channels
    prune.ln_structured(
        intermediate,
        name="weight",
        amount=prune_dim,
        n=2,
        dim=0,
    )

    prune.ln_structured(
        output,
        name="weight",
        amount=prune_dim,
        n=2,
        dim=1,
    )

    prune.remove(intermediate, "weight")
    prune.remove(output, "weight")


def _prune_attention(layer, ratio):
    if ratio <= 0:
        return

    attn = layer.attention.self
    num_heads = attn.num_attention_heads

    heads_to_prune = int(num_heads * ratio)
    if heads_to_prune <= 0:
        return

    heads = list(range(num_heads))
    prune_heads = heads[:heads_to_prune]

    layer.attention.prune_heads(prune_heads)
