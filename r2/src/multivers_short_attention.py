"""Preserve the original attention graph while reducing padding of short inputs."""
def set_window(model, tokenized, adaptive=True):
    # With L<=256, the original radius 256 connects every pair of real tokens.
    # Radius L-1 preserves all those connections. Padding tokens remain masked.
    length = int(tokenized['attention_mask'].sum(dim=1).max().item())
    window = 2*max(1, length-1) if adaptive and length <= 256 else 512
    model.encoder.config.attention_window = [window]*len(model.encoder.encoder.layer)
    for layer in model.encoder.encoder.layer:
        layer.attention.self.one_sided_attn_window_size = window//2
    return window
