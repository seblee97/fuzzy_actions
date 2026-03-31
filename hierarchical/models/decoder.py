"""Autoregressive GRU decoder for primitive action sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceDecoder(nn.Module):
    """GRU-based autoregressive decoder: (z, s1) → primitive action sequence.

    During training, uses teacher forcing: the ground-truth action at step
    ``t`` is fed as input at step ``t+1``.  During inference, use
    :meth:`generate` for greedy autoregressive decoding.

    Architecture::

        context = [z ; enc_s1]  →  Linear  →  GRU initial hidden state
        input_t = embed(a_{t-1})  (a_{-1} = BOS token)
        GRU output  →  Linear  →  logits_t  (B, n_actions)

    At each step, ``z`` is also concatenated to the action embedding so the
    decoder has global context throughout the sequence.

    Parameters
    ----------
    z_dim:
        Dimensionality of the latent action z.
    n_actions:
        Number of primitive actions (output vocabulary size).
    hidden_dim:
        GRU hidden state dimensionality.
    n_layers:
        Number of stacked GRU layers.
    embed_dim:
        Dimensionality of the encoder output.  When provided, enc(s1) is
        used as additional context for the initial hidden state.
    input_dim:
        Dimensionality of the action token embeddings fed into the GRU.
        Defaults to ``hidden_dim``.
    """

    def __init__(
        self,
        z_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        embed_dim: int | None = None,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embed_dim = embed_dim

        # BOS token index = n_actions (one beyond the valid action range)
        self._BOS = n_actions
        tok_input_dim = input_dim or hidden_dim

        # Context → initial hidden state
        context_dim = z_dim + (embed_dim if embed_dim is not None else 0)
        self.h0_proj = nn.Linear(context_dim, hidden_dim * n_layers)

        # Action token embedding (includes BOS)
        self.action_embed = nn.Embedding(n_actions + 1, tok_input_dim)

        # GRU input: action_embed concatenated with z for global context
        self.gru = nn.GRU(
            tok_input_dim + z_dim, hidden_dim, n_layers, batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, n_actions)

    # ------------------------------------------------------------------

    def _init_hidden(
        self, z: torch.Tensor, enc_s1: torch.Tensor | None
    ) -> torch.Tensor:
        """Build the (n_layers, B, hidden_dim) initial hidden state."""
        ctx = torch.cat([z, enc_s1], dim=-1) if enc_s1 is not None else z
        h = self.h0_proj(ctx)  # (B, hidden_dim * n_layers)
        B = z.shape[0]
        return h.view(B, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        enc_s1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        z:       (B, z_dim) latent action.
        actions: (B, T) int64 ground-truth action sequence.
        enc_s1:  (B, embed_dim) optional start-state context.

        Returns
        -------
        logits: (B, T, n_actions)
            Logit at position ``t`` is the prediction for ``actions[:, t]``.
        """
        B, T = actions.shape
        h0 = self._init_hidden(z, enc_s1)

        # Input sequence: [BOS, a_0, a_1, ..., a_{T-2}]
        bos = actions.new_full((B, 1), self._BOS)
        inputs = torch.cat([bos, actions[:, :-1]], dim=1)  # (B, T)

        embeds = self.action_embed(inputs)  # (B, T, tok_input_dim)
        z_rep = z.unsqueeze(1).expand(-1, T, -1)  # (B, T, z_dim)
        gru_inputs = torch.cat([embeds, z_rep], dim=-1)  # (B, T, tok_input_dim + z_dim)

        out, _ = self.gru(gru_inputs, h0)  # (B, T, hidden_dim)
        return self.output_proj(out)  # (B, T, n_actions)

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        max_len: int,
        enc_s1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding.

        Parameters
        ----------
        z:       (B, z_dim)
        max_len: Maximum sequence length to generate.
        enc_s1:  (B, embed_dim) optional start-state context.

        Returns
        -------
        actions: (B, max_len) int64 predicted action ids.
        """
        B = z.shape[0]
        h = self._init_hidden(z, enc_s1)
        token = z.new_full((B,), self._BOS, dtype=torch.long)

        actions_out = []
        for _ in range(max_len):
            embed = self.action_embed(token)  # (B, tok_input_dim)
            gru_in = torch.cat([embed, z], dim=-1).unsqueeze(1)  # (B, 1, *)
            out, h = self.gru(gru_in, h)
            logit = self.output_proj(out.squeeze(1))  # (B, n_actions)
            token = logit.argmax(dim=-1)
            actions_out.append(token)

        return torch.stack(actions_out, dim=1)  # (B, max_len)
