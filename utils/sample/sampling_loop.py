import torch

from utils.preprocessing import AA_ALPHABET
from utils.model import generation_diagnostics


class BatchedFastDCAGibbsSampler:
    """
    Batched fast single-site Gibbs sampler for a Potts / DCA model.

    Model:
        P(seq) ∝ exp( sum_i h[i, seq[i]] + sum_{i<j} J[i,j,seq[i],seq[j]] )

    Shapes:
        h:      (L, q)
        J:      (L, L, q, q)
        seq:    (B, L)
        cache:  (B, L, q)

    cache[b, k, a] =
        h[k, a] + sum_{j != k} J[k, j, a, seq[b, j]]
    """

    def __init__(self, h: torch.Tensor, J: torch.Tensor, temp: float = 1.0):
        if h.ndim != 2:
            raise ValueError(f"h must have shape (L, q), got {tuple(h.shape)}")
        if J.ndim != 4:
            raise ValueError(f"J must have shape (L, L, q, q), got {tuple(J.shape)}")

        L, q = h.shape
        if J.shape != (L, L, q, q):
            raise ValueError(f"J must have shape {(L, L, q, q)}, got {tuple(J.shape)}")
        if temp <= 0:
            raise ValueError("temp must be positive")

        self.h = h
        self.J = J
        self.L = L
        self.q = q
        self.temp = temp
        self.device = h.device

    @staticmethod
    def symmetrize_J(J: torch.Tensor) -> torch.Tensor:
        return 0.5 * (J + J.permute(1, 0, 3, 2))

    def init_sequences(self, batch_size: int, init_seq: torch.Tensor | None = None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if init_seq is None:
            return torch.randint(self.q, (batch_size, self.L), device=self.device)

        init_seq = init_seq.to(self.device)

        if init_seq.ndim == 1:
            if init_seq.shape != (self.L,):
                raise ValueError(f"1D init_seq must have shape ({self.L},)")
            init_seq = init_seq.unsqueeze(0).expand(batch_size, -1).clone()

        elif init_seq.ndim == 2:
            if init_seq.shape != (batch_size, self.L):
                raise ValueError(f"2D init_seq must have shape ({batch_size}, {self.L})")
            init_seq = init_seq.clone()

        else:
            raise ValueError("init_seq must be 1D or 2D")

        if init_seq.min() < 0 or init_seq.max() >= self.q:
            raise ValueError("init_seq contains out-of-range states")

        return init_seq

    def build_cache(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, L)
        returns cache: (B, L, q)
        """
        B = seq.shape[0]
        cache = self.h.unsqueeze(0).expand(B, -1, -1).clone()  # (B, L, q)

        # Add contributions from each position j to all sites
        for j in range(self.L):
            aj = seq[:, j]  # (B,)
            # J[:, j, :, :] -> (L, q, q)
            # gather over last dim using aj for each batch element
            contrib = self.J[:, j, :, :].unsqueeze(0).expand(B, -1, -1, -1)  # (B, L, q, q)
            aj_idx = aj.view(B, 1, 1, 1).expand(-1, self.L, self.q, 1)
            contrib = torch.gather(contrib, dim=3, index=aj_idx).squeeze(3)  # (B, L, q)
            cache += contrib

        # remove self-interactions if present
        for i in range(self.L):
            ai = seq[:, i]  # (B,)
            self_term = self.J[i, i, :, :].unsqueeze(0).expand(B, -1, -1)  # (B, q, q)
            ai_idx = ai.view(B, 1, 1).expand(-1, self.q, 1)
            self_term = torch.gather(self_term, dim=2, index=ai_idx).squeeze(2)  # (B, q)
            cache[:, i, :] -= self_term

        return cache

    def gibbs_update_site(self, seq: torch.Tensor, cache: torch.Tensor, i: int):
        """
        Batched update at site i for all chains.

        seq:   (B, L)
        cache: (B, L, q)
        """
        B = seq.shape[0]
        old_ai = seq[:, i].clone()  # (B,)

        logits = cache[:, i, :] / self.temp  # (B, q)
        probs = torch.softmax(logits, dim=1)
        new_ai = torch.multinomial(probs, num_samples=1).squeeze(1)  # (B,)

        changed = (new_ai != old_ai)
        if not torch.any(changed):
            return seq, cache

        seq[:, i] = new_ai

        # Update cache[b, :, :] += J[:, i, :, new_ai[b]] - J[:, i, :, old_ai[b]]
        Ji = self.J[:, i, :, :]  # (L, q, q)
        Ji = Ji.unsqueeze(0).expand(B, -1, -1, -1)  # (B, L, q, q)

        new_idx = new_ai.view(B, 1, 1, 1).expand(-1, self.L, self.q, 1)
        old_idx = old_ai.view(B, 1, 1, 1).expand(-1, self.L, self.q, 1)

        add_term = torch.gather(Ji, dim=3, index=new_idx).squeeze(3)  # (B, L, q)
        sub_term = torch.gather(Ji, dim=3, index=old_idx).squeeze(3)  # (B, L, q)

        delta = add_term - sub_term
        cache += delta

        # correct self row if J[i,i] nonzero
        Jii = self.J[i, i, :, :].unsqueeze(0).expand(B, -1, -1)  # (B, q, q)
        new_self = torch.gather(
            Jii, dim=2,
            index=new_ai.view(B, 1, 1).expand(-1, self.q, 1)
        ).squeeze(2)
        old_self = torch.gather(
            Jii, dim=2,
            index=old_ai.view(B, 1, 1).expand(-1, self.q, 1)
        ).squeeze(2)

        cache[:, i, :] -= (new_self - old_self)

        return seq, cache

    def gibbs_sweep(self, seq: torch.Tensor, cache: torch.Tensor, random_order: bool = True):
        order = torch.randperm(self.L, device=self.device).tolist() if random_order else list(range(self.L))
        for i in order:
            seq, cache = self.gibbs_update_site(seq, cache, i)
        return seq, cache

    def sample(
        self,
        batch_size: int,
        num_samples: int,
        burn_in_frac: float = 0.1,
        sweeps_per_sample: int = 1,
        thinning: int = 1,
        init_seq: torch.Tensor | None = None,
        random_order: bool = True,
        store_samples: bool = True,
        return_last_state: bool = False,
        return_diagnostics: bool = False,
        train_X=None,
        train_weights=None,
        alphabet: str = AA_ALPHABET,
        diagnostics_max_pairs: int = 20000,
        diagnostics_seed: int = 0,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if not (0 <= burn_in_frac < 1):
            raise ValueError("burn_in_frac must be in [0, 1)")
        if sweeps_per_sample <= 0:
            raise ValueError("sweeps_per_sample must be > 0")
        if thinning <= 0:
            raise ValueError("thinning must be > 0")

        seq = self.init_sequences(batch_size, init_seq=init_seq)
        cache = self.build_cache(seq)

        burn_in_steps = max(0, int(round(num_samples * burn_in_frac / max(1e-12, 1 - burn_in_frac))))
        total_outer = burn_in_steps + num_samples

        samples = [] if store_samples else None

        for outer in range(total_outer):
            for _ in range(sweeps_per_sample * thinning):
                seq, cache = self.gibbs_sweep(seq, cache, random_order=random_order)

            if outer >= burn_in_steps and store_samples:
                samples.append(seq.clone())

        out = None
        if store_samples:
            out = torch.stack(samples, dim=0)  # (num_samples, B, L)

        diagnostics = None
        if return_diagnostics:
            samples_for_diagnostics = out if out is not None else seq
            diagnostics = generation_diagnostics(
                samples_for_diagnostics,
                train_X=train_X,
                train_weights=train_weights,
                alphabet=alphabet,
                max_pairs=diagnostics_max_pairs,
                seed=diagnostics_seed,
            )

        if return_last_state and return_diagnostics:
            return out, seq.clone(), cache.clone(), diagnostics
        if return_last_state:
            return out, seq.clone(), cache.clone()
        if return_diagnostics:
            return out, diagnostics
        return out
