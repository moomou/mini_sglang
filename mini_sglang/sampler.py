# logits -> repetition/freq penalty -> /temperature -> topk mask -> topp mask -> softmax -> multinomial
import torch 
from dataclasses import dataclass

@dataclass
class SamplingParams:
    # config is PER request
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    rep_penalty: torch.Tensor
    seed: int | None = None
    

class Sampler:
    def __call__(
        self, 
        logits, 
        sample_cfg: SamplingParams, 
        prev_ids: torch.Tensor,
    ):
        if (sample_cfg.rep_penalty != 1.0).any() and prev_ids.numel() > 0:
            rep_penalty = sample_cfg.rep_penalty
            score = logits.gather(-1, prev_ids)
            score = torch.where(score > 0, score / rep_penalty, score * rep_penalty)
            logits.scatter_(-1, prev_ids, score)

        temp = sample_cfg.temperature
        mask_greedy = (temp == 0)

        t_safe = temp.masked_fill(mask_greedy, 1.0)
        # logits:(num_seq,  vocab) / t_safe.unsqueeze:(num_seqs, 1)
        logits = logits / t_safe.unsqueeze(-1)

        # top_k
        k = sample_cfg.top_k
        if k.max() > 0:
            k_max = int(k.max())
            topk_vals, topk_idx = logits.topk(k_max, dim=-1)
            keep = torch.arange(k_max, device=logits.device)[None, :] < k.unsqueeze(-1)
            
            new = torch.full_like(logits, float('-inf'))
            new.scatter_(-1, 
            topk_idx, 
            torch.where(keep, topk_vals, float('-inf')))

            logits = torch.where((k > 0).unsqueeze(-1), new, logits)

        if (sample_cfg.top_p < 1.0).any():
            # top_p
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            sorted_probs = sorted_logits.softmax(dim=-1)
            cum = sorted_probs.cumsum(dim=-1)

            remove = cum > sample_cfg.top_p.unsqueeze(dim=-1)
            remove[..., 1:] = remove[...,:-1].clone()
            remove[..., 0] = False

            remove_unsorted = torch.empty_like(remove)
            remove_unsorted.scatter_(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove_unsorted, float('-inf'))

        if mask_greedy.all():
            next_ids = logits.argmax(-1)
        else:
            probs = logits.float().softmax(dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # greedy override
            next_ids = torch.where(mask_greedy, logits.argmax(dim=-1), next_ids)

        return next_ids