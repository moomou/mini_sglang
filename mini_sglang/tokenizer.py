import torch
from transformers import AutoTokenizer

def load_tokenizer(model_dir):
    return AutoTokenizer.from_pretrained(model_dir, use_fast=True)

K = 4
class IncrementalDetokenizer:
    def __init__(self, tokenizer, prompt_ids: torch.Tensor):
        self.tokenizer = tokenizer
        self.tokens = prompt_ids.tolist()

        self.emitted_len = len(
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        )

    def push_naive(self, new_token_id: int) -> str:
        prev = self.tokenizer.decode(
            self.tokens, skip_special_tokens=False)

        self.tokens.append(new_token_id)
        new = self.tokenizer.decode(
            self.tokens, skip_special_tokens=False)

        diff = len(new) - len(prev)
        to_emit = new[-diff:]
        return to_emit

    def push(self, new_token_id: int) -> str:
        self.tokens.append(new_token_id)

        start = max(0, len(self.tokens) - K)
        token_window = self.tokens[start: ]

        decoded_window = self.tokenizer.decode(
            token_window, skip_special_tokens=False)

        if decoded_window.endswith("\ufffd"):
            # mid utf-8
            return ""

        prefix_text = self.tokenizer.decode(
            self.tokens[start:-1],
            skip_special_tokens=False,
        )

        if "\ufffd" in prefix_text:
            # previous step had partial chars that this token may have resolved.
       # The window-diff is unreliable; fall back to the global cursor for this emit.
            full = self.tokenizer.decode(self.tokens, skip_special_tokens=False)
            new_text = full[self.emitted_len:]
        else:
            new_text = decoded_window[len(prefix_text):]

        self.emitted_len += len(new_text)
        return new_text

    def flush(self) -> str:
        final = self.tokenizer.decode(
            self.tokens, skip_special_tokens=False)
        if len(final) == self.emitted_len:
            return ""

        start = self.emitted_len
        self.emitted_len = len(final)
        return final[start:]

        
