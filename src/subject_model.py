"""Step 4: scoring and intervention machinery for the subject model.

See docs/PLAN_TECHNICAL.md Step 4. Everything the experiments do to the subject model goes through
this module, so there is exactly one place where a silent bug can live -- and one test suite
(src/test_scoring.py) guarding it.

Three conventions fixed here, because getting any of them wrong produces plausible numbers rather
than an exception:

  RIGHT PADDING for scoring. Left padding is only needed for generation. With left padding a plain
  forward pass gets wrong position_ids (HF derives them from arange, not from the mask), which
  corrupts every score in a way that is invisible in aggregate.

  BOS IS EMITTED BY THE TEMPLATE. The Llama-2 chat template writes bos_token itself, so the
  rendered string must be tokenized with add_special_tokens=False or you get two BOS tokens.

  hidden_states[i] IS THE INPUT TO BLOCK i. hidden_states[0] is the embedding output and
  hidden_states[n_layers] is the final normed-input stream, so the tuple has n_layers + 1 entries.
"""

from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# The canonical Llama-2 chat template. The NousResearch mirror ships none and transformers 4.45
# removed the built-in default, so apply_chat_template() raises without this. See WORKLOG entry 5.
LLAMA2_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message != false %}"
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "{% if message['role'] == 'user' %}"
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' ' + content.strip() + ' ' + eos_token }}"
    "{% endif %}{% endfor %}"
)


# --------------------------------------------------------------------------- loading

def load_subject(model_id, dtype=torch.bfloat16, device="cuda"):
    """Load a subject model in full precision. Never quantize this one -- see DESIGN.md 4.1."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.chat_template is None:
        if "llama-2" not in model_id.lower():
            raise ValueError("%s has no chat_template and no known default" % model_id)
        tok.chat_template = LLAMA2_CHAT_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # scoring, not generation

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    model.eval()

    # Fail loudly now rather than silently scoring a malformed prompt for four hours.
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=False, add_generation_prompt=True)
    if "llama-2" in model_id.lower():
        assert "[INST]" in rendered, "Llama-2 template did not render [INST]: %r" % rendered
    return model, tok


def layers_of(model):
    """The decoder blocks. Same path for Llama and Qwen2."""
    return model.model.layers


# --------------------------------------------------------------------------- encoding

def render(tok, messages, add_generation_prompt=True):
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt)


def encode_prompt(tok, messages, add_generation_prompt=True):
    """Chat-formatted prompt -> token ids. add_special_tokens=False: the template emits BOS."""
    text = render(tok, messages, add_generation_prompt)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_answer(tok, answer, leading_space=True):
    """Candidate answer -> token ids, tokenized in isolation and concatenated as IDS, never as a
    string: string concatenation can merge across the boundary and change the tokens being scored.
    """
    text = (" " if leading_space else "") + answer.strip()
    return tok(text, add_special_tokens=False)["input_ids"]


def read_position(tok, prefix_turns):
    """Index of the probe read position: the last token of the prompt truncated after the final
    user message of the prefix, with the generation prompt appended -- the token at which the model
    is about to reply to *this user*.

    Because attention is causal and chat templates are concatenative, the activation at this index
    is identical whether or not the question and answer follow it. That is what makes activation
    extraction cost one forward pass per prefix instead of one per (item, prefix).
    Verified by test_read_position_is_causally_invariant.
    """
    last_user = max(i for i, t in enumerate(prefix_turns) if t["role"] == "user")
    ids = encode_prompt(tok, prefix_turns[:last_user + 1], add_generation_prompt=True)
    return len(ids) - 1


# --------------------------------------------------------------------------- scoring

@torch.no_grad()
def score_answers(model, tok, prompt_ids_list, answer_ids_list, batch_size=8):
    """Mean per-token log-probability of each answer given its prompt.

    Returns a float list, one entry per (prompt, answer) pair. Length-normalized, because lures and
    correct answers differ systematically in length.
    """
    assert len(prompt_ids_list) == len(answer_ids_list)
    device = next(model.parameters()).device
    out = []

    for start in range(0, len(prompt_ids_list), batch_size):
        prompts = prompt_ids_list[start:start + batch_size]
        answers = answer_ids_list[start:start + batch_size]
        seqs = [p + a for p, a in zip(prompts, answers)]
        width = max(len(s) for s in seqs)

        input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):                       # right padding
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        for i, (p, a) in enumerate(zip(prompts, answers)):
            # token j is predicted by position j-1, so answer token k sits at len(p)+k and is
            # predicted from len(p)+k-1
            idx = torch.arange(len(p), len(p) + len(a), device=device)
            tokens = input_ids[i, idx]
            out.append(logprobs[i, idx - 1].gather(-1, tokens.unsqueeze(-1)).mean().item())
    return out


def delta(model, tok, prompt_ids, correct_ids, lure_ids_list, batch_size=8):
    """The primary DV: mean-token-logprob(lure) - mean-token-logprob(correct), averaged over lures.

    Positive = pulled toward the falsehood.
    """
    answers = [correct_ids] + list(lure_ids_list)
    scores = score_answers(model, tok, [prompt_ids] * len(answers), answers, batch_size)
    return sum(scores[1:]) / len(scores[1:]) - scores[0]


# --------------------------------------------------------------------------- activations

@torch.no_grad()
def read_residual(model, tok, ids, position, batch_size=8):
    """Residual stream at one position, every layer. Returns (n_layers + 1, d_model) float32.

    `ids` may be one sequence or a list of them; `position` likewise. Sequences are run one batch at
    a time with right padding, which is safe here because `position` always indexes a real token.
    """
    single = isinstance(ids[0], int)
    seqs = [ids] if single else list(ids)
    positions = [position] if single else list(position)
    device = next(model.parameters()).device
    collected = []

    for start in range(0, len(seqs), batch_size):
        batch, pos = seqs[start:start + batch_size], positions[start:start + batch_size]
        width = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, s in enumerate(batch):
            input_ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1

        hs = model(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                   use_cache=False, output_hidden_states=True).hidden_states
        # hidden_states[i] is the INPUT to block i; hidden_states[0] is the embedding output.
        stacked = torch.stack(hs, dim=0)                   # (n_layers+1, batch, seq, d_model)
        for i, p in enumerate(pos):
            collected.append(stacked[:, i, p, :].float().cpu())
    return collected[0] if single else torch.stack(collected)


# --------------------------------------------------------------------------- interventions

@contextmanager
def hooked(model, layer_idxs, make_hook):
    """Install a forward hook on each named decoder block for the duration of the block.

    `make_hook(layer_idx)` returns fn(module, args, output) -> new output. The residual stream is
    output[0]; the rest of the tuple must be passed through untouched.
    """
    handles = []
    try:
        for idx in layer_idxs:
            handles.append(layers_of(model)[idx].register_forward_hook(make_hook(idx)))
        yield
    finally:
        for h in handles:
            h.remove()


def steering_hook(direction, alpha):
    """C3: add alpha * ||h|| * direction at every position.

    alpha is scaled by the residual norm at that layer -- absolute alpha does not transfer across
    layers. alpha=0 must be an exact no-op; test_alpha_zero_is_identity checks that.
    """
    def make(_idx):
        def hook(_module, _args, output):
            h = output[0]
            d = direction.to(h.device, h.dtype)
            scale = h.norm(dim=-1, keepdim=True).mean()
            return (h + alpha * scale * d,) + tuple(output[1:])
        return hook
    return make


def ablation_hook(direction, mu):
    """C4: mean-ablate the direction -- set its projection to `mu`, the mean under neutral prefixes.

    Mean-ablation, not zero-ablation: zeroing pushes activations off-distribution and yields an
    artifact that looks exactly like the desired result.
    """
    def make(_idx):
        def hook(_module, _args, output):
            h = output[0]
            d = direction.to(h.device, h.dtype)
            proj = (h * d).sum(dim=-1, keepdim=True)
            return (h - (proj - mu) * d,) + tuple(output[1:])
        return hook
    return make


def random_direction(d_model, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm()
