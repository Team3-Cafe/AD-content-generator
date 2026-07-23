from __future__ import annotations


CLIP_MAX_TOKENS = 77
SPECIAL_TOKEN_COUNT = 2
POSITIVE_TOKEN_TARGET = 72


def _token_ids(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    )
    return list(encoded["input_ids"])


def _token_limit(tokenizer) -> int:
    configured = int(
        getattr(tokenizer, "model_max_length", CLIP_MAX_TOKENS)
    )
    if configured <= SPECIAL_TOKEN_COUNT or configured > 100_000:
        return CLIP_MAX_TOKENS
    return min(configured, CLIP_MAX_TOKENS)


def fit_clip_prompt(
    tokenizer,
    prompt: str,
    *,
    label: str,
    required_prefix: str = "",
) -> str:
    """Fit one prompt to CLIP while preserving required leading terms."""
    prompt = str(prompt or "").strip().strip(",")
    required_prefix = str(required_prefix or "").strip().strip(",")

    if required_prefix and prompt:
        required_keys = {
            " ".join(part.lower().split())
            for part in required_prefix.split(",")
            if part.strip()
        }
        prompt = ", ".join(
            part.strip()
            for part in prompt.split(",")
            if (
                part.strip()
                and " ".join(part.lower().split()) not in required_keys
            )
        )

    combined = ", ".join(
        part for part in (required_prefix, prompt) if part
    )

    max_length = _token_limit(tokenizer)
    original_count = len(
        _token_ids(tokenizer, combined, add_special_tokens=True)
    )
    truncated = original_count > max_length
    final_prompt = combined

    if truncated:
        content_budget = max_length - SPECIAL_TOKEN_COUNT
        required_ids = _token_ids(
            tokenizer,
            required_prefix,
            add_special_tokens=False,
        )
        prompt_ids = _token_ids(
            tokenizer,
            prompt,
            add_special_tokens=False,
        )
        separator_ids = (
            _token_ids(tokenizer, ", ", add_special_tokens=False)
            if required_ids and prompt_ids
            else []
        )

        selected_ids = required_ids[:content_budget]
        remaining = content_budget - len(selected_ids)
        if remaining > 0 and separator_ids:
            selected_separator = separator_ids[:remaining]
            selected_ids.extend(selected_separator)
            remaining -= len(selected_separator)
        if remaining > 0:
            selected_ids.extend(prompt_ids[:remaining])

        final_prompt = tokenizer.decode(
            selected_ids,
            skip_special_tokens=True,
        ).strip().strip(",")

    final_count = len(
        _token_ids(tokenizer, final_prompt, add_special_tokens=True)
    )
    while truncated and final_count > max_length and selected_ids:
        selected_ids.pop()
        final_prompt = tokenizer.decode(
            selected_ids,
            skip_special_tokens=True,
        ).strip().strip(",")
        final_count = len(
            _token_ids(tokenizer, final_prompt, add_special_tokens=True)
        )

    status = "truncated" if truncated else "kept"
    print(
        f"[Prompt tokens] {label}: {final_count}/{max_length} "
        f"(original={original_count}, {status})"
    )
    return final_prompt


def fit_clip_prompt_parts(
    tokenizer,
    parts,
    *,
    label: str,
    required_prefix: str = "",
    target_tokens: int = POSITIVE_TOKEN_TARGET,
):
    """Compose whole semantic clauses under a conservative CLIP budget."""
    max_length = _token_limit(tokenizer)
    target = max(
        SPECIAL_TOKEN_COUNT + 1,
        min(int(target_tokens), max_length),
    )
    required_prefix = str(required_prefix or "").strip().strip(",")
    normalized_parts = [
        str(part or "").strip().strip(",")
        for part in parts
        if str(part or "").strip().strip(",")
    ]
    raw_prompt = ", ".join(
        item for item in (required_prefix, *normalized_parts) if item
    )
    original_count = len(
        _token_ids(tokenizer, raw_prompt, add_special_tokens=True)
    )

    selected = []
    dropped = []
    for part in normalized_parts:
        candidate = ", ".join(
            item for item in (required_prefix, *selected, part) if item
        )
        candidate_count = len(
            _token_ids(tokenizer, candidate, add_special_tokens=True)
        )
        if candidate_count <= target:
            selected.append(part)
        else:
            dropped.append(part)

    final_prompt = ", ".join(
        item for item in (required_prefix, *selected) if item
    )
    final_prompt = fit_clip_prompt(
        tokenizer,
        final_prompt,
        label=label,
    )
    final_count = len(
        _token_ids(tokenizer, final_prompt, add_special_tokens=True)
    )
    status = "all clauses kept" if not dropped else (
        f"{len(dropped)} lower-priority clause(s) dropped"
    )
    print(
        f"[Prompt budget] {label}: {final_count}/{target} target "
        f"(raw={original_count}, {status})"
    )
    return final_prompt, {
        "prompt": final_prompt,
        "token_count": final_count,
        "target_tokens": target,
        "model_limit": max_length,
        "raw_token_count": original_count,
        "selected_clauses": selected,
        "dropped_clauses": dropped,
    }
