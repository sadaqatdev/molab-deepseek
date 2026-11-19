# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers>=4.51",
#     "accelerate",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="DeepSeek on molab")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import gc
    import os
    import shutil
    import threading

    import accelerate  # noqa: F401  (needed for device_map; imported so molab installs it)
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
        gc,
        os,
        shutil,
        threading,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    # DeepSeek on molab

    Runs an open-weights DeepSeek R1 reasoning model on this notebook's GPU.
    Nothing leaves the notebook: no API key, no external inference service.

    1. Attach the GPU: **notebook specs button** in the molab header.
    2. Pick a model below and press **Load model**.
    3. Chat at the bottom. The model's reasoning streams first, then its answer.
    """)
    return


@app.cell
def _(mo, os, shutil, torch):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            mo.md(
                "**No GPU attached.** Click the notebook specs button in the "
                "molab header, switch the GPU on, then run this notebook again."
            ),
            kind="danger",
        ),
    )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    _cache_dir = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    os.makedirs(_cache_dir, exist_ok=True)
    disk_free_gb = shutil.disk_usage(_cache_dir).free / 1e9

    mo.callout(
        mo.md(
            f"**GPU:** {gpu_name} · **VRAM:** {vram_gb:.0f} GB · "
            f"**Free disk for weights:** {disk_free_gb:.0f} GB"
        ),
        kind="success",
    )
    return disk_free_gb, vram_gb


@app.cell
def _(mo):
    # label -> (Hugging Face repo id, weight size in GB)
    MODELS = {
        "R1 Distill Qwen 1.5B · 4 GB · fastest": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            3.6,
        ),
        "R1 Distill Qwen 7B · 15 GB": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            15.2,
        ),
        "R1-0528 Qwen3 8B · 16 GB · recommended": (
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            16.4,
        ),
        "R1 Distill Qwen 14B · 30 GB": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            29.5,
        ),
        "R1 Distill Qwen 32B · 66 GB · strongest, slow first load": (
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            65.5,
        ),
    }

    model_form = mo.ui.dropdown(
        options=MODELS,
        value="R1-0528 Qwen3 8B · 16 GB · recommended",
        label="Model",
        full_width=True,
    ).form(submit_button_label="Load model")
    model_form
    return (model_form,)


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    disk_free_gb,
    gc,
    mo,
    model_form,
    torch,
    vram_gb,
):
    mo.stop(
        model_form.value is None,
        mo.md("*Pick a model above and press **Load model**.*"),
    )

    model_id, _size_gb = model_form.value

    mo.stop(
        _size_gb > vram_gb * 0.9,
        mo.callout(
            mo.md(
                f"`{model_id}` needs about {_size_gb:.0f} GB of VRAM but this GPU "
                f"has {vram_gb:.0f} GB. Pick a smaller model."
            ),
            kind="danger",
        ),
    )

    # Release the previously loaded model before loading another one.
    gc.collect()
    torch.cuda.empty_cache()

    with mo.status.spinner(
        title=f"Loading {model_id}",
        subtitle=(
            f"First load downloads about {_size_gb:.0f} GB "
            f"({disk_free_gb:.0f} GB disk free)."
        ),
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # device_map puts the weights straight on the GPU instead of staging
        # them through system RAM. `torch_dtype` is the spelling every
        # transformers release accepts (newer ones also take `dtype`).
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", torch_dtype=torch.bfloat16
        )
        model.eval()

    mo.callout(
        mo.md(
            f"**Loaded** `{model_id}` · "
            f"{torch.cuda.memory_allocated() / 1e9:.1f} GB of VRAM in use"
        ),
        kind="success",
    )
    return model, model_id, tokenizer


@app.cell
def _(
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    mo,
    model,
    model_id,
    threading,
    tokenizer,
    torch,
):
    _OPEN, _CLOSE = "<think>", "</think>"


    def _message_text(message):
        """Answer text of a chat message, without any reasoning parts."""
        _parts = message.parts or []
        if _parts:
            _texts = []
            for _part in _parts:
                if isinstance(_part, dict):
                    _kind, _text = _part.get("type"), _part.get("text")
                else:
                    _kind = getattr(_part, "type", None)
                    _text = getattr(_part, "text", None)
                if _kind == "text" and _text:
                    _texts.append(_text)
            return "".join(_texts)
        return message.content if isinstance(message.content, str) else ""


    def _held_back(text, tag):
        """Length of the longest suffix of `text` that could start `tag`."""
        for _n in range(min(len(tag) - 1, len(text)), 0, -1):
            if tag.startswith(text[-_n:]):
                return _n
        return 0


    class _StopOnEvent(StoppingCriteria):
        def __init__(self, event):
            self.event = event

        def __deepcopy__(self, memo):
            return self  # keep sharing the event if transformers copies us

        def __call__(self, input_ids, scores, **kwargs):
            return torch.full(
                (input_ids.shape[0],),
                self.event.is_set(),
                dtype=torch.bool,
                device=input_ids.device,
            )


    def _deepseek_chat(messages, config):
        # Earlier reasoning is never fed back: only user turns and final answers.
        _conversation = [
            {"role": _m.role, "content": _text}
            for _m in messages
            if _m.role in ("user", "assistant")
            and (_text := _message_text(_m).strip())
        ]
        _prompt = tokenizer.apply_chat_template(
            _conversation, tokenize=False, add_generation_prompt=True
        )
        _inputs = tokenizer(
            _prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        _temperature = config.temperature if config.temperature is not None else 0.6
        _gen_kwargs = dict(
            **_inputs,
            max_new_tokens=int(config.max_tokens or 4096),
            pad_token_id=tokenizer.eos_token_id,
        )
        if _temperature > 0:
            _gen_kwargs.update(
                do_sample=True,
                temperature=_temperature,
                top_p=config.top_p if config.top_p is not None else 0.95,
            )
        else:
            _gen_kwargs.update(do_sample=False)

        _stop = threading.Event()
        _streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        _errors = []

        def _generate():
            try:
                with torch.inference_mode():
                    model.generate(
                        streamer=_streamer,
                        stopping_criteria=StoppingCriteriaList([_StopOnEvent(_stop)]),
                        **_gen_kwargs,
                    )
            except BaseException as _exc:  # surface GPU errors in the chat
                _errors.append(_exc)
                _streamer.end()

        _worker = threading.Thread(target=_generate, daemon=True)
        _worker.start()

        # Some DeepSeek templates open the reasoning block in the prompt itself;
        # others let the model emit <think>. Handle both, and models that skip it.
        _state = "reasoning" if _prompt.rstrip().endswith(_OPEN) else "start"
        _pending = ""
        _answered = False
        _rid = "reasoning-0"

        try:
            if _state == "reasoning":
                yield {"type": "reasoning-start", "id": _rid}

            for _piece in _streamer:
                _pending += _piece

                if _state == "start":
                    _lead = _pending.lstrip()
                    if _lead.startswith(_OPEN):
                        _pending = _lead[len(_OPEN) :]
                        _state = "reasoning"
                        yield {"type": "reasoning-start", "id": _rid}
                    elif _lead and not _OPEN.startswith(_lead):
                        _state = "answer"
                    else:
                        continue  # not enough text yet to tell

                if _state == "reasoning":
                    _idx = _pending.find(_CLOSE)
                    if _idx == -1:
                        _keep = _held_back(_pending, _CLOSE)
                        _emit = _pending[: len(_pending) - _keep]
                        _pending = _pending[len(_pending) - _keep :]
                        if _emit:
                            yield {
                                "type": "reasoning-delta",
                                "id": _rid,
                                "delta": _emit,
                            }
                        continue
                    if _pending[:_idx]:
                        yield {
                            "type": "reasoning-delta",
                            "id": _rid,
                            "delta": _pending[:_idx],
                        }
                    yield {"type": "reasoning-end", "id": _rid}
                    _pending = _pending[_idx + len(_CLOSE) :]
                    _state = "answer"

                if _state == "answer":
                    if not _answered:
                        # drop the blank lines between </think> and the answer
                        _pending = _pending.lstrip()
                    if _pending:
                        _answered = True
                        yield _pending
                        _pending = ""

            if _errors:
                raise _errors[0]

            if _state == "reasoning":
                if _pending:
                    yield {"type": "reasoning-delta", "id": _rid, "delta": _pending}
                yield {"type": "reasoning-end", "id": _rid}
            elif _state == "start" and _pending.strip():
                _answered = True
                yield _pending

            if not _answered:
                yield (
                    "*The model used all of its tokens on reasoning. "
                    "Raise **max tokens** in the chat settings and ask again.*"
                )
        finally:
            # Runs on normal completion and when the user presses Stop.
            _stop.set()


    chat = mo.ui.chat(
        _deepseek_chat,
        prompts=[
            "Explain how a transformer's attention mechanism works, with a tiny numeric example.",
            "How many r's are in the word 'strawberry'? Show your reasoning.",
            "Write a Python function that checks whether a number is prime, then explain its complexity.",
        ],
        show_configuration_controls=True,
        config={"max_tokens": 4096, "temperature": 0.6, "top_p": 0.95},
    )

    mo.vstack([mo.md(f"### Chat with `{model_id}`"), chat])
    return


if __name__ == "__main__":
    app.run()
