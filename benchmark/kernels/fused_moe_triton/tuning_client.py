import argparse
import os
import time
import urllib.request
from typing import Any, Dict, Optional

import openai

"""
# Edit the code file srt/models/deepseek_v2.py in the Python site package and add the logic for saving topk_ids:
# import get_tensor_model_parallel_rank
# DeepseekV2MoE::forward_normal
if hidden_states.shape[0] >= 4096 and get_tensor_model_parallel_rank() == 0:
    topk_ids_dir = xxxx
    if not hasattr(self, "save_idx"):
        self.save_idx = 0
    if self.save_idx <= 1:
        torch.save(topk_output.topk_ids, f"{topk_ids_dir}/topk_ids_layer{self.layer_id}_idx{self.save_idx}.pt")
    self.save_idx += 1
"""


def read_long_prompt():
    import json

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/tuning_text.json", "r") as fp:
        text = fp.read()
    rst = json.loads(text)
    return rst["prompt"]


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    import json
    print(f"[tuning_client] POST: {url}")
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def truncate_prompt_to_server_context(
    prompt: str,
    ip: str,
    port: int,
    model: str,
    max_output_tokens: int,
    max_input_tokens: Optional[int] = None,
) -> str:
    """Truncate `prompt` to fit server context length.

    Uses the server's tokenizer via `/v1/tokenize` and `/v1/detokenize`.
    This avoids guessing token counts client-side.
    """

    base_url = f"http://{ip}:{port}/v1"
    tok = _post_json(
        f"{base_url}/tokenize",
        {
            "model": model,
            "prompt": prompt,
            "add_special_tokens": False,
        },
    )
    token_ids = tok["tokens"]
    prompt_tokens = int(tok["count"])
    server_max_model_len = int(tok["max_model_len"])

    # Be conservative: reserve output tokens in the same context window.
    # Some servers validate input-only, but this makes the client robust.
    hard_limit = server_max_model_len - int(max_output_tokens)
    if max_input_tokens is not None:
        hard_limit = min(hard_limit, int(max_input_tokens))
    hard_limit = max(hard_limit, 1)

    if prompt_tokens <= hard_limit:
        return prompt

    detok = _post_json(
        f"{base_url}/detokenize",
        {
            "model": model,
            "tokens": token_ids[:hard_limit],
            "skip_special_tokens": True,
        },
    )
    truncated = detok["text"]
    print(
        f"[tuning_client] Prompt too long: {prompt_tokens} tokens; "
        f"truncated to {hard_limit} (server max_model_len={server_max_model_len})."
    )
    return truncated


def openai_stream_test(model, ip, port, max_input_tokens: Optional[int] = None):
    client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="None")
    max_tokens = 100
    qst = truncate_prompt_to_server_context(
        read_long_prompt(),
        ip=ip,
        port=port,
        model=model,
        max_output_tokens=max_tokens,
        max_input_tokens=max_input_tokens,
    )

    messages = [
        {"role": "user", "content": qst},
    ]
    msg2 = dict(
        model=model,
        messages=messages,
        temperature=0.6,
        top_p=0.75,
        max_tokens=max_tokens,
    )
    response = client.chat.completions.create(**msg2, stream=True)
    time_start = time.time()
    time_cost = []
    for chunk in response:
        time_end = time.time()
        # if chunk.choices[0].delta.content:
        #    print(chunk.choices[0].delta.content, end="", flush=True)
        time_cost.append(time_end - time_start)
        time_start = time.time()

    ttft = time_cost[0] + time_cost[1]
    tpot = sum(time_cost[2:]) / len(time_cost[2:])
    print(f"\nTTFT {ttft}, TPOT {tpot}")
    return ttft, tpot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="auto")
    parser.add_argument(
        "--ip",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=(
            "Optional hard cap for prompt tokens before sending to server. "
            "If unset, uses the server tokenizer's max_model_len minus max_tokens."
        ),
    )
    args = parser.parse_args()
    openai_stream_test(args.model, args.ip, args.port, args.max_input_tokens)
