"""File-bridge VLM backend answered by a live Claude Code agent.

Drop-in replacement for GeminiVLMServer: the tree search calls
generate_single_thought() exactly as before, but instead of an HTTPS call to
Gemini, each request is written to disk and the process blocks until a
response file appears. A Claude agent session watches the bridge directory,
reads the prompt + images, and writes the response in the same tagged text
format Gemini would produce, so all inherited parsers work unchanged.

Bridge protocol (one directory per request, atomic via .tmp + rename):

    <bridge_dir>/request_0001/
        meta.json     {"phase": ..., "num_images": N}
        prompt.txt    full prompt text ("<IMG>" placeholders mark image slots)
        img_00.png    images in placeholder order
        response.txt  written by the agent; its raw content is returned as the
                      VLM output (must contain <thinking>/<answer> tags for
                      actor phases, "## ..." sections for judge phases)

On timeout the same "Error: max retries exceeded" sentinel as the Gemini
wrapper is returned, so the all-None parse / graceful-terminate contract is
preserved.

Enable with skills.vlm.provider=claude_bridge (see conf/skills/default.yaml).
Run with actor/judge concurrency 1 so requests arrive in order.
"""

import argparse
import base64
import datetime
import json
import logging
import os
import threading
import time

from skills.gemini import GeminiVLMServer

logger = logging.getLogger(__name__)

_TIMEOUT_SENTINEL = "Error: max retries exceeded"


class ClaudeAgentBridgeVLM(GeminiVLMServer):
    def __init__(self, args) -> None:
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        bridge_root = args.get("bridge_dir") or os.path.join(os.getcwd(), "claude_bridge")
        run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_pid{os.getpid()}"
        self.bridge_dir = os.path.join(bridge_root, run_tag)
        self.poll_interval = float(args.get("bridge_poll_interval", 2.0))
        self.timeout_s = float(args.get("bridge_timeout_s", 3600.0))
        self._counter = 0
        self._counter_lock = threading.Lock()
        os.makedirs(self.bridge_dir, exist_ok=True)

        super().__init__(args)
        logging.warning(
            "[claude_bridge] VLM requests will be served by a Claude agent. "
            "Bridge dir: %s (timeout %.0fs)", self.bridge_dir, self.timeout_s,
        )

    def init_model(self):
        # No API client; responses come from the agent via the file bridge.
        return None

    def generate_single_thought(self, prompt, phase, **kwargs):
        prompt = dict(prompt)
        prompt["_phase"] = phase
        return super().generate_single_thought(prompt, phase, **kwargs)

    def _generate_raw(
        self,
        prompt=None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        max_output_tokens: int = None,
    ) -> str:
        query_text = prompt["text"]
        query_images = prompt.get("images", None) or []
        phase = prompt.get("_phase", "unknown")

        with self._counter_lock:
            self._counter += 1
            idx = self._counter

        req_dir = os.path.join(self.bridge_dir, f"request_{idx:04d}")
        tmp_dir = req_dir + ".tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        with open(os.path.join(tmp_dir, "prompt.txt"), "w") as f:
            f.write(query_text)
        for i, b64 in enumerate(query_images):
            if isinstance(b64, str) and b64.startswith("data:image/"):
                b64 = b64.split(",")[1]
            with open(os.path.join(tmp_dir, f"img_{i:02d}.png"), "wb") as f:
                f.write(base64.b64decode(b64))
        with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
            json.dump({"phase": phase, "num_images": len(query_images)}, f)
        os.rename(tmp_dir, req_dir)

        logging.warning("[claude_bridge] waiting for agent: %s (phase=%s)", req_dir, phase)
        resp_path = os.path.join(req_dir, "response.txt")
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if os.path.exists(resp_path):
                with open(resp_path) as f:
                    text = f.read()
                if text.strip():
                    return text
            time.sleep(self.poll_interval)

        logging.error("[claude_bridge] timed out after %.0fs on %s", self.timeout_s, req_dir)
        return _TIMEOUT_SENTINEL
