import os
import sys
import json
import math
import threading
import asyncio

import gdown
import torch
torch.set_num_threads(1)  # free tier gives ~0.15 CPU; more threads just adds overhead
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ==================== CONFIG ====================

MODEL_PATH = "final_model.pt"
FILE_ID = "1PAS6vt7P50uLaxjsmibxyATyQs8x7mhj"

class Config:
    vocab_size = 50257
    n_embed = 512
    n_head = 8
    n_blocks = 8
    context_length = 256
    dropout = 0.1
    device = "cpu"  # Render free tier has no GPU

config = Config()

# The checkpoint was originally pickled with Config living in whatever module
# was __main__ at training time (e.g. `python train.py`). Depending on how
# this server is launched (plain `python main.py` vs `uvicorn main:app --reload`,
# which runs a separate reloader/supervisor process as __main__), `main` may
# or may not be the actual __main__ module. Aliasing Config onto sys.modules
# under both names makes torch.load's unpickler able to find it either way.
sys.modules["__main__"].Config = Config

# ==================== GLOBAL STATE (for status/progress reporting) ====================

class ModelState:
    def __init__(self):
        self.status = "starting"   # starting -> downloading -> loading -> ready -> error
        self.progress = 0.0        # 0-100, download percentage
        self.detail = "Waking up..."
        self.error = None
        self.model = None
        self.tokenizer = None

state = ModelState()

# ==================== MODEL ARCHITECTURE (must match training) ====================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim
    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos_emb + rotated * sin_emb

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rope = RoPE(self.head_dim, max_seq_len=config.context_length)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1).bool()
        )
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.rope(q, T)
        k = self.rope(k, T)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.w2 = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.w3 = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embed)
        self.mlp = SwiGLU(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SmallLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.ln_f = RMSNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.context_length
        x = self.token_embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens=150, temperature=0.8, top_k=40, eos_token_id=None):
        """Yields one new token id at a time instead of returning all at once."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            token_id = idx_next.item()
            if eos_token_id is not None and token_id == eos_token_id:
                return
            idx = torch.cat((idx, idx_next), dim=1)
            yield token_id

# ==================== DOWNLOAD + LOAD (runs in background thread on startup) ====================

def _download_with_progress():
    """Downloads via gdown in a background thread, polling file size for progress.
    gdown doesn't expose a progress callback, so we estimate progress from the
    growing file size against the known final size from the Drive listing."""
    KNOWN_SIZE_BYTES = 226.7 * 1024 * 1024

    def _do_download():
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=True
        )

    dl_thread = threading.Thread(target=_do_download, daemon=True)
    dl_thread.start()

    while dl_thread.is_alive():
        if os.path.exists(MODEL_PATH):
            size = os.path.getsize(MODEL_PATH)
            pct = min(99.0, (size / KNOWN_SIZE_BYTES) * 100)
            state.progress = round(pct, 1)
            state.detail = f"Downloading model... {state.progress}%"
        threading.Event().wait(1.0)

    dl_thread.join()

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10 * 1024 * 1024:
        raise RuntimeError(
            "Download failed or file is too small — Google Drive may have served "
            "a virus-scan warning page instead of the file."
        )
    state.progress = 100.0


def load_model_in_background():
    try:
        if not os.path.exists(MODEL_PATH):
            state.status = "downloading"
            state.detail = "Downloading model checkpoint..."
            _download_with_progress()
        else:
            state.progress = 100.0

        state.status = "loading"
        state.detail = "Loading tokenizer..."
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        state.detail = "Loading model weights into memory..."
        # Belt-and-suspenders: the sys.modules alias above should already make
        # `__main__.Config` resolvable, but if this process was launched in a
        # way that still can't find it there, explicitly allow-listing the
        # class covers the gap. Safe here since this is our own checkpoint.
        try:
            torch.serialization.add_safe_globals([Config])
        except AttributeError:
            pass  # older torch versions without safe_globals support

        # Free tier has 512MB RAM total, which is tight for torch + transformers
        # + a 227MB checkpoint. Keep peak memory down by discarding the raw
        # checkpoint dict as soon as we've pulled the state_dict out of it,
        # and forcing a GC pass before/after the heaviest allocations.
        import gc
        gc.collect()

        checkpoint = torch.load(MODEL_PATH, map_location=config.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        del checkpoint
        gc.collect()

        model = SmallLM(config).to(config.device)
        # assign=True makes load_state_dict swap tensors in directly instead of
        # copying into pre-allocated ones, avoiding a moment where both the
        # freshly-initialized weights and the checkpoint's weights exist at once.
        try:
            model.load_state_dict(state_dict, assign=True)
        except TypeError:
            # older torch versions without the assign= kwarg
            model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
        model.eval()

        state.model = model
        state.tokenizer = tokenizer
        state.status = "ready"
        state.detail = "Model ready."
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        state.detail = f"Failed: {e}"

# ==================== FASTAPI APP ====================

app = FastAPI(title="SmallLM Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend's domain once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    threading.Thread(target=load_model_in_background, daemon=True).start()

@app.get("/status")
def get_status():
    return {
        "status": state.status,
        "progress": state.progress,
        "detail": state.detail,
        "error": state.error,
    }

class GenerateRequest(BaseModel):
    instruction: str
    input_text: str = ""
    max_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 40

def build_prompt(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    async def event_generator():
        if state.status != "ready":
            yield f"data: {json.dumps({'type': 'error', 'message': f'Model not ready (status: {state.status})'})}\n\n"
            return

        model = state.model
        tokenizer = state.tokenizer
        prompt = build_prompt(req.instruction, req.input_text)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)

        max_tokens = min(max(req.max_tokens, 1), 512)

        try:
            token_gen = model.generate_stream(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                eos_token_id=tokenizer.eos_token_id,
            )
            for token_id in token_gen:
                piece = tokenizer.decode([token_id], skip_special_tokens=True)
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
                await asyncio.sleep(0)  # yield control so the response actually flushes
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disables proxy buffering (relevant behind nginx-like proxies)
        },
    )

@app.get("/")
def root():
    return {"message": "SmallLM API is running. See /status and POST /generate/stream."}
