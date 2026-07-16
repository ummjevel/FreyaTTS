"""FreyaTTS high-level synthesis pipeline.

Usage:
    from freyatts import FreyaTTS
    tts = FreyaTTS.from_pretrained("freyavoice/freya-tts", device="cuda")
    wav = tts.synthesize("안녕하세요, 어떻게 도와드릴까요?")
    tts.save_wav(wav, "output.wav")
"""

import json
import math
import os
import re

import numpy as np
import torch

from .hangul import decompose_hangul
from .model import DEFAULT_SEED, FreyaDiT
from .vae import load_audio_vae

FILL_ID = 0
UNK_ID = 1

SAMPLE_RATE = 48000

# minimum voiced fraction (pyin) below which a clause is re-sampled
VOICED_FRAC_MIN = 0.06

# English words/acronyms the character vocabulary cannot pronounce as written
# (Latin letters pass through unchanged rather than being read aloud);
# transliterated to their Korean-spelling pronunciation. Words already
# commonly written in Hangul (온라인, 모바일, ...) need no entry here.
BRAND = {
    "ai": "에이아이",
    "tv": "티비",
    "pc": "피씨",
    "qr": "큐알",
    "atm": "에이티엠",
    "ars": "에이알에스",
    "app": "앱",
    "wifi": "와이파이",
    "sos": "에스오에스",
}

_UNITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

# 1-12 in native-Korean attributive form, for clock hours ("한시", "열두시", ...)
_NATIVE_HOUR = ["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열", "열한", "열두"]

_SCALES = [(10 ** 12, "조"), (10 ** 8, "억"), (10 ** 4, "만")]


def _spell_group(n):
    """Spell 0-9999 in Sino-Korean (no leading '일' before 천/백/십, same as
    100 -> '백' not '일백')."""
    parts = []
    if n >= 1000:
        d = n // 1000
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("천")
        n %= 1000
    if n >= 100:
        d = n // 100
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("백")
        n %= 100
    if n >= 10:
        d = n // 10
        if d > 1:
            parts.append(_UNITS[d])
        parts.append("십")
        n %= 10
    if n > 0:
        parts.append(_UNITS[n])
    return "".join(parts)


def spell_number(n):
    """Spell a non-negative integer in Sino-Korean (0 -> '영').

    Grouped by 만/억/조 (10^4 steps), not by 10^3 like English/Turkish --
    Korean numerals read 12345 as '만이천삼백사십오', not 'twelve thousand...'.
    """
    if n == 0:
        return "영"
    parts = []
    remaining = n
    for scale_val, scale_name in _SCALES:
        if remaining >= scale_val:
            head = remaining // scale_val
            remaining %= scale_val
            # '일만'이 아니라 '만' -- same leading-one drop as _spell_group
            parts.append(scale_name if head == 1 else _spell_group(head) + scale_name)
    if remaining > 0 or not parts:
        parts.append(_spell_group(remaining))
    return "".join(parts)


def spell_hour(h):
    """Native-Korean hour word for a 12-hour clock (0 and 12 both -> '열두')."""
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return _NATIVE_HOUR[h12 - 1]


def expand_digits(text):
    """Rewrite digit runs in their spoken Korean form.

    A digit string is orthographically short but phonetically long, and the
    duration head sizes the utterance from the character sequence, so numbers
    left as digits come out truncated. Spelling them out before synthesis is
    part of the input contract.

    Clock times read as native-Korean hour + Sino-Korean minute (matching the
    todak-vox persona convention, e.g. "9:28" -> "아홉시 이십팔분"); decimals
    read digit-by-digit after '점'; long account-style digit runs (6+ digits,
    no separators) read digit by digit; everything else as one Sino-Korean
    integer.
    """
    def spoken(match):
        s = match.group(0)
        if ":" in s:
            hour, minute = s.split(":")
            out = spell_hour(int(hour)) + "시"
            if int(minute):
                out += " " + spell_number(int(minute)) + "분"
            return out
        s = s.replace(",", "")
        if "." in s:
            whole, frac = s.split(".", 1)
            out = spell_number(int(whole)) if whole else "영"
            out += " 점 " + " ".join(_UNITS[int(ch)] if int(ch) else "영" for ch in frac)
            return out
        if len(s) >= 6:
            return " ".join(_UNITS[int(ch)] if int(ch) else "영" for ch in s)
        return spell_number(int(s))

    return re.sub(r"\d+:\d+|\d[\d,]*(?:\.\d+)?", spoken, text)


def normalize(text):
    """Light text normalization: transliterate foreign acronyms, spell out
    digit runs, collapse punctuation, decompose Hangul syllables to jamo.

    Jamo decomposition happens last and must match whatever text the
    training manifest was built with (see training/build_manifest_ko.py) --
    it is the tokenization contract, not cosmetic normalization.
    """
    t = text
    for k in sorted(BRAND, key=len, reverse=True):
        # Korean particles attach directly to acronyms with no space ("AI가",
        # "TV를"), so \b (a \w/\W transition) doesn't fire at the Latin/Hangul
        # boundary -- Hangul counts as \w too. Block only a *longer* Latin
        # run on either side instead, so "AI가" matches but "AIRPLANE" doesn't.
        t = re.sub(r"(?i)(?<![a-z])" + re.escape(k) + r"(?![a-z])", BRAND[k], t)

    t = expand_digits(t)
    t = t.replace("...", ", ").replace("…", ", ").replace(" — ", ", ").replace(" - ", ", ")
    t = re.sub(r"\s+", " ", t).strip()
    t = decompose_hangul(t)
    return t


class FreyaTTS:
    """Text-to-speech pipeline around FreyaDiT and the frozen VoxCPM2 AudioVAE.

    A fixed noise seed gives one deterministic voice. Long inputs are split
    at clause boundaries and synthesized per clause with the same seed, then
    concatenated with short gaps.
    """

    def __init__(self, model, vae, char_to_id, device="cuda", seed=DEFAULT_SEED, t_floor=8, max_words=11):
        self.model = model
        self.vae = vae
        self.char_to_id = char_to_id
        self.device = device
        self.seed = DEFAULT_SEED if seed is None else seed
        self.t_floor = t_floor
        self.max_words = max_words
        self.sample_rate = SAMPLE_RATE

    @classmethod
    def from_pretrained(cls, model_id_or_path: str = "freyavoice/freya-tts", device: str = "cuda") -> "FreyaTTS":
        """Load FreyaTTS from a Hugging Face repo id or a local directory.

        Expects `config.json` and `model.safetensors` in the repo/directory.
        The AudioVAE is fetched separately from openbmb/VoxCPM2.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        if os.path.isdir(model_id_or_path):
            config_path = os.path.join(model_id_or_path, "config.json")
            weights_path = os.path.join(model_id_or_path, "model.safetensors")
        else:
            config_path = hf_hub_download(model_id_or_path, "config.json")
            weights_path = hf_hub_download(model_id_or_path, "model.safetensors")

        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        model = FreyaDiT(
            vocab=cfg["vocab"],
            d=cfg["d"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            ff=cfg["ff"],
        )
        model.load_state_dict(load_file(weights_path), strict=True)
        model = model.to(device).eval()

        vae = load_audio_vae(device)

        vocab_path = os.path.join(os.path.dirname(__file__), "char_vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            char_to_id = json.load(f)

        return cls(model, vae, char_to_id, device=device)

    def synthesize(self, text: str, steps: int = 32, seed: int = DEFAULT_SEED) -> np.ndarray:
        """Synthesize `text` and return a float32 waveform at 48 kHz.

        Args:
            text: Input text (Korean).
            steps: Euler ODE steps for the flow-matching sampler.
            seed: Noise seed, which selects the speaker — the model has no speaker
                conditioning, so x0 *is* the voice. The default (DEFAULT_SEED) gives
                whichever voice the current checkpoint was SFT'd on, deterministically;
                other seeds give other people, and most are not that voice at all.
        """
        wav, _, _ = self._synth(text, steps=steps, seed=seed)
        return wav

    def save_wav(self, wav: np.ndarray, path: str):
        """Write a waveform returned by `synthesize` to a 48 kHz wav file."""
        import soundfile as sf

        sf.write(path, wav, self.sample_rate)

    # ---- internals ----

    def _ids(self, text):
        return torch.tensor([[self.char_to_id.get(ch, UNK_ID) for ch in text]], device=self.device)

    @torch.no_grad()
    def _synth_one(self, text, steps=32, seed=None):
        seed = self.seed if seed is None else seed
        ids = self._ids(text)
        cmask = torch.ones_like(ids, dtype=torch.bool)

        # duration head: masked mean over text features -> log frame count
        te = self.model.text_encode(ids)
        pooled = (te * cmask[..., None].float()).sum(1) / (cmask.sum(1, keepdim=True) + 1e-6)
        T = int(round(math.exp(float(self.model.dur(pooled).squeeze(-1)))))
        # floor keeps short inputs from collapsing, cap keeps runaways bounded
        T = max(self.t_floor, ids.shape[1] + 4, min(300, T))

        # fixed seed = fixed voice
        latents = self.model.sample(ids, T, steps=steps, cmask=cmask, seed=seed)

        wav = self.vae.decode(latents.transpose(1, 2).float()).squeeze().float().cpu().numpy()
        return wav

    def _voiced_ok(self, wav):
        # a near-zero voiced fraction means the sample collapsed to noise/silence
        try:
            import librosa

            y = librosa.resample(wav.astype(np.float32), orig_sr=self.sample_rate, target_sr=16000)
            f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=16000)
            voiced_frac = float(np.mean(~np.isnan(f0)))
            return voiced_frac >= VOICED_FRAC_MIN
        except Exception:
            return True

    def _clauses(self, text):
        """Split text at punctuation into chunks of at most `max_words` words."""
        parts = re.split(r"(?<=[\.\?\!,:;])\s+", text)
        out = []
        cur = ""
        for p in parts:
            if len((cur + " " + p).split()) <= self.max_words:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    out.append(cur)
                cur = p
        if cur:
            out.append(cur)

        # hard-split any clause that is still too long
        final = []
        for c in out:
            words = c.split()
            if len(words) <= self.max_words + 4:
                final.append(c)
            else:
                for i in range(0, len(words), self.max_words):
                    final.append(" ".join(words[i : i + self.max_words]))
        return [c for c in final if c.strip()]

    def _synth(self, text, steps=32, seed=None, do_norm=True, do_chunk=True):
        seed = self.seed if seed is None else seed
        t = normalize(text) if do_norm else text
        # very short inputs rely on the duration floor

        if do_chunk and len(t.split()) > self.max_words:
            chunks = self._clauses(t)
        else:
            chunks = [t]

        wavs = []
        gap = np.zeros(int(0.12 * self.sample_rate), dtype=np.float32)
        for c in chunks:
            w = self._synth_one(c, steps=steps, seed=seed)
            if not self._voiced_ok(w):
                # unvoiced collapse: retry with seeds near the voice seed
                for offset in (1, 2, 3):
                    w2 = self._synth_one(c, steps=steps, seed=seed + offset)
                    if self._voiced_ok(w2):
                        w = w2
                        break
            wavs.append(w.astype(np.float32))
            wavs.append(gap)

        if len(wavs) > 1:
            wav = np.concatenate(wavs[:-1])
        else:
            wav = wavs[0]
        return wav, t, chunks
