---
language:
  - ko
license: apache-2.0
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - tts
  - korean
  - flow-matching
  - diffusion-transformer
  - speech-synthesis
library_name: freyatts
---

# FreyaTTS (Korean fork)

FreyaTTS is a 183M-parameter TTS model, originally released for Turkish by
[freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS). This fork
retargets it to Korean: a Hangul jamo character vocabulary
(`freyatts/char_vocab.json`, `freyatts/hangul.py`) replaces the Turkish
92-symbol table, and text normalization (`freyatts/pipeline.py`) spells out
digits and clock times in Korean instead of Turkish. Everything else --
architecture, the frozen [AudioVAE2](https://huggingface.co/openbmb/VoxCPM2)
latent space, the training pipeline -- is unchanged and equally language
agnostic. It is tokenizer-free at the character (jamo) level -- no
phonemizer or G2P -- and generates speech with a non-autoregressive
conditional flow-matching DiT (25 Hz, 64-dim latents, 16 kHz encode / 48 kHz
decode). Output is 48 kHz mono.

- **Upstream:** https://github.com/freyavoiceai/FreyaTTS
- **License:** Apache-2.0 (code, and the AudioVAE2 dependency it downloads)
- **Status:** vocabulary and text pipeline are adapted; **no Korean weights
  are trained yet** -- see README's Training section for the manifest ->
  pretrain -> SFT path.

## Usage

```python
from freyatts import FreyaTTS

tts = FreyaTTS.from_pretrained("/path/to/a/korean/checkpoint", device="cuda")
wav = tts.synthesize("안녕하세요, 어떻게 도와드릴까요?")   # np.float32, 48 kHz
tts.save_wav(wav, "output.wav")
```

There is no pretrained Korean checkpoint published yet -- `from_pretrained`
needs a local directory with `config.json` + `model.safetensors` from your
own training run (see README).

## Model details

- **Architecture:** conditional flow-matching diffusion transformer, non-autoregressive, 32-step Euler ODE, no CFG
- **Parameters:** 183.2M
- **Input:** character-level Korean, jamo-decomposed (51 jamo + punctuation/digits/Latin, 127-symbol vocabulary)
- **Latent space:** frozen AudioVAE2 (Apache-2.0, [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)), 64-dim at 25 Hz, decodes to 48 kHz
- **Training:** from scratch on Korean speech; pretraining followed by SFT stage 1/2 (voice lock, short-utterance coverage)
- **Voice:** single target speaker, no cloning

## Evaluation

Not yet run -- the original release's WER/CER/RTF numbers are for the
Turkish model on Freya-TR-Eval and do not apply to a Korean checkpoint.
`eval/benchmark.py` and `eval/speed.py` are already Korean-ready
(`eval/prompts_ko.json`, Whisper `language="ko"`); re-run them once a
checkpoint exists. There is no Korean equivalent of Freya-TR-Eval published
yet -- supply your own held-out sentences via `--data`.

## Speed (architecture-level, from the Turkish release; expect similar for Korean at the same size)

- RTX 4090: RTF 0.10-0.11, TTFT ~0.5 s, 1.5 GB VRAM, 9.4 audio-s/s at concurrency 4
- Apple M3 laptop CPU: RTF 0.70 (fp32); ~0.12 end to end via Core ML on Apple silicon

## Citation

Architecture described in the upstream technical report, [arXiv:2607.09530](https://arxiv.org/abs/2607.09530):

```bibtex
@misc{pamuk2026freyattstechnicalreport,
      title={FreyaTTS Technical Report}, 
      author={Ahmet Erdem Pamuk and Ömer Yentür and Ahmet Tunga Bayrak and Yavuz Alp Sencer Öztürk and Mustafa Yavuz},
      year={2026},
      eprint={2607.09530},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.09530}, 
}
```
