<h2 align="center">FreyaTTS (Korean fork): An Efficient 183M Speech Foundation Model</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2607.09530"><img src="https://img.shields.io/badge/arXiv-2607.09530-b31b1b" alt="arXiv"></a>
  <a href="https://github.com/freyavoiceai/FreyaTTS"><img src="https://img.shields.io/badge/upstream-freyavoiceai%2FFreyaTTS-blue" alt="Upstream"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
</p>

This is a **Korean-language fork** of [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS), a 183M-parameter TTS model originally released for Turkish. The architecture, the frozen AudioVAE2 latent space, and the training pipeline are all language-agnostic by construction; only the character vocabulary and the digit/clock-time text normalization were Turkish-specific. This fork replaces both with Korean equivalents:

- `freyatts/hangul.py` + `freyatts/char_vocab.json` -- Hangul jamo decomposition (see [Korean vocabulary](#korean-vocabulary) below), 127 symbols
- `freyatts/pipeline.py` -- Sino-Korean/native-Korean number and clock-time spelling, Korean acronym transliteration
- `training/build_manifest_ko.py` -- new script to build a training manifest from a Korean (wav, transcript) corpus
- `eval/prompts_ko.json`, `eval/benchmark.py`, `eval/speed.py` -- Korean eval prompts and WER scoring (Whisper `language="ko"`)

**No Korean weights are trained yet.** This fork prepares the code so a Korean checkpoint *can* be trained (see [Training](#training)); it does not ship one. Everything the original README said about the Turkish release's benchmarked WER/CER/RTF numbers is architecture-level context, not a claim about Korean quality -- that has to be measured after training.

It is tokenizer-free at the character (jamo) level -- no phonemizer, no G2P -- and generates speech with a **non-autoregressive conditional flow-matching DiT** in the frozen AudioVAE2 latent space (25 Hz, 64-dim latents, 16 kHz encode / 48 kHz decode).

The result is a model that runs comfortably where 2B-class TTS models cannot: 1.5 GB of VRAM on a GPU, real time on a laptop CPU, and well under real time on the Apple Neural Engine -- architecture-level numbers from the Turkish release, expected to carry over at the same model size, not yet re-measured for Korean.

### Highlights

- **Small and fast** - 183.2M parameters; the Turkish release measured RTF 0.10-0.11 and TTFT ~0.5 s on an RTX 4090, about 3.2x faster RTF and 3.7x less VRAM than the 2B VoxCPM2
- **Tokenizer-free Korean** - Hangul syllables decomposed to jamo at the character level, 127-symbol vocabulary, no phonemizer or G2P stage to maintain
- **Non-autoregressive** - a single 32-step Euler ODE per clause, no classifier-free guidance needed
- **48 kHz output** - the frozen AudioVAE2 decodes 25 Hz latents straight to 48 kHz audio
- **Runs on CPUs** - the Turkish release measured real time on an Apple M3 laptop CPU (RTF 0.70 fp32); re-benchmark before relying on this for Korean, especially on weaker CPUs (e.g. Raspberry Pi/CM4-class ARM cores are considerably slower per-core than Apple M-series)
- **Apache-2.0** - code and the AudioVAE2 dependency are both free for commercial use (see [License](#license) for the one thing *you* control: training-corpus licensing)

---

## Quick Start

### Installation

```sh
git clone https://github.com/ummjevel/FreyaTTS.git
cd FreyaTTS
pip install -r requirements.txt
```

**There is no pretrained Korean checkpoint yet.** `infer.py` / `batch_infer.py` / the Python API below all need a local directory with `config.json` + `model.safetensors` from a training run you complete yourself (see [Training](#training)) -- `from_pretrained("freyavoice/freya-tts", ...)` would load the original *Turkish* weights, which won't produce intelligible Korean. Once you have a checkpoint directory, e.g. `checkpoints/sft_stage2/final`, point every command below at it.

### Command line

```sh
python infer.py --text "안녕하세요, 어떻게 도와드릴까요?" --model checkpoints/sft_stage2/final --out output.wav
```

### Batch inference

Non-autoregressive generation batches naturally: one masked ODE solve serves a
whole batch of requests. On an RTX 4090 the Turkish release reached about 65
seconds of audio per wall-second at batch size 8 in under 4 GB of VRAM.

```sh
python batch_infer.py --texts texts.txt --model checkpoints/sft_stage2/final --outdir wavs/ --batch-size 8
```

### Python API

```python
from freyatts import FreyaTTS

tts = FreyaTTS.from_pretrained("checkpoints/sft_stage2/final", device="cuda")
wav = tts.synthesize("안녕하세요, 어떻게 도와드릴까요?")   # np.float32, 48 kHz
tts.save_wav(wav, "output.wav")
```

`from_pretrained` also accepts a Hugging Face repo id once you publish one. `synthesize` takes optional `steps` (default 32) and `seed` (default `DEFAULT_SEED`) arguments.

**The seed selects the speaker.** FreyaTTS conditions only on text — there is no speaker embedding, speaker id, or reference-audio prefix — so the initial flow-matching noise `x0` *is* the voice. SFT collapses the model onto whichever speaker's corpus you fine-tune on, but the model doesn't condition on that speaker's identity, so a different seed gives a different (arbitrary) person, and most seeds won't sound like your target speaker at all. Pick your production seed empirically after SFT stage 1 (synthesize a handful of seeds, keep the one that locked onto your speaker) and pin it as the new `DEFAULT_SEED` in `freyatts/model.py`. Use one seed per utterance (the pipeline already shares it across clauses of a long input); passing `seed=None` opts into random-speaker sampling. Long inputs are normalized and split into clauses automatically, then joined with short pauses. Normalization spells out digit runs and clock times in Korean (`9:28` becomes `아홉시 이십팔분`, matching the native-hour + Sino-Korean-minute convention already used in todak-vox's persona layer): the duration predictor sizes the utterance from the character sequence, so numbers left as digits come out truncated.

---

## Korean vocabulary

Hangul composes each syllable block (가, 닥, ...) into a single codepoint --
11,172 possible blocks, too many and too sparse to embed directly at
character level the way Turkish's ~90-symbol Latin alphabet works. Instead,
`freyatts/hangul.py` decomposes every block into its choseong/jungseong/
jongseong jamo (already single codepoints in the Hangul Compatibility Jamo
block), which collapses the vocabulary to 51 unique jamo symbols -- still a
pure Unicode-arithmetic mapping, not a phonemizer or pronunciation
dictionary, so the model stays tokenizer-free in the same sense the Turkish
release was.

```python
from freyatts.hangul import decompose_hangul, compose_hangul

decompose_hangul("토닥아, 오늘 날씨 어때?")
# 'ㅌㅗㄷㅏㄱㅇㅏ, ㅇㅗㄴㅡㄹ ㄴㅏㄹㅆㅣ ㅇㅓㄸㅐ?'
compose_hangul(decompose_hangul("토닥아, 오늘 날씨 어때?")) == "토닥아, 오늘 날씨 어때?"  # True
```

`freyatts/char_vocab.json` (127 symbols: `<FILL>`, `<UNK>`, space, 12
punctuation marks, 10 digits, 52 Latin letters, 51 jamo) is generated from
`hangul.py`'s tables rather than hand-counted -- regenerate it with:

```sh
python3 -c "
import json, importlib.util
spec = importlib.util.spec_from_file_location('hangul', 'freyatts/hangul.py')
hangul = importlib.util.module_from_spec(spec); spec.loader.exec_module(hangul)
specials = ['<FILL>', '<UNK>']
base = [' ', '!', '\"', \"'\", ',', '-', '.', ':', ';', '?', '…', '~']
digits = [str(d) for d in range(10)]
latin = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
symbols = specials + base + digits + latin + hangul.JAMO_SYMBOLS
json.dump({s: i for i, s in enumerate(symbols)}, open('freyatts/char_vocab.json', 'w', encoding='utf-8'), ensure_ascii=False)
"
```

Latin letters are kept (unlike a Hangul-only vocabulary) because everyday
Korean text mixes in untransliterated acronyms (AI, TV, PC, ...) constantly
-- `freyatts/pipeline.py`'s `BRAND` dict transliterates the common ones
("AI" -> 에이아이) before decomposition; anything not in that dict passes
through as raw Latin letters, which the model was never trained to read, so
extend `BRAND` as you find gaps in your data.

**Known limitation inherited from the Turkish original:** `expand_digits()`
reads any digit run under 6 digits as one spelled-out number, not
digit-by-digit -- fine for prices/quantities, but a hyphenated phone number
like `010-1234-5678` gets read as three arbitrary numbers ("십-천이백삼십사-...")
rather than "공일공-일이삼사-...". If your training corpus contains phone
numbers or IDs written with separators, either pre-expand them digit-by-digit
in your manifest before calling `normalize()`, or extend `expand_digits()`
with a phone-number-shaped regex branch.

---

## Model

FreyaTTS is a conditional flow-matching diffusion transformer (DiT) operating in the latent space of the frozen AudioVAE2:

- **Text encoding:** character-level Korean (jamo-decomposed), 127 symbols shipped with the package (`freyatts/char_vocab.json`)
- **Generation:** non-autoregressive flow matching, 32 Euler ODE steps, no CFG
- **Latent space:** 64-dim latents at 25 Hz; AudioVAE2 encodes at 16 kHz and decodes at 48 kHz
- **Training:** from scratch on Korean speech; pretraining followed by SFT stage 1/2 (voice lock, short-utterance coverage)

AudioVAE2 is not retrained. It is downloaded from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0) at load time via the `voxcpm` package -- this part is untouched by the Korean fork and needs no re-training itself.

## Performance

Not yet measured for Korean. The Turkish release's numbers (kept below for
architecture-level context -- expect similar orders of magnitude at the same
183M size, but WER/CER are meaningless across languages and RTF depends on
your target hardware):

| Setting | RTF | Notes |
| ------- | --- | ----- |
| RTX 4090 | 0.10-0.11 | TTFT ~0.5 s, 1.5 GB VRAM, 9.4 audio-s/s at concurrency 4 |
| Apple M3 CPU (fp32) | 0.70 | real time on a laptop, no GPU |
| Apple silicon, Core ML | ~0.12 | end to end via the Neural Engine |

If your deployment target is a weaker CPU (e.g. Raspberry Pi/CM4-class ARM,
no GPU), re-run `eval/speed.py` there before assuming real-time -- a
Cortex-A72 core is considerably slower per-core than Apple M-series, and
0.70 RTF on an M3 could land above 1.0 (slower than real time) on that class
of hardware.

Once you have a Korean checkpoint, reproduce with `eval/benchmark.py`
(WER/CER, `--data your_ko_eval.jsonl`) and `eval/speed.py`
(`--prompts eval/prompts_ko.json`, latency/TTFT/RTF/concurrency).

---

## Training

The training pipeline itself needs no changes for Korean -- it already reads
`vocab_json` from config and applies no language-specific logic. What's
Korean-specific is getting your corpus into the `{"audio": ..., "text": ...}`
manifest format it expects, with `text` already digit-expanded and
jamo-decomposed (that's the tokenization contract -- training never
normalizes on the fly, unlike the inference-time `FreyaTTS.synthesize()`
path).

### 0. Pick a corpus (license matters here)

FreyaTTS's own code and its AudioVAE2 dependency are both Apache-2.0, but a
model **fine-tuned on a non-commercially-licensed voice corpus inherits that
restriction in practice** even though the code doesn't. For a Korean single-
speaker TTS corpus specifically:

| Corpus | License | Notes |
| --- | --- | --- |
| KSS (Korean Single Speaker Speech) | ❌ CC BY-NC-SA 4.0 | The most commonly used Korean TTS corpus in tutorials/papers -- **avoid for anything commercial** |
| [Zeroth-Korean](https://openslr.org/40/) | ✅ CC BY 4.0 | 51.6h, 105 speakers (ASR-oriented, not single-speaker) -- filter to your best-covered speaker(s), or train a small multi-speaker model |
| [Common Voice Korean](https://commonvoice.mozilla.org/) | ✅ CC0 | Crowd-sourced, quality/equipment varies -- weaker fit for a consistent persona voice |
| Your own voice-actor recording | ✅ full ownership | Cleanest option for a production persona voice; costs studio time instead of a license risk |

### 1. Build the manifest

```sh
python training/build_manifest_ko.py \
    --metadata /path/to/corpus/metadata.csv \
    --wav-dir /path/to/corpus/wavs \
    --out data/manifest.jsonl
```

Expects a delimited (default `|`) text file, one utterance per line:
`audio_id_or_path|transcript[|...extra columns ignored...]`. It resolves
`--audio-col` (default 0) against `--wav-dir`, applies `freyatts.pipeline
.normalize()` to `--text-col` (default 1) -- digit/clock-time spelling,
acronym transliteration, jamo decomposition -- and writes
`{"audio": ..., "text": ...}` JSONL. Pass `--debug-readable path.jsonl` to
also dump a pre-decomposition (human-readable) copy for spot-checking.

### 2. Encode, pretrain, SFT

```sh
# encode audio to AudioVAE2 latents once, up front
python training/precompute_latents.py --manifest data/manifest.jsonl --output-dir data/latents

# pretraining
python training/pretrain.py --config training/configs/pretrain.yaml

# SFT stage 1/2 (voice lock, short-utterance coverage)
python training/sft.py --config training/configs/sft_stage1.yaml
python training/extract_short_segments.py --manifest data/manifest.jsonl --out data/latents_short  # feeds stage 2
python training/sft.py --config training/configs/sft_stage2.yaml
```

`configs/*.yaml` already default `vocab_json: freyatts/char_vocab.json`
(now the Korean vocab) -- no config changes needed to switch languages, only
the data.

`extract_short_segments.py` force-aligns with torchaudio's `MMS_FA` aligner,
which works over a romanized (a-z) vocabulary. The Turkish original shipped
a `TR_TO_ASCII` diacritic-strip table for this alignment-only step, which
doesn't generalize to Hangul syllable blocks; this fork romanizes through
[`uroman`](https://github.com/isi-nlp/uroman) instead (the tool MMS_FA's own
multilingual examples are built around) -- extracted text keeps the
original Hangul, romanization is alignment-only. Word-span length and
duration thresholds are also retuned for Korean (`--max_words 3`, `--max_s
3.0`, vs. the Turkish defaults of 2 and 1.6) to reach 1-3 eojeol
conversational replies within a 1-3s window; the underlying 0.2s minimum
duration is unchanged, so single-eojeol acknowledgements ("네", "응") under
1s are still kept, unlike pipelines that hard-drop anything under 1s.
`training/run_extract_short_ko.sh` wraps the tuned invocation with the
reasoning for each argument inline.

---

## License

FreyaTTS code (this fork included) is released under the [Apache-2.0](LICENSE) license. The frozen AudioVAE2 dependency is also Apache-2.0. **The one license decision left to you is the training corpus** -- see the table above.

## Acknowledgments

- [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS) for the original Turkish model this fork retargets
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (Apache-2.0) for the AudioVAE2 that FreyaTTS generates into; FreyaTTS reuses it frozen and unchanged
- The flow matching and DiT literature this model builds on

## Citation

FreyaTTS is described in our technical report, [arXiv:2607.09530](https://arxiv.org/abs/2607.09530):

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
