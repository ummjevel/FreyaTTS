<h2 align="center">FreyaTTS (Korean fork): A 337M On-Device Voice-Chat TTS, Distilled from Qwen3-TTS</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2607.09530"><img src="https://img.shields.io/badge/arXiv-2607.09530-b31b1b" alt="arXiv"></a>
  <a href="https://github.com/freyavoiceai/FreyaTTS"><img src="https://img.shields.io/badge/upstream-freyavoiceai%2FFreyaTTS-blue" alt="Upstream"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>
</p>

This is a **Korean-language fork** of [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS), retargeted to Korean and trained from scratch, then distilled onto 5 voices synthesized by a [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) teacher. The architecture, the frozen AudioVAE2 latent space, and the training pipeline are all language-agnostic by construction; only the character vocabulary and the digit/clock-time text normalization were Turkish-specific. This fork replaces both with Korean equivalents:

- `freyatts/hangul.py` + `freyatts/char_vocab.json` -- Hangul jamo decomposition (see [Korean vocabulary](#korean-vocabulary) below), 127 symbols
- `freyatts/pipeline.py` -- Sino-Korean/native-Korean number and clock-time spelling, Korean acronym transliteration
- `training/build_manifest_ko.py` -- new script to build a training manifest from a Korean (wav, transcript) corpus
- `eval/prompts_ko.json`, `eval/benchmark.py`, `eval/speed.py` -- Korean eval prompts and WER scoring (Whisper `language="ko"`)

> **한국어 요약**: [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS)(터키어용으로 공개된 flow-matching TTS)를 한국어로 포팅한 fork입니다. 아키텍처와 AudioVAE2 잠재공간, 학습 파이프라인은 언어에 무관하게 그대로 재사용하고, 문자 vocab과 숫자/시각 표기만 한국어(자모 분해, 한자어/고유어 숫자 읽기)로 교체했습니다.

**Status: Korean weights are trained and voice-cloned onto 5 target voices.** This is a from-scratch Korean pretrain (not a fine-tune of the Turkish checkpoint), followed by distillation from a Qwen3-TTS teacher onto 5 synthetic target voices. See [Voices](#voices) and [Evaluation](#evaluation) below for what's actually measured, not projected.

> **상태 (한국어)**: 한국어 가중치는 이미 학습 완료 상태이며, 5개 타겟 목소리로 distillation까지 끝났습니다. 터키어 체크포인트를 파인튜닝한 게 아니라 한국어로 처음부터(from scratch) pretrain했습니다. 실제 측정치는 [Evaluation](#evaluation) 참고 — 추정치가 아니라 실측입니다.

**Correction vs. the upstream paper's parameter count**: the upstream FreyaTTS technical report states 183.2M parameters, but `training/configs/pretrain.yaml`'s actual dims (`d_model=768, depth=22, ff=2048`) build a **337M**-parameter model (confirmed at load time: `eval/results/speed_final.json` reports `"params": 337182785`). Every *voice-locked* checkpoint in this repo (`checkpoints/distill_voice{A..E}`) is this 337M config, not 183M.

Three smaller re-pretrains have since been trained from scratch to 150k steps on the same corpus and benchmarked -- **88M**, **127M**, and **183M** (`checkpoints/pretrain_{88M,127M,183M}/`). The 183M variant (`d_model=640, depth=16, heads=10, ff=2048`) loads at 183,220,545 parameters, matching the paper's stated 183.2M exactly. **All three score better CER than the 337M model** (see [Size sweep](#size-sweep-pretrain-checkpoints-150k-steps)), so the larger config is not buying accuracy on this corpus. The 183M and 88M have since been distilled onto all five voices, and the 183M is now the recommended configuration -- see [On-device configuration](#on-device-configuration).

> **파라미터 수 정정 (한국어)**: 원 논문은 183.2M이라고 표기하지만, 실제 `pretrain.yaml` 설정(d_model=768, depth=22, ff=2048)으로 빌드되는 모델은 **337M**입니다 (`eval/results/speed_final.json`의 `"params": 337182785`로 실측 확인). 목소리가 고정된 체크포인트(`checkpoints/distill_voice{A..E}`)는 전부 337M입니다. 이후 동일 코퍼스로 **88M / 127M / 183M** 세 가지를 150k step까지 from-scratch 재학습하고 벤치까지 마쳤습니다. 이 중 183M(d_model=640, depth=16)은 실측 183,220,545개로 논문 표기 183.2M과 정확히 일치합니다. **세 모델 모두 337M보다 WER이 낮습니다** ([Size sweep](#size-sweep-pretrain-checkpoints-150k-steps) 참고) — 이 코퍼스에서는 큰 설정이 정확도를 사주지 못한다는 뜻입니다. 183M과 88M은 이후 5개 목소리 전부에 distill을 마쳤고, **현재 권장 구성은 183M**입니다 ([On-device configuration](#on-device-configuration) 참고).

It is tokenizer-free at the character (jamo) level -- no phonemizer, no G2P -- and generates speech with a **non-autoregressive conditional flow-matching DiT** in the frozen AudioVAE2 latent space (25 Hz, 64-dim latents, 16 kHz encode / 48 kHz decode).

### Highlights

- **183M is the configuration to use**, not the 337M the voices were first built on. With 16 ODE steps and clause splitting it scores CER 0.086 against 0.123 for the 337M default, in half the size -- see [On-device configuration](#on-device-configuration)
- **Runs on a CPU** - exported to ONNX (`eval/export_onnx.py`), the 183M reaches RTF 0.40-0.54 on four CPU threads, so no GPU is required at inference. The 337M/32-step default is RTF 3.26 and cannot keep up with real time on a CPU
- **Clause splitting is not optional** - accuracy falls off with utterance length (median CER 0.000 under 20 characters, 0.137 over 60), so `max_words` does real work: 0.123 to 0.072 on the 337M. An ONNX caller that skips it loses most of that back
- **Smaller is not worse here** - 88M/127M/183M pretrains all beat the 337M on CER (see [Size sweep](#size-sweep-pretrain-checkpoints-150k-steps))
- **Tokenizer-free Korean** - Hangul syllables decomposed to jamo at the character level, 127-symbol vocabulary, no phonemizer or G2P stage to maintain
- **Non-autoregressive** - a single 32-step Euler ODE per clause, no classifier-free guidance needed
- **48 kHz output** - the frozen AudioVAE2 decodes 25 Hz latents straight to 48 kHz audio
- **5 distilled voices, uneven quality** - see [Voices](#voices); 2 of 5 are self-rated as below the target bar, shipped anyway for transparency, not hidden
- **Apache-2.0 code** - see [License & Attribution](#license--attribution) for what that does and doesn't cover (pretrain corpus + teacher-model terms are separate from the code license)

> **하이라이트 (한국어)**: 권장 구성은 **183M + 16스텝 + 절 분할**(CER 0.086, 337M 기본값 0.123) / ONNX로 export하면 **CPU 4스레드에서 RTF 0.40~0.54**로 GPU 없이 동작 / 문장이 길수록 정확도가 떨어지므로 **절 분할이 필수**(20자 이하 중앙 CER 0.000, 60자 초과 0.137) / 자모 단위 tokenizer-free 처리 / 48kHz 출력 / 코드는 Apache-2.0이나 학습 데이터·teacher 모델 라이선스는 별도 확인 필요.

---

## Voices

Five target voices were built by using [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (free-form Korean instruct) to design a male conversational voice, locking the best take with `generate_voice_clone`, then using that locked reference to synthesize an 11,554-sentence Korean teacher corpus (`text_data/clean.txt`) which the 337M FreyaTTS student was distilled on (`checkpoints/distill_voice{A..E}`).

**These are synthetic Qwen3-TTS-designed voices, not recordings of a real person** -- there is no human voice-donor to credit or clear rights for on the voice identity itself. The underlying model weights (pretrain backbone) were trained on a licensed Korean speech corpus -- see [License & Attribution](#license--attribution).

| Voice | Description | Self-rated quality | Reference text |
| --- | --- | --- | --- |
| `voiceA_hot1B` | 저음, 차분하고 자연스러운 대화체 톤 (최선) | **ok** -- best of the 5 | "음, 그건 이렇게 하면 되지 않을까요?" |
| `voiceB_cool4` | 중고음, 세련되고 절제된 톤 | **부족 (화질↓)** -- below target quality | "괜찮아요, 제가 도와드릴게요." |
| `voiceC_cool5` | 차분하고 담백한 톤 | **상대적 양호** -- acceptable | "괜찮아요, 제가 도와드릴게요." |
| `voiceD_young3` | 밝고 경쾌한 중저음, 생기 있는 톤 | **보통** -- average | "음, 그건 이렇게 하면 되지 않을까요?" |
| `voiceE_young2` | 밝고 앳된 고음, 클론 일관성 좋음 | **부족 (화질↓, 기계톤)** -- below target, robotic artifacts | "아 진짜요? 완전 신기하네요." |

Reference clips: `confirmed_voices/voice{A..E}_*.wav`. Quality self-ratings are from internal listening review (`confirmed_voices/best_seeds.json`), not a formal MOS study. UTMOSv2 has since been run on all five (`utmos-eval/results/`), and the teacher audio each voice was distilled from was measured too -- that turned out to be the ceiling: teacher UTMOS ranges 3.081 (voiceC) to 3.606 (voiceD), and every student lands 0.22-0.47 below its own teacher. Per-voice WER/CER *has* now been measured ([Distilled voices](#distilled-voices-evalresultsbench_distill_voicejson)), and it disagrees with these ratings: `voiceE`, rated below target, has the best WER of the five. Intelligibility and perceived quality are separate axes here.

> **목소리 (한국어)**: 5개 목소리는 전부 Qwen3-TTS VoiceDesign으로 디자인한 **합성 정체성**이며 실존 인물 녹음이 아닙니다. 잠금된 레퍼런스로 11,554문장 teacher 코퍼스를 합성해 337M FreyaTTS student를 distill했습니다. **품질은 균일하지 않습니다** — A(최선)와 C(양호)는 쓸 만하지만, **B와 E는 자체 평가상 "부족"** 판정입니다 (화질 저하/기계적 톤). 정식 MOS나 UTMOS 등 음질 자동 평가는 아직 이 5개 distill 모델에 대해 돌리지 않았고, 위 표의 판정은 내부 청취 평가입니다. 다만 **목소리별 WER/CER은 측정을 마쳤고, 청취 평가와 어긋납니다** — "부족" 판정인 voiceE가 5개 중 WER이 가장 낮습니다(0.212). 명료도와 체감 음질은 별개 축이라는 뜻이니, WER을 품질 순위로 읽지 마세요.

---

## Quick Start

### Installation

```sh
git clone https://github.com/ummjevel/FreyaTTS.git
cd FreyaTTS
pip install -r requirements.txt
```

Trained checkpoints live under `checkpoints/pretrain` (base, no voice lock) and `checkpoints/distill_voice{A..E}` (voice-locked, see [Voices](#voices)). `from_pretrained("freyavoice/freya-tts", ...)` would load the original *Turkish* weights, which won't produce intelligible Korean -- point every command below at a local Korean checkpoint directory instead.

### Command line

```sh
python infer.py --text "안녕하세요, 어떻게 도와드릴까요?" --model checkpoints/distill_voiceA/final --out output.wav
```

### Batch inference

Non-autoregressive generation batches naturally: one masked ODE solve serves a
whole batch of requests.

```sh
python batch_infer.py --texts texts.txt --model checkpoints/distill_voiceA/final --outdir wavs/ --batch-size 8
```

### Python API

```python
from freyatts import FreyaTTS

tts = FreyaTTS.from_pretrained("checkpoints/distill_voiceA/final", device="cuda")
wav = tts.synthesize("안녕하세요, 어떻게 도와드릴까요?")   # np.float32, 48 kHz
tts.save_wav(wav, "output.wav")
```

`from_pretrained` also accepts a Hugging Face repo id once you publish one. `synthesize` takes optional `steps` (default 32) and `seed` (default `DEFAULT_SEED`) arguments.

**The defaults are not the recommended settings.** They reproduce the original
337M/32-step/`max_words=11` build. For the measured-best configuration, point at a
183M checkpoint and set both knobs:

```python
tts = FreyaTTS.from_pretrained("checkpoints/distill183M_voiceD/final/hf", device="cuda")
tts.max_words = 4                       # clause splitting; see On-device configuration
wav = tts.synthesize(text, steps=16, seed=11)   # seed 11 = voiceD
```

That scores CER 0.086 against 0.123 for the defaults. Each voice has its own
locked seed (`confirmed_voices/best_seeds.json`): A/B/E = 9, C = 1, D = 11.

**The seed selects the speaker.** FreyaTTS conditions only on text — there is no speaker embedding, speaker id, or reference-audio prefix — so the initial flow-matching noise `x0` *is* the voice. SFT/distillation collapses the model onto whichever corpus you fine-tune on, but the model doesn't condition on that speaker's identity, so a different seed gives a different (arbitrary) person, and most seeds won't sound like your target speaker at all. The locked seed per voice is recorded in `confirmed_voices/best_seeds.json`. Use one seed per utterance (the pipeline already shares it across clauses of a long input); passing `seed=None` opts into random-speaker sampling. Long inputs are normalized and split into clauses automatically, then joined with short pauses. Normalization spells out digit runs and clock times in Korean (`9:28` becomes `아홉시 이십팔분`): the duration predictor sizes the utterance from the character sequence, so numbers left as digits come out truncated.

> **사용법 (한국어)**: `checkpoints/distill_voice{A..E}/final`이 실제 학습된 목소리별 체크포인트입니다. **seed가 곧 화자입니다** — 텍스트 조건만 있고 별도 화자 임베딩이 없어서, 초기 노이즈(x0)가 목소리를 결정합니다. 목소리별로 잠긴 seed는 `confirmed_voices/best_seeds.json`에 기록되어 있습니다.

---

## Korean vocabulary

Hangul composes each syllable block (가, 닥, ...) into a single codepoint --
11,172 possible blocks, too many and too sparse to embed directly at
character level. Instead, `freyatts/hangul.py` decomposes every block into
its choseong/jungseong/jongseong jamo, which collapses the vocabulary to 51
unique jamo symbols -- still a pure Unicode-arithmetic mapping, not a
phonemizer or pronunciation dictionary, so the model stays tokenizer-free.

```python
from freyatts.hangul import decompose_hangul, compose_hangul

decompose_hangul("안녕, 오늘 날씨 어때?")
# 'ㅇㅏㄴㄴㅕㅇ, ㅇㅗㄴㅡㄹ ㄴㅏㄹㅆㅣ ㅇㅓㄸㅐ?'
compose_hangul(decompose_hangul("안녕, 오늘 날씨 어때?")) == "안녕, 오늘 날씨 어때?"  # True
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

Latin letters are kept because everyday Korean text mixes in
untransliterated acronyms (AI, TV, PC, ...) constantly --
`freyatts/pipeline.py`'s `BRAND` dict transliterates the common ones ("AI" ->
에이아이) before decomposition; anything not in that dict passes through as
raw Latin letters, which the model was never trained to read, so extend
`BRAND` as you find gaps in your data.

**Known limitation inherited from the Turkish original:** `expand_digits()`
reads any digit run under 6 digits as one spelled-out number, not
digit-by-digit -- fine for prices/quantities, but a hyphenated phone number
like `010-1234-5678` gets read as three arbitrary numbers rather than
digit-by-digit. If your training corpus contains phone numbers or IDs
written with separators, either pre-expand them digit-by-digit in your
manifest before calling `normalize()`, or extend `expand_digits()` with a
phone-number-shaped regex branch.

> **한국어 vocab (한국어)**: 완성형 한글(11,172자)을 그대로 임베딩하기엔 너무 많고 희소해서, `hangul.py`가 초성/중성/종성 자모(51개)로 분해합니다. 발음사전이나 phonemizer가 아니라 순수 유니코드 연산이라 tokenizer-free 원칙은 유지됩니다. 알려진 한계: 6자리 미만 숫자열은 자릿수 그대로 안 읽고 하나의 수로 읽어서, 하이픈 있는 전화번호는 잘못 읽힐 수 있습니다.

---

## Model

FreyaTTS is a conditional flow-matching diffusion transformer (DiT) operating in the latent space of the frozen AudioVAE2:

- **Text encoding:** character-level Korean (jamo-decomposed), 127 symbols shipped with the package (`freyatts/char_vocab.json`)
- **Generation:** non-autoregressive flow matching, 32 Euler ODE steps, no CFG
- **Latent space:** 64-dim latents at 25 Hz; AudioVAE2 encodes at 16 kHz and decodes at 48 kHz
- **Parameters:** 337M (`d_model=768, depth=22, heads=12, ff=2048`) for the shipped voices; 88M/127M/183M pretrain-only variants also exist and score better on WER (see [Size sweep](#size-sweep-pretrain-checkpoints-150k-steps) and the parameter-count correction above)
- **Training:** from scratch on a licensed Korean speech corpus (pretraining, `checkpoints/pretrain`), then distilled per-voice onto Qwen3-TTS-synthesized data (`checkpoints/distill_voice{A..E}`)

AudioVAE2 is not retrained. It is downloaded from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0) at load time via the `voxcpm` package.

---

## Evaluation

All runs use the same 300 held-out Korean dev sentences (`eval/eval_ko_dev.jsonl`), Whisper `language="ko"` for WER/CER (`eval/benchmark.py`), on an NVIDIA H100 80GB (`eval/speed.py`). Every number below is a **single run with the default seed** -- no seed-variance or confidence interval has been measured, so treat differences of ~0.01 WER as noise.

### Accuracy of the 337M pretrain over training (`eval/results/bench_step*.json`)

| Checkpoint | WER | CER | RTF |
| --- | --- | --- | --- |
| step200000 | 0.312 | 0.173 | 0.20 |
| step250000 | 0.320 | 0.187 | 0.21 |
| final (= step250000 weights) | 0.319 | 0.194 | 0.17 |

WER plateaus around 0.31-0.32 from step 200k onward -- further pretraining past 200k did not meaningfully improve accuracy.

### Size sweep (pretrain checkpoints, 150k steps)

Three smaller configs were pretrained from scratch on the same corpus, each for 150k steps, and benchmarked identically:

| Model | `d_model` / depth / heads / ff | Params | Steps | WER | CER | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| 88M | 512 / 12 / 8 / 1536 | 87,510,081 | 150k | 0.285 | 0.167 | **0.121** |
| 127M | 512 / 16 / 8 / 2048 | 127,374,401 | 150k | 0.309 | 0.185 | 0.143 |
| **183M** | 640 / 16 / 10 / 2048 | 183,220,545 | 150k | **0.262** | **0.153** | 0.149 |
| 337M (`pretrain/final`) | 768 / 22 / 12 / 2048 | 337,182,785 | 250k | 0.319 | 0.194 | 0.171 |

Two things stand out, and neither has been chased down yet:

- **All three smaller models beat the 337M on WER**, and the 183M does so by a wide margin (0.262 vs 0.319) with fewer training steps and 1.15x faster RTF. On a 560k-utterance corpus the 337M config appears to be past the useful size, not undertrained -- its own WER curve was already flat from step 200k. Note the comparison is not step-matched (337M `final` is 250k steps); the 337M's step-200k benchmark, 0.312, is still worse than the 183M's.
- **127M scores worse than 88M**, which is a size/accuracy inversion that a single run cannot explain. Until it is re-run with different seeds, treat the 127M row as unreliable rather than as evidence about width-vs-depth.

None of the smaller models has a `speed_*.json` yet, so their TTFT and peak VRAM are unmeasured -- the RTF column above comes from the benchmark run. None has been distilled onto a voice.

### Distilled voices (`eval/results/bench_distill_voice*.json`)

Each voice-locked 337M checkpoint, same dev set and protocol:

| Voice | WER | CER | RTF | Listening self-rating |
| --- | --- | --- | --- | --- |
| `voiceA_hot1B` | 0.224 | 0.123 | 0.48 | ok -- best of the 5 |
| `voiceB_cool4` | 0.243 | 0.135 | 0.39 | below target |
| `voiceC_cool5` | 0.221 | 0.118 | 0.32 | acceptable |
| `voiceD_young3` | 0.237 | 0.132 | 0.21 | average |
| `voiceE_young2` | **0.212** | **0.114** | 0.33 | below target (robotic) |

Distillation improves intelligibility across the board -- every voice lands well under the 0.319 WER of the 337M pretrain it started from, which is expected since each is fit to a single synthetic speaker.

The RTF column is **not** a property of the voices: all five are the identical 337M architecture, so the 0.21-0.48 spread is measurement noise from whatever else shared the GPU during each run, not a speed difference between voices. Use `eval/results/speed_final.json` for speed, not this column.

**WER does not track the listening ratings here.** `voiceE` scores the *best* WER of the five while being one of the two rated below target for robotic artifacts, and `voiceB` is mid-pack on WER despite the same rating. WER measures whether Whisper can read the audio back, not whether it sounds like a person -- so these numbers should not be read as a quality ranking. **No UTMOS or MOS study has been run**, which is exactly the gap that would settle it.

### On-device configuration

The shipped voices were built at 337M with 32 ODE steps and `max_words=11`. That
combination is neither the most accurate nor remotely deployable. Measuring size,
step count and clause splitting as a grid (`eval/results/ondevice_grid.jsonl`)
gives a better default.

**183M / 16 steps / `max_words=4`**, all five voices, 300-sentence dev set:

| Voice | CER | UTMOS | Rhythm CV |
| --- | --- | --- | --- |
| voiceD | **0.0860** | **3.063** | 0.523 |
| voiceE | 0.0897 | 2.765 | 0.518 |
| voiceA | 0.0920 | 2.786 | 0.499 |
| voiceC | 0.0981 | 2.670 | 0.462 |
| voiceB | 0.1106 | 2.639 | 0.491 |

The 88M at the same settings runs 0.125-0.159, worse on every voice, so 183M is
the floor worth shipping. For reference the 337M default scores 0.1233 on voiceA
against this configuration's 0.0920 -- **half the parameters, better accuracy**.

**Clause splitting carries much of that.** Accuracy is length-dependent:

| Input length | Median CER (337M) |
| --- | --- |
| <= 20 chars | **0.0000** |
| 21-40 | 0.0526 |
| 41-60 | 0.1053 |
| > 60 | 0.1374 |

Tightening `max_words` from 11 to 4 takes the 337M from 0.1233 to 0.0721 and
collapses the length correlation (+0.336 to +0.071). The model is accurate on
short inputs and loses content on long ones -- repeats and drops, not
mispronunciations -- because the duration head predicts one total frame count for
the whole clause from a mean-pooled text embedding.

**Rhythm CV is reported because step and size reductions must not flatten
delivery.** It is the coefficient of variation of syllable onset intervals: the
teacher corpus measures 0.545, real human speech 0.450, and this model holds
0.46-0.55 across every size and step count tested. That variety is what
listeners identify as natural delivery, and it survives shrinking.

### CPU inference (ONNX)

`eval/export_onnx.py` writes three graphs -- duration, DiT (ODE loop unrolled),
and the AudioVAE2 decoder -- verified against PyTorch to `max_abs_diff 4.5e-05`
on an identical initial noise, so the voice is unchanged.

| Config | ONNX size | RTF @ 1 thread | RTF @ 4 threads |
| --- | ---: | ---: | ---: |
| 337M / 32 steps | 1.6 GB | 3.26-4.02 | — |
| 183M / 16 steps | 918 MB | 1.32-1.47 | **0.54-0.65** |
| 183M / 8 steps | 909 MB | **0.90-0.95** | **0.40-0.45** |

Step count barely moves the file size -- the ODE loop is unrolled but the weights
are stored once as initializers -- so it is purely a speed knob. Eight steps
clears real time even single-threaded.

**Porting the model is not enough.** The exported graphs are the network only;
normalization and clause splitting live in `freyatts/pipeline.py`. An ONNX caller
that skips them scored CER 0.1317 where the same weights driven through the
python pipeline scored 0.1016 on the same sentences. `eval/bench_onnx_cpu.py`
reproduces both. Splitting costs a little speed -- more, shorter forward passes
per utterance -- and moves 4-thread RTF from 0.54-0.65 to 0.59-0.66.

The port is close to lossless but not exactly lossless. On a matched 150-sentence
subset, PyTorch scores CER 0.0945 and the ONNX path 0.1016. Both run the same
weights, steps, seed and split, so the 0.007 is the runtime, not the recipe --
small enough to deploy on, large enough not to call the two interchangeable.
(The 0.0860 headline figure is over all 300 sentences and is not comparable to
either.)

fp16 and int8 conversion were attempted and are **not** currently usable: the
unrolled 16-step graph defeats both `onnxconverter-common` (Einsum and Cast
operands end up with mismatched types) and `onnxruntime.quantization`
(`MatMulInteger` shape error), and dynamic int8 only reached 1.21x on size
anyway because most DiT parameters sit in ops it does not touch.

> **온디바이스 구성 (한국어)**: 권장은 **183M / 16스텝 / `max_words=4`** — voiceD 기준 CER 0.0860으로,
> 337M 기본 구성(0.1233)보다 정확하면서 파라미터는 절반입니다. **절 분할이 핵심**입니다:
> 이 모델은 짧은 입력에서 정확하고(20자 이하 중앙 CER 0.000) 길어지면 내용을 빠뜨리거나 반복합니다
> (60자 초과 0.137). 발음을 틀리는 게 아니라 길이 예측이 문장 전체를 평균 하나로 뭉개서 생기는 문제입니다.
> ONNX로 CPU 4스레드 RTF 0.54(16스텝)·0.40(8스텝)이라 GPU가 필요 없지만, **정규화와 분할을 호출부에
> 반드시 같이 구현**해야 합니다(빠뜨리면 CER 0.1016 → 0.1317). 같은 150문장 기준 PyTorch 0.0945 /
> ONNX 0.1016으로 런타임 차이가 0.007 남아 있습니다 — 배포에 무리는 없지만 동일하다고 할 수는 없습니다.
> fp16·int8 변환은 현재 표준 도구로 실패하며, 언롤된 그래프의 타입 처리 문제입니다.
>
> **스텝 수를 더 줄이려면 그냥 8스텝으로 돌리세요.** ZipVoice식 reflow distillation을 시도했지만
> 학생이 모든 스텝에서 teacher보다 나빴습니다(아래 참고). distill 없이 8스텝은 CER 0.0987로,
> 16스텝 대비 +0.013에 속도 2배이고 리듬(0.526)도 그대로입니다.

### Speed (`eval/results/speed_final.json`, H100 80GB, 337M params)

| Bucket | Latency | TTFT | RTF |
| --- | --- | --- | --- |
| short | 1.35 s | 0.68 s | 0.58 |
| medium | 1.42 s | 0.71 s | 0.34 |
| long | 2.41 s | 0.77 s | 0.20 |

Peak VRAM 2.07 GB, load time 11.3 s. **These are H100 datacenter numbers, not on-device numbers** -- no Apple Neural Engine / CPU / mobile benchmark has been run for this Korean checkpoint. Treat any such number from the upstream Turkish paper as inapplicable here until re-measured on this checkpoint and target hardware.

> **평가 (한국어)**: 전부 동일 조건(300문장 held-out dev set, Whisper ko, H100) 실측치이며, **각 설정당 1회 단일 시드 측정**이라 0.01 내외 차이는 노이즈로 보셔야 합니다. 요약하면 — (1) 소형 3종(88M/127M/183M)이 **전부 337M보다 WER이 낮고**, 특히 183M은 0.262 vs 0.319로 차이가 큽니다. 이 코퍼스(56만 발화)에서 337M은 학습 부족이 아니라 과대 설정으로 보입니다(337M 자체 WER 곡선도 200k부터 평평). (2) **127M이 88M보다 나쁜 역전**은 단일 실행으로 설명이 안 되므로, 시드를 바꿔 재실행하기 전까지 127M 행은 신뢰하지 마세요. (3) distill 5종은 전부 pretrain보다 WER이 좋아졌지만, **WER 순위와 청취 평가가 어긋납니다** — voiceE는 WER 최상(0.212)인데 청취 평가는 "부족(기계톤)"입니다. WER은 Whisper가 알아듣는지를 볼 뿐 사람처럼 들리는지를 재지 않으므로 품질 순위로 읽으면 안 되고, 이를 가릴 **UTMOS/MOS는 아직 미측정**입니다. 소형 모델은 `speed_*.json`이 없어 TTFT·VRAM 미측정이고 distill도 아직 없습니다. 속도는 H100 서버 기준이며 on-device(모바일/CPU/Apple Neural Engine) 측정은 없습니다 — 터키어 원논문의 CPU/모바일 수치를 이 한국어 체크포인트에 그대로 적용하면 안 됩니다.

---

## Training

The training pipeline itself needs no changes for Korean -- it already reads
`vocab_json` from config and applies no language-specific logic.

### 0. Pretrain corpus

The `checkpoints/pretrain` backbone was trained on **AI Hub dataset #133,
"감성 및 발화 스타일 동시 고려 음성합성 데이터"** (Korean speech-synthesis corpus with
simultaneous emotion/speaking-style annotation), licensed for both commercial
and non-commercial AI model development per AI Hub's usage policy. **AI
Hub's usage policy requires attribution in derivative works**: any model
trained on this data must state that it is a result of a project by the
National Information Society Agency (한국지능정보사회진흥원). See [License &
Attribution](#license--attribution).

If you retrain on a different corpus, re-check its license -- AI Hub license
terms vary per dataset, and some require a separate commercial-use
agreement. For a general-purpose Korean single-speaker TTS corpus:

| Corpus | License | Notes |
| --- | --- | --- |
| AI Hub #133 (감성 및 발화 스타일 동시 고려 음성합성 데이터) | ✅ commercial + non-commercial, attribution required | Used for this repo's `checkpoints/pretrain` |
| KSS (Korean Single Speaker Speech) | ❌ CC BY-NC-SA 4.0 | The most commonly used Korean TTS corpus in tutorials/papers -- **avoid for anything commercial** |
| [Zeroth-Korean](https://openslr.org/40/) | ✅ CC BY 4.0 | 51.6h, 105 speakers (ASR-oriented, not single-speaker) |
| [Common Voice Korean](https://commonvoice.mozilla.org/) | ✅ CC0 | Crowd-sourced, quality/equipment varies |

### 1. Build the manifest

```sh
python training/build_manifest_ko.py \
    --metadata /path/to/corpus/metadata.csv \
    --wav-dir /path/to/corpus/wavs \
    --out data/manifest.jsonl
```

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

### 3. Distill onto a Qwen3-TTS-synthesized voice

```sh
# design + lock a voice with Qwen3-TTS VoiceDesign, then batch-clone the teacher corpus
python text_data/synth_teacher.py --voice-ref confirmed_voices/voiceA_hot1B.wav --ref-text "..." \
    --text-file text_data/clean.txt --out text_data/teacher/voiceA_manifest.jsonl

# normalize (jamo) + precompute latents on the synthesized (audio, text) pairs
python training/precompute_latents.py --manifest data/manifest_distill_voiceA.jsonl --output-dir data/latents_distill_voiceA

# SFT the pretrain checkpoint onto the per-voice synthetic corpus
accelerate launch training/sft.py --config training/configs/sft_stage1.yaml \
    --init checkpoints/pretrain/step200000/model.pt --data data/latents_distill_voiceA --out checkpoints/distill_voiceA
```

A short (~3k step) SFT transfers the voice but inherits the pretrain
checkpoint's pronunciation/audio quality as-is; a longer continued-pretrain
(~20k steps, `lr=1.5e-4`, initialized from the base pretrain) measurably
improves pronunciation and audio quality before the per-voice SFT, confirmed
by internal listening review on 3 of the 5 voices.

> **학습 (한국어)**: `checkpoints/pretrain` backbone은 **AI Hub #133 "감성 및 발화 스타일 동시 고려 음성합성 데이터"**로 학습했습니다. 이 데이터셋은 영리·비영리 모두 이용 가능하지만 **파생물에 출처 표시 의무**가 있습니다 (아래 [License & Attribution](#license--attribution) 참고). 5개 목소리는 이 backbone을 Qwen3-TTS로 합성한 목소리별 데이터로 distill(SFT)한 것입니다. 3k step 짧은 SFT는 목소리만 전이되고 pretrain 품질을 그대로 물려받으며, ~20k step 긴 continued-pretrain을 먼저 거치면 발음/음질이 뚜렷이 개선됩니다 (5개 중 3개에서 내부 청취로 확인).

---

## License & Attribution

**Code**: FreyaTTS code (this fork included) is Apache-2.0 ([LICENSE](LICENSE)).

**AudioVAE2** (frozen, not retrained): Apache-2.0, from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2).

**Qwen3-TTS** (teacher model used to synthesize the per-voice training data): Apache-2.0, from [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

**Pretrain corpus**: AI Hub dataset #133, "감성 및 발화 스타일 동시 고려 음성합성 데이터," built under a project of Korea's National Information Society Agency (한국지능정보사회진흥원, NIA). Per AI Hub's usage policy this data (and models built from it) may be used for commercial and non-commercial AI development, provided derivative works state they are a result of an NIA project. Accordingly:

> 본 모델은 한국지능정보사회진흥원의 「지능정보산업 인프라 조성」 사업으로 구축된 AI 허브 데이터셋 "133.감성 및 발화 스타일 동시 고려 음성합성 데이터"를 활용하여 학습되었습니다.
> (This model was trained using AI Hub dataset "133. Speech Synthesis Data Considering Emotion and Speaking Style," built under the National Information Society Agency's "Intelligent Information Industry Infrastructure Development" project.)

**Target voices**: the 5 released voices (`voiceA`-`voiceE`) are synthetic identities designed with Qwen3-TTS VoiceDesign, not recordings or clones of a specific real person -- there is no third-party voice-donor whose consent applies to the voice identity itself.

**Bottom line for downstream users**: redistributing the *code* and the *model weights* is unrestricted (Apache-2.0 + AI Hub's stated commercial-use terms), as long as the AI Hub attribution notice above is kept in any derivative work, including this repository and anything built from these checkpoints. AI Hub's raw source data (`.wav`/`.json` files) itself must **not** be redistributed -- only the resulting trained weights.

> **라이선스 및 출처 표시 (한국어)**: 코드는 Apache-2.0, AudioVAE2도 Apache-2.0, teacher로 쓴 Qwen3-TTS도 Apache-2.0입니다. Pretrain corpus는 AI Hub #133 "감성 및 발화 스타일 동시 고려 음성합성 데이터"(한국지능정보사회진흥원 사업 결과물)이며, 영리·비영리 모두 이용 가능하나 **2차적 저작물에 위 출처 표시 문구를 반드시 포함**해야 합니다. 5개 목소리는 Qwen3-TTS로 디자인한 합성 정체성으로 실존 인물이 아니므로 별도의 화자 동의 문제는 없습니다. **AI Hub 원본 데이터(wav/json 파일 자체)는 재배포 금지** — 재배포 가능한 건 그걸로 학습된 모델 가중치뿐입니다.

## Acknowledgments

- [freyavoiceai/FreyaTTS](https://github.com/freyavoiceai/FreyaTTS) for the original Turkish model this fork retargets
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM) (Apache-2.0) for the AudioVAE2 that FreyaTTS generates into
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0) for the teacher model used to synthesize per-voice training data
- AI Hub / 한국지능정보사회진흥원(NIA) for the pretrain corpus (#133)
- The flow matching and DiT literature this model builds on

## Citation

FreyaTTS's architecture is described in the upstream technical report, [arXiv:2607.09530](https://arxiv.org/abs/2607.09530) (note the parameter-count correction above -- this repo's models are 337M, not the paper's 183.2M):

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
