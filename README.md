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

**Correction vs. the upstream paper's parameter count**: the upstream FreyaTTS technical report states 183.2M parameters, but `training/configs/pretrain.yaml`'s actual dims (`d_model=768, depth=22, ff=2048`) build a **337M**-parameter model (confirmed at load time: `eval/results/speed_final.json` reports `"params": 337182785`). Every checkpoint in this repo (`checkpoints/pretrain`, `checkpoints/distill_voice{A..E}`) is this 337M config, not 183M. A smaller re-pretrain at the paper's actual 183M dims (`d_model=640, depth=16, ff=2048`) is in progress (`checkpoints/pretrain_183M/`, plus 88M and 127M variants) but has **no trained checkpoints yet** as of this writing.

> **파라미터 수 정정 (한국어)**: 원 논문은 183.2M이라고 표기하지만, 실제 `pretrain.yaml` 설정(d_model=768, depth=22, ff=2048)으로 빌드되는 모델은 **337M**입니다 (`eval/results/speed_final.json`의 `"params": 337182785`로 실측 확인). 이 저장소의 모든 체크포인트(`checkpoints/pretrain`, `checkpoints/distill_voice{A..E}`)는 337M 설정입니다. 논문 스펙 그대로인 183M(d_model=640, depth=16) 재학습은 진행 중이지만 **아직 학습된 체크포인트가 하나도 없습니다**.

It is tokenizer-free at the character (jamo) level -- no phonemizer, no G2P -- and generates speech with a **non-autoregressive conditional flow-matching DiT** in the frozen AudioVAE2 latent space (25 Hz, 64-dim latents, 16 kHz encode / 48 kHz decode).

### Highlights

- **337M parameters** (not the 183.2M of the upstream paper -- see correction above), measured RTF 0.17-0.58 depending on utterance length on an H100 (see [Evaluation](#evaluation)) -- no Apple/CPU numbers have been measured for this Korean checkpoint yet
- **Tokenizer-free Korean** - Hangul syllables decomposed to jamo at the character level, 127-symbol vocabulary, no phonemizer or G2P stage to maintain
- **Non-autoregressive** - a single 32-step Euler ODE per clause, no classifier-free guidance needed
- **48 kHz output** - the frozen AudioVAE2 decodes 25 Hz latents straight to 48 kHz audio
- **5 distilled voices, uneven quality** - see [Voices](#voices); 2 of 5 are self-rated as below the target bar, shipped anyway for transparency, not hidden
- **Apache-2.0 code** - see [License & Attribution](#license--attribution) for what that does and doesn't cover (pretrain corpus + teacher-model terms are separate from the code license)

> **하이라이트 (한국어)**: 실측 337M 파라미터 (논문 183.2M 아님) / 자모 단위 tokenizer-free 한국어 처리 / non-autoregressive 32-step ODE / 48kHz 출력 / 5개 distill 목소리 중 2개는 자체 평가상 기준 미달이지만 투명하게 그대로 공개 / 코드 자체는 Apache-2.0이나 학습 데이터·teacher 모델 라이선스는 별도 확인 필요.

---

## Voices

Five target voices were built by using [Qwen3-TTS VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (free-form Korean instruct) to design a male conversational voice, locking the best take with `generate_voice_clone`, then using that locked reference to synthesize an 11,554-sentence Korean teacher corpus (`text_data/clean.txt`) which the 337M FreyaTTS student was distilled on (`checkpoints/distill_voice{A..E}`).

**These are synthetic Qwen3-TTS-designed voices, not recordings of a real person** -- there is no human voice-donor to credit or clear rights for on the voice identity itself. The underlying model weights (pretrain backbone) were trained on a licensed Korean speech corpus -- see [License & Attribution](#license--attribution).

| Voice | Description | Self-rated quality | Reference text |
| --- | --- | --- | --- |
| `voiceA_hot1B` | 쿨 hot guy, 중저음, 자연스러운 대화체 (최선) | **ok** -- best of the 5 | "음, 그건 이렇게 하면 되지 않을까요?" |
| `voiceB_cool4` | 도시세련, 쿨하지만 차갑지 않음 | **부족 (화질↓)** -- below target quality | "괜찮아요, 제가 도와드릴게요." |
| `voiceC_cool5` | 절제따뜻, 담백한 쿨 | **상대적 양호** -- acceptable | "괜찮아요, 제가 도와드릴게요." |
| `voiceD_young3` | 발랄싱그 풋풋 연하남 (중저음, 어리고 싱그러운) | **보통** -- average | "음, 그건 이렇게 하면 되지 않을까요?" |
| `voiceE_young2` | 앳된 수줍은 연하남 (풋풋, 클론 일관성 좋음) | **부족 (화질↓, 기계톤)** -- below target, robotic artifacts | "아 진짜요? 완전 신기하네요." |

Reference clips: `confirmed_voices/voice{A..E}_*.wav`. Quality self-ratings are from internal listening review (`confirmed_voices/best_seeds.json`), not a formal MOS study -- no third-party or automated (e.g. UTMOS) score has been run on the distilled voices yet.

> **목소리 (한국어)**: 5개 목소리는 전부 Qwen3-TTS VoiceDesign으로 디자인한 **합성 정체성**이며 실존 인물 녹음이 아닙니다. 잠금된 레퍼런스로 11,554문장 teacher 코퍼스를 합성해 337M FreyaTTS student를 distill했습니다. **품질은 균일하지 않습니다** — A(최선)와 C(양호)는 쓸 만하지만, **B와 E는 자체 평가상 "부족"** 판정입니다 (화질 저하/기계적 톤). 정식 MOS나 UTMOS 등 자동 평가는 아직 이 5개 distill 모델에 대해 돌리지 않았고, 내부 청취 평가만 반영된 수치입니다.

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
- **Parameters:** 337M (`d_model=768, depth=22, heads=12, ff=2048`) -- see the parameter-count correction above
- **Training:** from scratch on a licensed Korean speech corpus (pretraining, `checkpoints/pretrain`), then distilled per-voice onto Qwen3-TTS-synthesized data (`checkpoints/distill_voice{A..E}`)

AudioVAE2 is not retrained. It is downloaded from [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) (Apache-2.0) at load time via the `voxcpm` package.

---

## Evaluation

Measured on the base pretrain checkpoint (`checkpoints/pretrain`), 300 held-out Korean dev sentences (`eval/eval_ko_dev.jsonl`), Whisper `language="ko"` for WER/CER (`eval/benchmark.py`), on an NVIDIA H100 80GB (`eval/speed.py`). **These numbers are for the pretrain checkpoint before per-voice distillation -- there is no separate WER/CER/UTMOS run yet for `distill_voice{A..E}` individually**; the [Voices](#voices) table above gives self-rated listening quality instead.

### Accuracy (WER/CER, `eval/results/bench_*.json`)

| Checkpoint | WER | CER | RTF |
| --- | --- | --- | --- |
| step200000 | 0.312 | 0.173 | 0.20 |
| step250000 | 0.320 | 0.187 | 0.21 |
| final (= step250000 weights) | 0.319 | 0.194 | 0.17 |

WER plateaus around 0.31-0.32 from step 200k onward -- further pretraining past 200k did not meaningfully improve accuracy (`eval/results/bench_step*.json`).

### Speed (`eval/results/speed_final.json`, H100 80GB, 337M params)

| Bucket | Latency | TTFT | RTF |
| --- | --- | --- | --- |
| short | 1.35 s | 0.68 s | 0.58 |
| medium | 1.42 s | 0.71 s | 0.34 |
| long | 2.41 s | 0.77 s | 0.20 |

Peak VRAM 2.07 GB, load time 11.3 s. **These are H100 datacenter numbers, not on-device numbers** -- no Apple Neural Engine / CPU / mobile benchmark has been run for this Korean checkpoint. Treat any such number from the upstream Turkish paper as inapplicable here until re-measured on this checkpoint and target hardware.

> **평가 (한국어)**: 위 WER/CER/RTF는 **distill 이전의 base pretrain 체크포인트** 기준 실측치입니다 (300문장 held-out dev set, Whisper 기반). 5개 distill 목소리 각각에 대한 정식 WER/CER/UTMOS 측정은 아직 하지 않았고, [Voices](#voices) 표의 청취 평가만 있습니다. 속도는 H100 서버 기준이며, on-device(모바일/CPU/Apple Neural Engine) 측정은 아직 없습니다 — 터키어 원논문의 CPU/모바일 수치를 이 한국어 체크포인트에 그대로 적용하면 안 됩니다.

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
