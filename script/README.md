# FreyaTTS pretraining on AIHub 동시고려 (133) — sbatch scripts

전체 파이프라인 (manifest → latents → pretrain). 모두 slurm `batch` 파티션, 단일 노드 `gpu-1`.

이 클러스터는 slurm이 GPU를 gres로 관리하지 **않습니다**. 물리 GPU를
`CUDA_VISIBLE_DEVICES`로 직접 지정합니다. **사용 가능한 GPU: 4,5,6,7**.

환경: `/data/users/voice/zoey/FreyaTTS/.venv` (Python 3.11, torch 2.11+cu128,
accelerate, voxcpm). 각 sbatch가 자동으로 activate 합니다.

## 실행 순서 (repo 루트에서 제출)

```bash
cd /data/users/voice/zoey/FreyaTTS

# 1) manifest (CPU, GPU 불필요) — data/manifest_{train,dev}.jsonl
sbatch script/01_build_manifest.sbatch

# 2) latents (GPU 4장 병렬, job array 0-3 → GPU 4,5,6,7) — data/latents/*.pt
sbatch script/02_precompute_latents.sbatch

# 3) pretrain (GPU 여러 장, Accelerate bf16) — checkpoints/pretrain/
GPUS=4,5,6,7 sbatch script/03_pretrain.sbatch
#   또는:  sbatch script/03_pretrain.sbatch 6,7      (특정 GPU만)
#   또는:  sbatch script/03_pretrain.sbatch 6        (단일 GPU)
```

### 의존성 체인으로 한 번에 제출 (권장)
```bash
J1=$(sbatch --parsable script/01_build_manifest.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 script/02_precompute_latents.sbatch)
sbatch        --dependency=afterok:$J2 script/03_pretrain.sbatch
```
`02`는 4-task job array라서 `afterok:$J2` 는 **4개 task 모두 성공** 시 `03`을 시작합니다.
`01`이 train manifest를 round-robin 4등분(`manifest_train.part{0..3}.jsonl`)하고,
각 array task가 자기 part를 `--prefix part<N>` 로 같은 `data/latents/`에 인코딩합니다.

## GPU 지정 방법 (공통)

`GPUS`/`GPUID` 환경변수 또는 첫 번째 인자로 물리 GPU id를 넘깁니다.
**사용 가능한 GPU: 4,5,6,7** (기본값도 이 범위). 예) 6,7번만 쓰려면
`GPUS=6,7 sbatch script/03_pretrain.sbatch`.

## 자주 쓰는 옵션 (환경변수 override)

- `03_pretrain`: `RESUME=checkpoints/pretrain/step30000 GPUS=... sbatch ...` 로 재개
  (10k 스텝마다 체크포인트 저장). `DATA`, `OUT`, `CONFIG` 도 override 가능.
- `02_precompute_latents`: `MANIFEST=data/manifest_dev.jsonl OUT=data/latents_dev sbatch ...`
  로 dev 셋 인코딩.

## 평가 (pretrain/SFT 체크포인트)

FreyaTTS는 화자 조건이 없고 **seed가 곧 목소리**입니다. 따라서 **pretrain 단계**
체크포인트는 목표 목소리 재현이 아니라 **지능도(WER/CER) + 속도(RTF/VRAM)** 로
평가합니다(목소리 고정은 SFT 단계).

`04_eval.sbatch` 가 3단계를 한 번에 수행합니다: `model.pt` →
`config.json`+`model.safetensors` 변환(`training/convert_ckpt.py`) → WER/CER
(Whisper large-v3, ko, `eval/benchmark.py`) → 속도(`eval/speed.py`).

```bash
# 특정 스텝 체크포인트 평가 (GPU 4)
CKPT=checkpoints/pretrain/step30000/model.pt GPUID=4 sbatch script/04_eval.sbatch
#   또는:  sbatch script/04_eval.sbatch checkpoints/pretrain/final/model.pt 5
# 빠른 확인만:  LIMIT=100 CKPT=... sbatch script/04_eval.sbatch
```

- 평가셋: `eval/eval_ko_dev.jsonl` (dev split 원문에서 300문장 샘플, `{"text":...}`)
- 결과: `eval/results/bench_<tag>.json`, `eval/results/speed_<tag>.json`
- 첫 실행 시 faster-whisper large-v3(~3GB)를 HF 캐시로 내려받습니다.
- venv에 `faster-whisper`, `jiwer` 설치 완료.

## SFT (voice lock + 짧은 발화) — pretrain 이후

pretrain은 화자 임의(다화자)라 프로덕션 목소리가 없습니다. SFT가 **단일 성우로
목소리를 고정**하고 짧은 발화 커버리지를 채웁니다. 순서:

파일명 필드1 = 발화 스타일: **D=대화체** M=독백체 F=구연체 A=애니체 K=친절체
S=중계체 N=낭독체. 필드4 = 화자ID. 대화체 최다 화자는 **010(5,394발화≈8h)**, 005, 063.

```bash
# 05) 단일 스타일·성우 데이터 구축 (manifest 필터 → latents)
STYLE=D SPEAKER=010 GPUID=4 sbatch script/05_build_sft_data.sbatch   # 대화체·화자010 → data/latents_sft
#   STYLE="" 로 두면 스타일 무관(그 화자 전체). 기본값 STYLE=D SPEAKER=010.

# 06) SFT stage 1 (voice lock): pretrain/final에서 fine-tune, 3000 step
GPUS=4,5,6,7 sbatch script/06_sft_stage1.sbatch                # → checkpoints/sft_stage1
#   500 step마다 저장 → 이후 seed 몇 개 합성해 목표 목소리 잡힌 체크포인트 선택

# 07) 짧은 발화 추출 (05가 만든 data/manifest_sft.jsonl 사용) + stage1 latents와 섞기
GPUID=4 sbatch script/07_extract_short.sbatch                 # → data/latents_sft_short

# 08) SFT stage 2 (짧은 발화 커버리지): stage1/final에서 이어서
GPUS=4,5,6,7 sbatch script/08_sft_stage2.sbatch                # → checkpoints/sft_stage2
```

- **스타일·화자 선택**: `05`가 STYLE(필드1)·SPEAKER(필드4)로 필터. 기본은
  대화체(D)·화자010. 다른 조합은 `STYLE=M SPEAKER=005 ...` 처럼 지정.
- SFT는 짧은 stage(각 3000 step)라 `save_every=500`로 잡아 중간 체크포인트에서
  최적점(~step 1000)을 고릅니다. pretrain과 동일하게 NCCL P2P 우회·NaN 가드 적용.
- **production seed 고정**: stage1 후 `freyatts/model.py`의 `DEFAULT_SEED`를 목표
  목소리가 잡힌 seed로 갱신 (README 본문 참고).
- 최종 평가: `CKPT=checkpoints/sft_stage2/final/model.pt GPUID=4 sbatch script/04_eval.sbatch`

## pretrain 스텝 연장 (선택)

기본 150000 step. 더 돌리려면 `STEPS`로 cosine LR 스케줄을 늘려서 재개합니다
(그냥 resume하면 LR이 바닥 2.5e-5 고정이라 효과 미미):
```bash
STEPS=250000 RESUME=checkpoints/pretrain/final GPUS=4,5,6,7 sbatch script/03_pretrain.sbatch
```
단, cfm loss는 이미 diminishing-returns 구간이라 이득은 작습니다 — 품질은 SFT가 좌우.

## 하이퍼파라미터

`training/configs/pretrain.yaml` — 183M DiT, 150k steps, per-device batch 64,
lr 5e-4 (warmup 2k → cosine), max 20s(500 frame) 클립. 수정은 그 파일에서.

## 로그

`script/logs/<job>_<jobid>.{out,err}`
