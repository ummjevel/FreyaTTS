# Vendored from openbmb/VoxCPM (github.com/OpenBMB/VoxCPM), Apache-2.0.
#
# FreyaTTS only needs this one self-contained module (AudioVAEV2) out of the
# full `voxcpm` pip package, which additionally pulls in funasr, gradio,
# modelscope, and a librosa/numba/llvmlite chain that has repeatedly failed
# to build on newer Python releases -- none of it needed for decoding
# latents to waveforms. Vendoring these ~960 lines (which only depend on
# numpy/torch/pydantic) removes that entire dependency chain.
#
# Weights are still downloaded at runtime from openbmb/VoxCPM2 on the Hub
# (see freyatts/vae.py) -- only the model *code* is vendored, not the
# weights.
from .audio_vae import AudioVAE, AudioVAEConfig
from .audio_vae_v2 import AudioVAE as AudioVAEV2, AudioVAEConfig as AudioVAEConfigV2
