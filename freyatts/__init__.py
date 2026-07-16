"""FreyaTTS (Korean fork): a 183M-parameter non-autoregressive Korean text-to-speech model."""

from .model import FreyaDiT
from .pipeline import FreyaTTS

__all__ = ["FreyaTTS", "FreyaDiT"]
