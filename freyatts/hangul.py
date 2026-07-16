"""Hangul syllable <-> jamo decomposition, pure Unicode arithmetic (no deps).

FreyaTTS is tokenizer-free: it wants a small, fixed character vocabulary and
no phonemizer/G2P stage. Turkish gets this for free (Latin script, ~90
symbols). Korean orthography composes each syllable block from 2-3 jamo into
a single codepoint (11,172 possible blocks in the U+AC00-U+D7A3 range) -- too
many, and too sparse per symbol, to embed directly. Decomposing every block
into its choseong/jungseong/jongseong parts (all already single codepoints in
the Hangul Compatibility Jamo block, U+3131-U+318E) collapses the vocabulary
back down to ~51 symbols while staying phonemizer-free: the mapping is a pure
Unicode formula, not a pronunciation dictionary.

Composition is the inverse and isn't needed by training or inference (the
model never emits text), but is kept for round-trip testing/debugging.
"""

HANGUL_BASE = 0xAC00
HANGUL_LAST = 0xD7A3

# 19 choseong (initial consonants)
CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")

# 21 jungseong (vowels)
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")

# 28 jongseong (final consonants); index 0 is "no final"
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

assert len(CHOSEONG) == 19
assert len(JUNGSEONG) == 21
assert len(JONGSEONG) == 28

# jongseong-only compounds -- simple finals reuse a choseong codepoint, so the
# *unique* symbol set contributed by jongseong is just these 11 compounds.
_JONGSEONG_ONLY = [j for j in JONGSEONG if j and j not in CHOSEONG]

# every unique jamo symbol the vocabulary needs to cover Korean text
JAMO_SYMBOLS = CHOSEONG + JUNGSEONG + _JONGSEONG_ONLY  # 19 + 21 + 11 = 51

_CHO_INDEX = {c: i for i, c in enumerate(CHOSEONG)}
_JUNG_INDEX = {c: i for i, c in enumerate(JUNGSEONG)}
_JONG_INDEX = {c: i for i, c in enumerate(JONGSEONG)}


def is_hangul_syllable(ch):
    """True if `ch` is one precomposed Hangul syllable block (가-힣)."""
    return len(ch) == 1 and HANGUL_BASE <= ord(ch) <= HANGUL_LAST


def decompose_syllable(ch):
    """Split one Hangul syllable block into (cho, jung, jong) jamo strings.

    `jong` is "" when the syllable has no final consonant.
    """
    code = ord(ch) - HANGUL_BASE
    cho, rem = divmod(code, 21 * 28)
    jung, jong = divmod(rem, 28)
    return CHOSEONG[cho], JUNGSEONG[jung], JONGSEONG[jong]


def decompose_hangul(text):
    """Expand every Hangul syllable block in `text` into cho+jung(+jong) jamo.

    Non-Hangul characters (space, punctuation, digits, Latin letters, jamo
    already standing alone) pass through unchanged. This is the only text
    transform training data and inference need to agree on: whatever string
    this function returns is what gets tokenized against char_vocab.json.
    """
    out = []
    for ch in text:
        if is_hangul_syllable(ch):
            out.extend(decompose_syllable(ch))
        else:
            out.append(ch)
    return "".join(out)


def compose_hangul(text):
    """Best-effort inverse of decompose_hangul, for round-trip debugging.

    Greedily folds cho(+jung(+jong)) jamo runs back into syllable blocks.
    Not used by training or inference -- the model only ever consumes
    decomposed jamo and only ever emits audio.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _CHO_INDEX and i + 1 < n and text[i + 1] in _JUNG_INDEX:
            cho = _CHO_INDEX[ch]
            jung = _JUNG_INDEX[text[i + 1]]
            i += 2
            jong = 0
            if i < n and text[i] in _JONG_INDEX and _JONG_INDEX[text[i]] != 0:
                # only consume as a final if it's not the start of the next
                # syllable (i.e. not followed by a vowel jamo)
                if not (i + 1 < n and text[i + 1] in _JUNG_INDEX):
                    jong = _JONG_INDEX[text[i]]
                    i += 1
            code = HANGUL_BASE + (cho * 21 + jung) * 28 + jong
            out.append(chr(code))
        else:
            out.append(ch)
            i += 1
    return "".join(out)
