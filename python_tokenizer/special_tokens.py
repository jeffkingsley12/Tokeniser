"""
Special Tokens for Luganda Tokenizer

Defines special token constants used throughout the tokenizer pipeline.

Design notes
------------
- Token *strings* are class-level constants so callers can reference them
  without instantiating the class (e.g. SpecialTokens.MORPH_SEP).
- Token *IDs* are also class-level constants so the mapping never drifts
  between get_all_tokens() and direct attribute access.
- IDs start at 100_000 to avoid collision with the C tokenizer's regular
  vocabulary (config.py: BASE_SYMBOL_OFFSET = 4096, MAX_SYLLABLES = 4096).
"""

from typing import Dict


class SpecialTokens:
    """Special token constants for Luganda tokenization."""

    # ── Token strings ─────────────────────────────────────────────────────────
    MORPH_SEP       = "<|morph_sep|>"
    SPACE           = "<|space|>"
    NUM_START_TOKEN = "<|num_start|>"
    NUM_END_TOKEN   = "<|num_end|>"
    DATE_START_TOKEN = "<|date_start|>"
    DATE_END_TOKEN  = "<|date_end|>"
    TIME_START_TOKEN = "<|time_start|>"
    TIME_END_TOKEN  = "<|time_end|>"
    UNKNOWN         = "<|unk|>"
    EOS             = "<|eos|>"
    PAD             = "<|pad|>"

    # ── Token IDs (class-level so both string and ID are single source of truth)
    MORPH_SEP_ID        = 100_000
    SPACE_ID            = 100_001
    NUM_START_TOKEN_ID  = 100_002
    NUM_END_TOKEN_ID    = 100_003
    DATE_START_TOKEN_ID = 100_004
    DATE_END_TOKEN_ID   = 100_005
    TIME_START_TOKEN_ID = 100_006
    TIME_END_TOKEN_ID   = 100_007
    UNKNOWN_ID          = 100_008
    EOS_ID              = 100_009
    PAD_ID              = 100_010

    @classmethod
    def get_all_tokens(cls) -> Dict[str, int]:
        """
        Return all special tokens as a dict mapping token string → token ID.

        IDs are in the 100_000+ range to avoid collision with the regular
        vocabulary. This method is the single source of truth — the class-level
        *_ID constants mirror these values and must stay in sync.

        Returns
        -------
        Dict[str, int]
            {token_string: token_id}
        """
        return {
            cls.MORPH_SEP:        cls.MORPH_SEP_ID,
            cls.SPACE:            cls.SPACE_ID,
            cls.NUM_START_TOKEN:  cls.NUM_START_TOKEN_ID,
            cls.NUM_END_TOKEN:    cls.NUM_END_TOKEN_ID,
            cls.DATE_START_TOKEN: cls.DATE_START_TOKEN_ID,
            cls.DATE_END_TOKEN:   cls.DATE_END_TOKEN_ID,
            cls.TIME_START_TOKEN: cls.TIME_START_TOKEN_ID,
            cls.TIME_END_TOKEN:   cls.TIME_END_TOKEN_ID,
            cls.UNKNOWN:          cls.UNKNOWN_ID,
            cls.EOS:              cls.EOS_ID,
            cls.PAD:              cls.PAD_ID,
        }

    @classmethod
    def get_inverse(cls) -> Dict[int, str]:
        """Return token ID → token string mapping."""
        return {v: k for k, v in cls.get_all_tokens().items()}
