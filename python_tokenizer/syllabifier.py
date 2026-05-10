"""
Syllabifier - Core syllable parsing for Luganda text.
Ported from src/syllabifier.c

Implements phonotactic validation and syllable boundary detection.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from .config import (
    VOWELS, CONSONANTS, VALID_ONSETS, MAX_SEQ_LEN, 
    SPECIAL_TOKENS_COUNT, BASE_SYMBOL_OFFSET
)
from .seeds import PHONO_SEEDS, MORPHEME_SEEDS


@dataclass
class Syllable:
    """Represents a parsed syllable."""
    onset: str          # Initial consonant(s)
    nucleus: str        # Vowel(s)
    coda: str          # Final consonant (if any)
    text: str          # Full syllable text
    
    def __str__(self) -> str:
        return self.text
    
    def __repr__(self) -> str:
        return f"Syllable('{self.text}', onset='{self.onset}', nucleus='{self.nucleus}')"


class SyllableTable:
    """
    Hash table mapping syllable strings to unique IDs.
    Ported from the C SyllableTable structure.
    """
    
    def __init__(self, capacity: int = BASE_SYMBOL_OFFSET):
        self.capacity = capacity
        self.count = 0
        self._table: Dict[str, int] = {}  # syllable -> id
        self._id_to_syllable: Dict[int, str] = {}  # id -> syllable
        self._frozen = False
        
    def intern(self, syllable: str) -> int:
        """Add syllable to table and return its ID."""
        if syllable in self._table:
            return self._table[syllable]
        
        if self._frozen:
            raise RuntimeError("Cannot intern into frozen SyllableTable")
        
        if self.count >= self.capacity:
            raise OverflowError(f"SyllableTable full ({self.count} >= {self.capacity})")
        
        syllable_id = self.count
        self._table[syllable] = syllable_id
        self._id_to_syllable[syllable_id] = syllable
        self.count += 1
        return syllable_id
    
    def intern_fixed(self, syllable: str, syllable_id: int) -> int:
        """Insert syllable at a specific ID (for seeding)."""
        if self._frozen:
            raise RuntimeError("Cannot intern into frozen SyllableTable")
        
        if syllable in self._table:
            return self._table[syllable]
        
        if syllable_id >= self.capacity:
            raise OverflowError(f"ID {syllable_id} >= capacity {self.capacity}")
        
        self._table[syllable] = syllable_id
        self._id_to_syllable[syllable_id] = syllable
        if syllable_id >= self.count:
            self.count = syllable_id + 1
        return syllable_id
    
    def lookup(self, syllable: str) -> Optional[int]:
        """Get ID for a syllable, or None if not found."""
        return self._table.get(syllable)
    
    def decode(self, syllable_id: int) -> Optional[str]:
        """Get syllable text for an ID."""
        return self._id_to_syllable.get(syllable_id)
    
    def freeze(self) -> None:
        """Freeze the table - no more inserts allowed."""
        self._frozen = True
    
    def seed_phonotactic(self) -> int:
        """Seed with phonotactic syllables."""
        seeded = 0
        for seed in PHONO_SEEDS:
            self.intern(seed.lower())
            seeded += 1
        return seeded
    
    def seed_morphemes(self) -> int:
        """Seed with morpheme patterns."""
        seeded = 0
        for seed in MORPHEME_SEEDS:
            # Clean and normalize
            clean = self._clean_seed(seed)
            if clean:
                self.intern(clean.lower())
                seeded += 1
        return seeded
    
    def seed_special(self) -> None:
        """Seed special tokens (IDs 0-4)."""
        from .config import SPECIAL_TOKENS
        for i, token in enumerate(SPECIAL_TOKENS):
            self.intern_fixed(token, i)
    
    def seed_bytes(self) -> None:
        """Seed all byte values (for UTF-8 passthrough)."""
        for i in range(256):
            ch = chr(i) if i >= 32 and i < 127 else f"<byte_{i}>"
            self.intern_fixed(ch, SPECIAL_TOKENS_COUNT + i)
    
    @staticmethod
    def _clean_seed(seed: str) -> str:
        """Strip comments and whitespace from seed string."""
        # Strip comments
        for comment_char in '#;':
            if comment_char in seed:
                seed = seed[:seed.index(comment_char)]
        # Strip whitespace
        return seed.strip()


class Syllabifier:
    """
    Main syllabification engine.
    Ported from the C syllabify() function.
    """
    
    def __init__(self, stbl: Optional[SyllableTable] = None):
        self.stbl = stbl or SyllableTable()
        self._frozen = False
        
    def syllabify(self, text: str) -> List[Syllable]:
        """
        Parse text into syllables using phonotactic constraints.
        
        This implements the Luganda syllabification algorithm:
        1. Scan for valid onsets (consonant clusters)
        2. Match vowel nucleus
        3. Handle coda if present
        """
        text = text.lower().strip()
        if not text:
            return []
        
        syllables = []
        i = 0
        n = len(text)
        
        while i < n:
            # Skip non-alphanumeric
            if not text[i].isalnum():
                i += 1
                continue
            
            # Try to parse a syllable
            syllable = self._parse_syllable(text, i)
            if syllable:
                syllables.append(syllable)
                i += len(syllable.text)
            else:
                # Skip unrecognized character
                i += 1
        
        return syllables
    
    def _parse_syllable(self, text: str, pos: int) -> Optional[Syllable]:
        """Parse a single syllable starting at position."""
        n = len(text)
        
        # Find onset (consonant cluster)
        onset = ""
        i = pos
        while i < n and text[i] in CONSONANTS:
            # Check if adding this char keeps a valid onset
            test_onset = text[pos:i+1]
            if test_onset in VALID_ONSETS or len(test_onset) == 1:
                onset = test_onset
                i += 1
            else:
                break
        
        # Must have a vowel nucleus
        if i >= n or text[i] not in VOWELS:
            # No vowel - try single character fallback
            if pos < n:
                ch = text[pos]
                return Syllable("", ch, "", ch)
            return None
        
        # Get nucleus (can be 1-2 vowels)
        nucleus = text[i]
        i += 1
        if i < n and text[i] in VOWELS:
            # Long vowel or diphthong
            nucleus += text[i]
            i += 1
        
        # Check for coda (final consonant before next syllable)
        coda = ""
        if i < n and text[i] in CONSONANTS:
            # Only single consonant coda in Luganda
            coda = text[i]
            i += 1
        
        full = onset + nucleus + coda
        return Syllable(onset, nucleus, coda, full)
    
    def syllabify_to_ids(self, text: str) -> List[int]:
        """Syllabify and return syllable IDs."""
        syllables = self.syllabify(text)
        ids = []
        for syl in syllables:
            syl_id = self.stbl.lookup(syl.text)
            if syl_id is not None:
                ids.append(syl_id)
            else:
                # Unknown syllable - use byte fallback
                for ch in syl.text:
                    byte_id = SPECIAL_TOKENS_COUNT + ord(ch)
                    ids.append(byte_id)
        return ids
    
    def freeze(self) -> None:
        """Freeze the syllabifier - no more table modifications."""
        self._frozen = True
        self.stbl.freeze()


# Legacy alias for compatibility
SyllabifyResult = List[Syllable]
