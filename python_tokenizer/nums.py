"""
Enhanced Luganda Linguistic Research System
============================================

Combines production-ready engineering from nums1.py with comprehensive
linguistic research capabilities from nums.py.

Key Features:
1. Complete linguistic metadata (etymology, phonology, morphology)
2. Diachronic analysis and historical forms
3. Dialectal variation tracking
4. Noun class system with detailed agreement patterns
5. Currency normalization
6. Comprehensive research tools and query interfaces
7. Export capabilities for linguistic research

PATCHED:
- FIX #1: get_metadata now normalizes to lowercase and checks for gemination.
- FIX #2: _build_complete_variant_metadata added and called in __init__.
- FIX #3: luganda_to_number replaced with enhanced hierarchical parser.
"""

import re
import json
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Set, Any
from functools import lru_cache, partial
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime


# ============================================================================
# LINGUISTIC METADATA STRUCTURES
# ============================================================================

class SymbolType(Enum):
    """Categories of numerical symbols in Luganda"""
    NUMERICAL_ADJECTIVE = auto()  # 1-5, agree with nouns
    NUMERICAL_NOUN = auto()       # 6-9, don't agree
    STANDARD_NOUN = auto()        # Large numbers (mutwalo, kakadde)
    PARTICLE = auto()             # Connectors (mu, na)
    ZERO = auto()                 # Special category for zero


class PhonologicalFeature(Enum):
    """Phonological characteristics"""
    GEMINATE_CONSONANT = auto()   # bbiri, kkumi (doubled consonants)
    PRENASALIZED = auto()         # nkumi, nsanvu (nasal + consonant)
    VOWEL_HARMONY = auto()        # Patterns in agreement forms
    NASAL_ASSIMILATION = auto()   # n' → m before labials
    TONAL_HIGH = auto()           # High tone marking
    TONAL_LOW = auto()            # Low tone marking


class MorphologicalStructure(Enum):
    """Morphological composition"""
    MONOMORPHEMIC = auto()        # Single root: emu, bbiri
    PREFIX_ROOT = auto()          # mu-kaaga, ki-kumi
    COMPOUND = auto()             # Complex formations
    DERIVED = auto()              # Derived from other forms


@dataclass
class EtymologyInfo:
    """Etymological information for a number word"""
    proto_bantu_root: Optional[str] = None
    cognates: Dict[str, str] = field(default_factory=dict)  # {language: form}
    semantic_shift: Optional[str] = None
    historical_forms: List[Tuple[str, str]] = field(default_factory=list)  # [(period, form)]
    notes: str = ""


@dataclass
class DialectalVariation:
    """Regional and dialectal variants"""
    standard_form: str
    variants: Dict[str, str] = field(default_factory=dict)  # {region/dialect: form}
    register: str = "neutral"  # formal, informal, archaic, etc.
    frequency: int = 100  # Relative frequency (0-100)
    notes: str = ""


@dataclass
class LinguisticMetadata:
    """Complete linguistic information for a numeral"""
    # Core identification
    word: str
    numeric_value: Optional[int]
    symbol_type: SymbolType
    
    # Grammatical information
    noun_class: Optional[int] = None
    plural_form: Optional[str] = None
    agreement_forms: Dict[str, str] = field(default_factory=dict)
    
    # Phonological features
    phonological_features: Set[PhonologicalFeature] = field(default_factory=set)
    syllable_structure: str = ""  # CV, CVC, etc.
    tone_pattern: str = ""  # H, L, HL, etc.
    
    # Morphological analysis
    morphological_type: MorphologicalStructure = MorphologicalStructure.MONOMORPHEMIC
    morpheme_breakdown: List[str] = field(default_factory=list)
    root: Optional[str] = None
    
    # Historical and comparative
    etymology: Optional[EtymologyInfo] = None
    dialectal_info: Optional[DialectalVariation] = None
    
    # Usage information
    frequency_rank: int = 0  # 1 = most frequent
    usage_contexts: Set[str] = field(default_factory=set)  # counting, monetary, ordinal, etc.
    collocations: List[str] = field(default_factory=list)
    example_sentences: List[str] = field(default_factory=list)
    
    # Research metadata
    first_documented: Optional[str] = None  # Date or source
    sources: List[str] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# CORE: NOUN CLASS SYSTEM (Enhanced with Linguistic Details)
# ============================================================================

class NounClass(Enum):
    """
    Luganda noun classes with comprehensive linguistic information.
    Format: (class_number, singular_prefix, plural_prefix, semantic_category)
    """
    CLASS_1 = (1, "omu", "aba", "humans")
    CLASS_2 = (2, "omu", "emi", "plants_trees")
    CLASS_3 = (3, "eli", "aga", "paired_items")
    CLASS_4 = (4, "eki", "ebi", "inanimate_objects")
    CLASS_5 = (5, "eri", "ama", "liquids_masses")
    CLASS_6 = (6, "aka", "obu", "diminutives")
    CLASS_7 = (7, "olu", "en", "long_thin_objects")
    CLASS_8 = (8, "olu", "aga", "abstract_concepts")
    CLASS_9 = (9, "", "zi", "animals_foreign_words")
    CLASS_10 = (10, "aka", "otu", "diminutive_plural")

    @property
    def number(self) -> int:
        return self.value[0]

    @property
    def singular_prefix(self) -> str:
        return self.value[1]

    @property
    def plural_prefix(self) -> str:
        return self.value[2]
    
    @property
    def semantic_category(self) -> str:
        return self.value[3]


class NumeralAgreement:
    """
    Enhanced numeral agreement with linguistic analysis.
    """
    STEMS = {1: "mu", 2: "biri", 3: "satu", 4: "nya", 5: "taano", 10: "kumi"}
    
    AGREEMENT_PREFIXES = {
        1: {
            NounClass.CLASS_1: "omu", NounClass.CLASS_2: "gumu", NounClass.CLASS_3: "limu",
            NounClass.CLASS_4: "kimu", NounClass.CLASS_5: "kamu", NounClass.CLASS_6: "kumu",
            NounClass.CLASS_7: "lumu",
        },
        2: {
            NounClass.CLASS_1: "babiri", NounClass.CLASS_2: "ebiri", NounClass.CLASS_3: "abiri",
            NounClass.CLASS_4: "bibiri", NounClass.CLASS_5: "bubiri",
        },
        3: {
            NounClass.CLASS_1: "basatu", NounClass.CLASS_2: "esatu", NounClass.CLASS_3: "asatu",
            NounClass.CLASS_4: "bisatu", NounClass.CLASS_5: "busatu",
        },
        4: {
            NounClass.CLASS_1: "bana", NounClass.CLASS_2: "ena", NounClass.CLASS_3: "ana",
            NounClass.CLASS_4: "bina", NounClass.CLASS_5: "buna",
        },
        5: {
            NounClass.CLASS_1: "bataano", NounClass.CLASS_2: "etaano", NounClass.CLASS_3: "ataano",
            NounClass.CLASS_4: "bitaano", NounClass.CLASS_5: "butaano",
        },
        10: {
            NounClass.CLASS_1: "abakumi", NounClass.CLASS_2: "egikumi", NounClass.CLASS_3: "agakumi",
            NounClass.CLASS_4: "ebikumi", NounClass.CLASS_5: "obukumi",
        }
    }

    @classmethod
    def get_agreement_form(cls, number: int, noun_class: NounClass) -> Optional[str]:
        if number not in cls.AGREEMENT_PREFIXES:
            return None
        return cls.AGREEMENT_PREFIXES[number].get(noun_class)

    @classmethod
    def is_agreeable(cls, number: int) -> bool:
        return number in cls.STEMS
    
    @classmethod
    def analyze_agreement(cls, number: int, noun_class: NounClass) -> Dict[str, Any]:
        """Provide detailed analysis of agreement formation"""
        if number not in cls.STEMS:
            return {"agreeable": False}
        
        stem = cls.STEMS[number]
        prefix = cls.AGREEMENT_PREFIXES[number].get(noun_class)
        
        return {
            "agreeable": True,
            "stem": stem,
            "prefix": prefix[:prefix.find(stem)] if prefix else None,
            "full_form": prefix,
            "noun_class": noun_class.number,
            "semantic_category": noun_class.semantic_category,
            "morphological_structure": f"prefix({prefix[:prefix.find(stem)] if prefix else ''}) + stem({stem})"
        }


# ============================================================================
# ENHANCED NUMERICAL SYSTEM WITH LINGUISTIC METADATA
# ============================================================================

class LugandaNumericalSystem:
    """
    Comprehensive Luganda numerical system with linguistic research capabilities.
    """
    
    def __init__(self):
        self.metadata_store: Dict[str, LinguisticMetadata] = {}
        self._build_base_numbers()
        self._build_decomposition_rules()
        self._build_reverse_mappings()
        self._build_linguistic_metadata()
        
        # FIX #2: Call new method to add all variant metadata
        self._build_complete_variant_metadata()
        
    def _build_base_numbers(self):
        """Build base number words."""
        self.BASE_NUMBERS = {
            0: 'zeero', 1: 'emu', 2: 'bbiri', 3: 'ssatu', 4: 'nnya',
            5: 'ttaano', 6: 'mukaaga', 7: 'musanvu', 8: 'munaana',
            9: 'mwenda', 10: 'kkumi'
        }

    def _build_decomposition_rules(self):
        """Build rules for decomposing numbers hierarchically."""
        self.DECOMPOSITION_RULES = [
            (1_000_000_000_000_000_000_000, 'busedde', 
             partial(self._handle_large_unit, 
                    unit_value=1_000_000_000_000_000_000_000,
                    singular='kasedde', 
                    plural='busedde')),
            (1_000_000_000_000_000_000, 'bufukunya', 
             partial(self._handle_large_unit,
                    unit_value=1_000_000_000_000_000_000,
                    singular='kafukunya',
                    plural='bufukunya')),
            (1_000_000_000_000, 'buwumbi',
             partial(self._handle_large_unit,
                    unit_value=1_000_000_000_000,
                    singular='kawumbi',
                    plural='buwumbi')),
            (1_000_000, 'bukadde',
             partial(self._handle_large_unit,
                    unit_value=1_000_000,
                    singular='kakadde',
                    plural='bukadde')),
            (100_000, 'kasiriivu', self._handle_hundred_thousands),
            (10_000, 'mutwalo', self._handle_ten_thousands),
            (1_000, 'lukumi', self._handle_thousands),
            (100, 'kikumi', self._handle_hundreds),
            (60, None, self._handle_sixties_nineties),
            (20, 'amakumi', self._handle_twenties_fifties),
            (11, 'kkumi', self._handle_teens),
        ]

    def _build_reverse_mappings(self):
        """Build comprehensive reverse lookup mappings."""
        self.WORD_TO_NUMBER = {word: num for num, word in self.BASE_NUMBERS.items()}
        
        variant_spellings = {
            'kumi': 10, 'kkumi': 10,
            'emu': 1, 'kimu': 1, 'omu': 1, 'gumu': 1, 'limu': 1, 'kamu': 1, 'kumu': 1, 'lumu': 1,
            'bbiri': 2, 'biri': 2, 'babiri': 2, 'ebiri': 2, 'bubiri': 2,
            'ssatu': 3, 'satu': 3, 'basatu': 3, 'esatu': 3, 'busatu': 3, 'asatu': 3,
            'nnya': 4, 'bana': 4, 'ena': 4, 'buna': 4, 'ana': 4,
            'ttaano': 5, 'taano': 5, 'bataano': 5, 'etaano': 5, 'butaano': 5, 'ataano': 5,
        }
        self.WORD_TO_NUMBER.update(variant_spellings)
        
        # BUG FIX: Use distinct keys for tens to avoid overwriting Class 3 agreement forms
        # 'asatu' (3) vs 'amakumi asatu' (30), 'ana' (4) vs 'amakumi ana' (40), etc.
        tens_mappings = {
            'abiri': 20, 'amakumi asatu': 30, 'amakumi ana': 40, 'amakumi ataano': 50, 'nkaaga': 60,
            'nsanvu': 70, 'kinaana': 80, 'kyenda': 90, 'kikumi': 100,
            'bibiri': 200, 'bisatu': 300, 'bina': 400, 'bitaano': 500,
            'tutaano': 500, 'lukaaga': 600, 'lusanvu': 700, 'lunaana': 800,
            'lwenda': 900, 'lukumi': 1000, 'nkumi': 1000, 'kakaaga': 6000,
            'kasanvu': 7000, 'kanaana': 8000, 'kenda': 9000,
            'mutwalo': 10000, 'mitwalo': 10000,
            'kasiriivu': 100000, 'busiriivu': 100000,
            'kakadde': 1000000, 'bukadde': 1000000,
            'kawumbi': 1_000_000_000_000, 'buwumbi': 1_000_000_000_000,
            'kafukunya': 1_000_000_000_000_000_000, 'bufukunya': 1_000_000_000_000_000_000,
            'kasedde': 1_000_000_000_000_000_000_000, 'busedde': 1_000_000_000_000_000_000_000,
        }
        self.WORD_TO_NUMBER.update(tens_mappings)

    def _build_linguistic_metadata(self):
        """Build comprehensive linguistic metadata for all numerals."""
        
        # Basic numbers (0-10) with full metadata
        self._add_metadata("zeero", 0, SymbolType.ZERO,
            phonological={PhonologicalFeature.GEMINATE_CONSONANT},
            etymology=EtymologyInfo(
                proto_bantu_root=None,
                cognates={"English": "zero", "Swahili": "sifuri"},
                notes="Borrowed from English"
            ),
            syllable="CV.CV", tone="LL",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            frequency_rank=1,
            usage_contexts={"counting", "mathematics"},
            example_sentences=["Ssiringi zeero - zero shillings"]
        )
        
        self._add_metadata("emu", 1, SymbolType.NUMERICAL_ADJECTIVE,
            phonological=set(),
            etymology=EtymologyInfo(
                proto_bantu_root="*-mʊ̀",
                cognates={"Swahili": "moja", "Kinyarwanda": "rimwe"},
                historical_forms=[("Proto-Bantu", "*-mʊ̀")]
            ),
            syllable="V.CV", tone="HL",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            root="mu",
            frequency_rank=2,
            usage_contexts={"counting", "agreement"},
            example_sentences=["Omuntu omu - one person"]
        )
        
        self._add_metadata("bbiri", 2, SymbolType.NUMERICAL_ADJECTIVE,
            phonological={PhonologicalFeature.GEMINATE_CONSONANT},
            etymology=EtymologyInfo(
                proto_bantu_root="*-bàdì",
                cognates={"Swahili": "mbili", "Kinyarwanda": "babiri"}
            ),
            syllable="CV.CV", tone="HL",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            root="biri",
            frequency_rank=3,
            usage_contexts={"counting", "agreement", "duplication"},
            example_sentences=["Abantu babiri - two people"]
        )
        
        self._add_metadata("ssatu", 3, SymbolType.NUMERICAL_ADJECTIVE,
            phonological={PhonologicalFeature.GEMINATE_CONSONANT},
            etymology=EtymologyInfo(
                proto_bantu_root="*-tàtʊ̀",
                cognates={"Swahili": "tatu", "Kinyarwanda": "batatu"}
            ),
            syllable="CV.CV", tone="HL",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            root="satu",
            frequency_rank=4,
            usage_contexts={"counting", "agreement"},
            example_sentences=["Emmere esatu - three items of food"]
        )
        
        self._add_metadata("nnya", 4, SymbolType.NUMERICAL_ADJECTIVE,
            phonological={PhonologicalFeature.PRENASALIZED},
            etymology=EtymologyInfo(
                proto_bantu_root="*-nàì",
                cognates={"Swahili": "nne", "Kinyarwanda": "bane"}
            ),
            syllable="CV", tone="H",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            root="nya",
            frequency_rank=5,
            usage_contexts={"counting", "agreement"},
            example_sentences=["Ebitabo bina - four books"]
        )
        
        self._add_metadata("ttaano", 5, SymbolType.NUMERICAL_ADJECTIVE,
            phonological={PhonologicalFeature.GEMINATE_CONSONANT},
            etymology=EtymologyInfo(
                proto_bantu_root="*-tánò",
                cognates={"Swahili": "tano", "Kinyarwanda": "batanu"}
            ),
            syllable="CV.V", tone="HL",
            morphology=MorphologicalStructure.MONOMORPHEMIC,
            root="taano",
            frequency_rank=6,
            usage_contexts={"counting", "agreement"},
            example_sentences=["Amawanga ataano - five nations"]
        )
        
        # Numbers 6-9 (numerical nouns with Class 2 prefixes)
        self._add_metadata("mukaaga", 6, SymbolType.NUMERICAL_NOUN,
            noun_class=2,
            phonological=set(),
            etymology=EtymologyInfo(
                proto_bantu_root="*-kàágà",
                cognates={"Swahili": "sita", "Kinyarwanda": "gatandatu"}
            ),
            syllable="CV.CV.V", tone="LHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["mu-", "kaaga"],
            root="kaaga",
            frequency_rank=7,
            usage_contexts={"counting"},
            example_sentences=["Abantu mukaaga - six people"]
        )
        
        self._add_metadata("musanvu", 7, SymbolType.NUMERICAL_NOUN,
            noun_class=2,
            phonological=set(),
            syllable="CV.CV.CV", tone="LHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["mu-", "sanvu"],
            root="sanvu",
            frequency_rank=8
        )
        
        self._add_metadata("munaana", 8, SymbolType.NUMERICAL_NOUN,
            noun_class=2,
            phonological=set(),
            syllable="CV.CV.V", tone="LHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["mu-", "naana"],
            root="naana",
            frequency_rank=9
        )
        
        self._add_metadata("mwenda", 9, SymbolType.NUMERICAL_NOUN,
            noun_class=2,
            phonological=set(),
            syllable="CV.CV", tone="HL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["mu-", "enda"],
            root="enda",
            frequency_rank=10
        )
        
        # Ten and its forms
        self._add_metadata("kkumi", 10, SymbolType.NUMERICAL_NOUN,
            noun_class=5,
            plural_form="amakumi",
            phonological={PhonologicalFeature.GEMINATE_CONSONANT},
            etymology=EtymologyInfo(
                proto_bantu_root="*-kʊ́mì",
                cognates={"Swahili": "kumi", "Kinyarwanda": "icumi"}
            ),
            syllable="CV.CV", tone="HL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["ki-", "kumi"],
            root="kumi",
            frequency_rank=11,
            usage_contexts={"counting", "base_ten"},
            example_sentences=["Kkumi - ten", "Kkumi na emu - eleven"]
        )
        
        self._add_metadata("amakumi", 10, SymbolType.NUMERICAL_NOUN,
            noun_class=5,
            phonological=set(),
            syllable="V.CV.CV.CV", tone="LLHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["ama-", "kumi"],
            root="kumi",
            frequency_rank=12,
            usage_contexts={"tens", "20-50"},
            notes="Plural form of kkumi, used in constructing 20-50"
        )
        
        # Hundreds
        self._add_metadata("kikumi", 100, SymbolType.NUMERICAL_NOUN,
            noun_class=4,
            plural_form="bikumi",
            phonological=set(),
            syllable="CV.CV.CV", tone="LHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["ki-", "kumi"],
            root="kumi",
            frequency_rank=13,
            notes="Class 4 singular, derived from kkumi"
        )
        
        # Thousands
        self._add_metadata("lukumi", 1000, SymbolType.NUMERICAL_NOUN,
            noun_class=7,
            plural_form="nkumi",
            phonological=set(),
            syllable="CV.CV.CV", tone="LHL",
            morphology=MorphologicalStructure.PREFIX_ROOT,
            morpheme_breakdown=["lu-", "kumi"],
            root="kumi",
            frequency_rank=14,
            notes="Class 7 singular"
        )
        
        # Large numbers with linguistic analysis
        self._add_metadata("mutwalo", 10000, SymbolType.STANDARD_NOUN,
            noun_class=2,
            plural_form="mitwalo",
            etymology=EtymologyInfo(
                notes="From 'okutwala' (to carry) - amount that can be carried"
            ),
            syllable="CV.CV.CV", tone="LHL",
            morphology=MorphologicalStructure.DERIVED,
            frequency_rank=15
        )
        
        self._add_metadata("kakadde", 1000000, SymbolType.STANDARD_NOUN,
            noun_class=6,
            plural_form="bukadde",
            etymology=EtymologyInfo(
                notes="Means 'old man/ancestor' - metaphor for large number"
            ),
            syllable="CV.CV.CV", tone="LHL",
            morphology=MorphologicalStructure.DERIVED,
            frequency_rank=16
        )

    def _add_metadata(self, word: str, value: Optional[int], symbol_type: SymbolType,
                     noun_class: Optional[int] = None, plural_form: Optional[str] = None,
                     phonological: Set[PhonologicalFeature] = None,
                     etymology: Optional[EtymologyInfo] = None,
                     syllable: str = "", tone: str = "",
                     morphology: MorphologicalStructure = MorphologicalStructure.MONOMORPHEMIC,
                     morpheme_breakdown: List[str] = None,
                     root: Optional[str] = None,
                     frequency_rank: int = 0,
                     usage_contexts: Set[str] = None,
                     example_sentences: List[str] = None,
                     notes: str = ""):
        """Helper to add linguistic metadata."""
        self.metadata_store[word] = LinguisticMetadata(
            word=word,
            numeric_value=value,
            symbol_type=symbol_type,
            noun_class=noun_class,
            plural_form=plural_form,
            phonological_features=phonological or set(),
            syllable_structure=syllable,
            tone_pattern=tone,
            morphological_type=morphology,
            morpheme_breakdown=morpheme_breakdown or [],
            root=root,
            etymology=etymology,
            frequency_rank=frequency_rank,
            usage_contexts=usage_contexts or set(),
            example_sentences=example_sentences or [],
            notes=notes
        )

    # ========================================================================
    # FIX #2: COMPLETE METADATA FOR ALL NUMBER VARIANTS
    # ========================================================================

    def _build_complete_variant_metadata(self):
        """
        Add metadata for ALL variants of numbers 1-10, including:
        - Base forms (emu, bbiri, ssatu, etc.)
        - Simplified forms (biri, satu, etc.)
        - All agreement forms (babiri, ebiri, bubiri, etc.)
        """
        
        # Variant mappings with linguistic info
        variant_data = {
            # Number 1 variants
            'emu': (1, SymbolType.NUMERICAL_ADJECTIVE, {PhonologicalFeature.GEMINATE_CONSONANT}),
            'mu': (1, SymbolType.NUMERICAL_ADJECTIVE, set()),
            
            # Number 2 variants
            'bbiri': (2, SymbolType.NUMERICAL_ADJECTIVE, {PhonologicalFeature.GEMINATE_CONSONANT}),
            'biri': (2, SymbolType.NUMERICAL_ADJECTIVE, set()),  # CRITICAL FIX
            
            # Number 3 variants
            'ssatu': (3, SymbolType.NUMERICAL_ADJECTIVE, {PhonologicalFeature.GEMINATE_CONSONANT}),
            'satu': (3, SymbolType.NUMERICAL_ADJECTIVE, set()),  # CRITICAL FIX
            
            # Number 4 variants
            'nnya': (4, SymbolType.NUMERICAL_ADJECTIVE, {PhonologicalFeature.PRENASALIZED}),
            'nya': (4, SymbolType.NUMERICAL_ADJECTIVE, set()),
            
            # Number 5 variants
            'ttaano': (5, SymbolType.NUMERICAL_ADJECTIVE, {PhonologicalFeature.GEMINATE_CONSONANT}),
            'taano': (5, SymbolType.NUMERICAL_ADJECTIVE, set()),  # CRITICAL FIX
            
            # Number 10 variants
            'kkumi': (10, SymbolType.NUMERICAL_NOUN, {PhonologicalFeature.GEMINATE_CONSONANT}),
            'kumi': (10, SymbolType.NUMERICAL_NOUN, set()),  # CRITICAL FIX
        }
        
        # Add all variants to metadata store
        for word, (value, sym_type, phon_features) in variant_data.items():
            if word not in self.metadata_store:
                self._add_metadata(
                    word=word,
                    value=value,
                    symbol_type=sym_type,
                    phonological=phon_features,
                    syllable="CV.CV" if len(word) > 3 else "CV",
                    tone="HL",
                    morphology=MorphologicalStructure.MONOMORPHEMIC,
                    root=word.lstrip('bstnk'),  # Remove geminate/prenasalized consonants
                    frequency_rank=100,
                    usage_contexts={"counting", "agreement"},
                    notes=f"Variant form of {value}"
                )
        
        # Add all Class 1-5 agreement forms for 1-5
        agreement_variants = {
            # Class 1 (people)
            'omu': (1, 1), 'babiri': (2, 1), 'basatu': (3, 1), 'bana': (4, 1), 'bataano': (5, 1),
            
            # Class 2 (plants/trees)
            'gumu': (1, 2), 'ebiri': (2, 2), 'esatu': (3, 2), 'ena': (4, 2), 'etaano': (5, 2),
            
            # Class 3 (paired items)
            'limu': (1, 3), 'abiri': (2, 3), 'asatu': (3, 3), 'ana': (4, 3), 'ataano': (5, 3),
            
            # Class 4 (inanimate)
            'kimu': (1, 4), 'bibiri': (2, 4), 'bisatu': (3, 4), 'bina': (4, 4), 'bitaano': (5, 4),
            
            # Class 5 (liquids/masses)
            'kamu': (1, 5), 'bubiri': (2, 5), 'busatu': (3, 5), 'buna': (4, 5), 'butaano': (5, 5),
            
            # Class 6 (diminutives)
            'kumu': (1, 6),
            
            # Class 7 (long/thin)
            'lumu': (1, 7),
        }
        
        for word, (value, noun_class) in agreement_variants.items():
            if word not in self.metadata_store:
                self._add_metadata(
                    word=word,
                    value=value,
                    symbol_type=SymbolType.NUMERICAL_ADJECTIVE,
                    noun_class=noun_class,
                    syllable="CV.CV" if len(word) > 3 else "CV",
                    tone="HL",
                    morphology=MorphologicalStructure.PREFIX_ROOT,
                    frequency_rank=80,
                    usage_contexts={"agreement"},
                    notes=f"Class {noun_class} agreement form of {value}"
                )

    # ========================================================================
    # NUMBER CONVERSION HANDLERS (N2L)
    # ========================================================================

    def _handle_large_unit(self, number: int, noun_class: Optional[NounClass],
                           unit_value: int, singular: str, plural: str) -> str:
        units = number // unit_value
        remainder = number % unit_value
        if units == 1:
            base = f"{singular} kamu"
        else:
            units_word = self.number_to_luganda(units, NounClass.CLASS_5)
            base = f"{plural} {units_word}"
        if remainder == 0:
            return base
        remainder_word = self.number_to_luganda(remainder, noun_class)
        return f"{base} mu {remainder_word}"

    def _handle_hundred_thousands(self, number: int, noun_class: Optional[NounClass]) -> str:
        hundred_thousands = number // 100000
        remainder = number % 100000
        if hundred_thousands == 1:
            base = "kasiriivu kamu"
        else:
            base_word = self.number_to_luganda(hundred_thousands, NounClass.CLASS_5)
            base = f"busiriivu {base_word}"
        if remainder == 0: return base
        return f"{base} mu {self.number_to_luganda(remainder, noun_class)}"

    def _handle_ten_thousands(self, number: int, noun_class: Optional[NounClass]) -> str:
        ten_thousands = number // 10000
        remainder = number % 10000
        if ten_thousands == 1:
            base = "mutwalo gumu"
        else:
            base_word = self.number_to_luganda(ten_thousands, NounClass.CLASS_2)
            base = f"mitwalo {base_word}"
        if remainder == 0: return base
        return f"{base} mu {self.number_to_luganda(remainder, noun_class)}"

    def _handle_thousands(self, number: int, noun_class: Optional[NounClass]) -> str:
        thousands = number // 1000
        remainder = number % 1000
        if thousands == 1:
            base = "lukumi"
        elif 2 <= thousands <= 5:
            agreement_map = {2: 'bbiri', 3: 'ssatu', 4: 'nnya', 5: 'ttaano'}
            base = f"nkumi {agreement_map[thousands]}"
        else:
            # BUG FIX: Use .get() to avoid KeyError for values outside {6-9}
            base = {6: 'kakaaga', 7: 'kasanvu', 8: 'kanaana', 9: 'kenda'}.get(thousands, f"{thousands}")
        if remainder == 0:
            return base
        return f"{base} mu {self.number_to_luganda(remainder, noun_class)}"

    def _handle_hundreds(self, number: int, noun_class: Optional[NounClass]) -> str:
        if number == 500:
            return 'tutaano'
        hundreds = number // 100
        remainder = number % 100
        if hundreds == 1:
            base = "kikumi"
        elif 2 <= hundreds <= 5:
            base = {2: 'bibiri', 3: 'bisatu', 4: 'bina', 5: 'bitaano'}[hundreds]
        else:
            # BUG FIX: Use .get() to avoid KeyError for values outside {6-9}
            base = {6: 'lukaaga', 7: 'lusanvu', 8: 'lunaana', 9: 'lwenda'}.get(hundreds, f"{hundreds}")
        if remainder == 0:
            return base
        return f"{base} mu {self.number_to_luganda(remainder, noun_class)}"

    def _handle_sixties_nineties(self, number: int, noun_class: Optional[NounClass]) -> str:
        base_tens = (number // 10) * 10
        ones = number % 10
        tens_word = {60: 'nkaaga', 70: 'nsanvu', 80: 'kinaana', 90: 'kyenda'}[base_tens]
        if ones == 0: return tens_word
        return f"{tens_word} mu {self._get_base_or_agreement(ones, noun_class)}"

    def _handle_twenties_fifties(self, number: int, noun_class: Optional[NounClass]) -> str:
        tens = number // 10
        ones = number % 10
        tens_word = {2: 'abiri', 3: 'asatu', 4: 'ana', 5: 'ataano'}[tens]
        if ones == 0: return tens_word
        return f"{tens_word} mu {self._get_base_or_agreement(ones, noun_class)}"

    def _handle_teens(self, number: int, noun_class: Optional[NounClass]) -> str:
        ones = number - 10
        if ones == 1: return "kkumi n'emu"
        return f"kkumi na {self._get_base_or_agreement(ones, noun_class)}"

    def _get_base_or_agreement(self, number: int, noun_class: Optional[NounClass]) -> str:
        if noun_class and NumeralAgreement.is_agreeable(number):
            agreed_form = NumeralAgreement.get_agreement_form(number, noun_class)
            if agreed_form: return agreed_form
        return self.BASE_NUMBERS.get(number, str(number))

    def number_to_luganda(self, number: int, noun_class: Optional[NounClass] = None) -> str:
        if not isinstance(number, int): raise TypeError("Input must be an integer.")
        if number == 0: return "zeero"
        if number < 0: return f"minus {self.number_to_luganda(abs(number), noun_class)}"
        if number in self.BASE_NUMBERS: return self._get_base_or_agreement(number, noun_class)
        for threshold, _, handler in self.DECOMPOSITION_RULES:
            if number >= threshold: return handler(number, noun_class)
        return f"[{number}]"

    # ========================================================================
    # FIX #3: ENHANCED LUGANDA-TO-NUMBER (L2N) PARSER
    # ========================================================================

    @lru_cache(maxsize=512)
    def luganda_to_number(self, luganda_text: str) -> Optional[int]:
        """
        ENHANCED: Convert Luganda text to number with comprehensive parsing.
        
        Handles:
        - Simple forms: "emu", "bbiri", "kkumi"
        - Complex forms: "kkumi na ttaano" (15)
        - Hierarchical: "nkumi bbiri mu kikumi mu abiri" (2102)
        - Shortened forms: "abiri" (20), "bibiri" (200)
        - Agreement forms: "babiri", "ebiri", "bubiri" (all = 2)
        
        Returns:
            int: The numeric value, or None if parsing fails
        """
        if not luganda_text:
            return None
        
        text = luganda_text.strip().lower()
        
        # Quick lookup for simple cases
        if text in self.WORD_TO_NUMBER:
            return self.WORD_TO_NUMBER[text]
        
        # Handle contracted form "kkumi n'emu" = 11
        if text == "kkumi n'emu" or text == "kumi n'emu":
            return 11
        
        # Split by hierarchical connector "mu"
        parts = [p.strip() for p in re.split(r'\s+mu\s+', text)]
        total = 0
        
        for part in parts:
            part_value = 0
            
            # === PATTERN 1: "kkumi na X" (teens and decade + units) ===
            if re.search(r'\bk*umi\s+na\s+', part):
                match = re.search(r'(k*umi)\s+na\s+(.+)', part)
                if match:
                    base_word = match.group(1)
                    ones_word = match.group(2).strip()
                    
                    # Get base value (kkumi = 10, but could be in larger context)
                    base_value = self.WORD_TO_NUMBER.get(base_word, 10)
                    ones_value = self.WORD_TO_NUMBER.get(ones_word, 0)
                    
                    if ones_value and 1 <= ones_value <= 9:
                        part_value = base_value + ones_value
                    else:
                        part_value = base_value
            
            # === PATTERN 2: Compound forms with spaces ===
            elif ' ' in part:
                words = part.split()
                
                if len(words) == 2:
                    first_word, second_word = words
                    
                    # Check for unit multiplier patterns
                    # e.g., "nkumi bbiri" = 1000 * 2 = 2000
                    plural_multipliers = {
                        'nkumi': 1000,      # thousands
                        'bikumi': 100,      # hundreds
                        'mitwalo': 10000,   # ten thousands
                        'busiriivu': 100000,
                        'bukadde': 1000000,
                        'buwumbi': 1_000_000_000_000,
                        'bufukunya': 1_000_000_000_000_000_000,
                        'busedde': 1_000_000_000_000_000_000_000,
                    }
                    
                    if first_word in plural_multipliers:
                        base_value = plural_multipliers[first_word]
                        multiplier = self.WORD_TO_NUMBER.get(second_word, 1)
                        part_value = base_value * multiplier
                    
                    # Check for singular unit forms
                    elif first_word in {'mutwalo', 'kasiriivu', 'kakadde', 'kawumbi', 
                                        'kafukunya', 'kasedde'}:
                        singular_values = {
                            'mutwalo': 10000, 'kasiriivu': 100000, 'kakadde': 1000000,
                            'kawumbi': 1_000_000_000_000,
                            'kafukunya': 1_000_000_000_000_000_000,
                            'kasedde': 1_000_000_000_000_000_000_000,
                        }
                        part_value = singular_values.get(first_word, 0)
                    
                    # Check for "amakumi X" pattern (tens: 20-50)
                    elif first_word == 'amakumi':
                        # "amakumi abiri" = 20, "amakumi asatu" = 30, etc.
                        multiplier = self.WORD_TO_NUMBER.get(second_word, 0)
                        if multiplier in {2, 3, 4, 5}:  # Class 3 agreement
                            part_value = 10 * multiplier
                    
                    # Default: simple addition
                    else:
                        part_value = (self.WORD_TO_NUMBER.get(first_word, 0) + 
                                      self.WORD_TO_NUMBER.get(second_word, 0))
                
                else:
                    # More than 2 words: sum them all
                    for word in words:
                        part_value += self.WORD_TO_NUMBER.get(word, 0)
            
            # === PATTERN 3: Single word lookup ===
            else:
                part_value = self.WORD_TO_NUMBER.get(part, 0)
            
            total += part_value
        
        # Return result (handle zero case explicitly)
        return total if total > 0 or text == 'zeero' else None

    # ========================================================================
    # LINGUISTIC RESEARCH TOOLS
    # ========================================================================

    # ========================================================================
    # FIX #1: ENHANCED get_metadata()
    # ========================================================================

    def get_metadata(self, word: str) -> Optional[LinguisticMetadata]:
        """
        Retrieve complete linguistic metadata for a word.
        FIXED: Now normalizes input to lowercase for consistent lookup.
        """
        if not word:
            return None
        
        # Normalize to lowercase for lookup
        normalized = word.lower().strip()
        
        # Direct lookup
        if normalized in self.metadata_store:
            return self.metadata_store[normalized]
        
        # Try removing gemination (bbiri -> biri)
        if len(normalized) > 2 and normalized[0] == normalized[1]:
            ungeminates = normalized[1:]
            if ungeminates in self.metadata_store:
                return self.metadata_store[ungeminates]
        
        return None

    def analyze_word(self, word: str) -> Dict[str, Any]:
        """Comprehensive linguistic analysis of a numeral."""
        metadata = self.get_metadata(word) # USES FIXED GET_METADATA
        if not metadata:
            return {"error": f"No metadata found for '{word}'"}
        
        analysis = {
            "word": metadata.word,
            "numeric_value": metadata.numeric_value,
            "type": metadata.symbol_type.name,
            "phonology": {
                "syllable_structure": metadata.syllable_structure,
                "tone_pattern": metadata.tone_pattern,
                "features": [f.name for f in metadata.phonological_features]
            },
            "morphology": {
                "type": metadata.morphological_type.name,
                "breakdown": metadata.morpheme_breakdown,
                "root": metadata.root
            },
            "grammar": {
                "noun_class": metadata.noun_class,
                "plural_form": metadata.plural_form,
                "agreement_forms": metadata.agreement_forms
            }
        }
        
        if metadata.etymology:
            analysis["etymology"] = {
                "proto_bantu": metadata.etymology.proto_bantu_root,
                "cognates": metadata.etymology.cognates,
                "notes": metadata.etymology.notes
            }
        
        return analysis

    def compare_cognates(self, word: str) -> Dict[str, str]:
        """Compare with cognates in related languages."""
        metadata = self.get_metadata(word) # USES FIXED GET_METADATA
        if not metadata or not metadata.etymology:
            return {}
        return metadata.etymology.cognates

    def get_by_feature(self, feature: PhonologicalFeature) -> List[str]:
        """Find all numerals with a specific phonological feature."""
        return [word for word, meta in self.metadata_store.items() 
                if feature in meta.phonological_features]

    def get_by_noun_class(self, noun_class: int) -> List[Tuple[str, int]]:
        """Find all numerals belonging to a specific noun class."""
        return [(meta.word, meta.numeric_value) 
                for meta in self.metadata_store.values()
                if meta.noun_class == noun_class]

    def get_agreement_paradigm(self, number: int) -> Dict[str, str]:
        """Get full agreement paradigm for a number across all noun classes."""
        if not NumeralAgreement.is_agreeable(number):
            return {}
        
        paradigm = {}
        for noun_class in NounClass:
            form = NumeralAgreement.get_agreement_form(number, noun_class)
            if form:
                paradigm[f"Class_{noun_class.number}_{noun_class.semantic_category}"] = form
        return paradigm

    def frequency_ranking(self) -> List[Tuple[str, int, int]]:
        """Get numerals ranked by frequency."""
        ranked = [(m.word, m.numeric_value, m.frequency_rank) 
                  for m in self.metadata_store.values() 
                  if m.frequency_rank > 0]
        return sorted(ranked, key=lambda x: x[2])

    def morphological_breakdown(self, word: str) -> Optional[Dict[str, Any]]:
        """Detailed morphological analysis."""
        metadata = self.get_metadata(word) # USES FIXED GET_METADATA
        if not metadata:
            return None
        
        return {
            "word": metadata.word,
            "type": metadata.morphological_type.name,
            "morphemes": metadata.morpheme_breakdown,
            "root": metadata.root,
            "structure": " + ".join(metadata.morpheme_breakdown) if metadata.morpheme_breakdown else metadata.word
        }

    def phonological_analysis(self, word: str) -> Optional[Dict[str, Any]]:
        """Detailed phonological analysis."""
        metadata = self.get_metadata(word) # USES FIXED GET_METADATA
        if not metadata:
            return None
        
        return {
            "word": metadata.word,
            "syllables": metadata.syllable_structure,
            "tone": metadata.tone_pattern,
            "features": [f.name for f in metadata.phonological_features],
            "has_gemination": PhonologicalFeature.GEMINATE_CONSONANT in metadata.phonological_features,
            "has_prenasalization": PhonologicalFeature.PRENASALIZED in metadata.phonological_features
        }

    def export_to_json(self, filepath: str = "luganda_numerals_research.json"):
        """Export all linguistic data to JSON for external analysis."""
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_entries": len(self.metadata_store),
                "system_version": "2.0-linguistic"
            },
            "numerals": {}
        }
        
        for word, meta in self.metadata_store.items():
            export_data["numerals"][word] = {
                "word": meta.word,
                "value": meta.numeric_value,
                "type": meta.symbol_type.name,
                "noun_class": meta.noun_class,
                "plural": meta.plural_form,
                "phonology": {
                    "syllables": meta.syllable_structure,
                    "tone": meta.tone_pattern,
                    "features": [f.name for f in meta.phonological_features]
                },
                "morphology": {
                    "type": meta.morphological_type.name,
                    "breakdown": meta.morpheme_breakdown,
                    "root": meta.root
                },
                "etymology": {
                    "proto_bantu": meta.etymology.proto_bantu_root if meta.etymology else None,
                    "cognates": meta.etymology.cognates if meta.etymology else {},
                    "notes": meta.etymology.notes if meta.etymology else ""
                },
                "frequency_rank": meta.frequency_rank,
                "usage_contexts": list(meta.usage_contexts),
                "examples": meta.example_sentences,
                "notes": meta.notes
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return filepath

    def generate_research_report(self) -> str:
        """Generate a comprehensive research report."""
        report = []
        report.append("=" * 80)
        report.append("LUGANDA NUMERICAL SYSTEM - LINGUISTIC RESEARCH REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overview statistics
        report.append("1. OVERVIEW STATISTICS")
        report.append("-" * 80)
        report.append(f"Total numeral entries: {len(self.metadata_store)}")
        
        by_type = defaultdict(int)
        for meta in self.metadata_store.values():
            by_type[meta.symbol_type.name] += 1
        
        report.append("\nDistribution by type:")
        for sym_type, count in sorted(by_type.items()):
            report.append(f"  {sym_type}: {count}")
        
        # Phonological features
        report.append("\n2. PHONOLOGICAL PATTERNS")
        report.append("-" * 80)
        
        geminate_words = self.get_by_feature(PhonologicalFeature.GEMINATE_CONSONANT)
        prenasalized_words = self.get_by_feature(PhonologicalFeature.PRENASALIZED)
        
        report.append(f"Geminate consonants ({len(geminate_words)}): {', '.join(geminate_words)}")
        report.append(f"Prenasalized ({len(prenasalized_words)}): {', '.join(prenasalized_words)}")
        
        # Noun class distribution
        report.append("\n3. NOUN CLASS DISTRIBUTION")
        report.append("-" * 80)
        
        for i in range(1, 11):
            words = self.get_by_noun_class(i)
            if words:
                noun_class = next((nc for nc in NounClass if nc.number == i), None)
                semantic = noun_class.semantic_category if noun_class else "unknown"
                report.append(f"Class {i} ({semantic}): {', '.join([w[0] for w in words])}")
        
        # Agreement paradigms
        report.append("\n4. AGREEMENT PARADIGMS")
        report.append("-" * 80)
        
        for num in [1, 2, 3, 4, 5]:
            report.append(f"\nNumber {num} ({self.BASE_NUMBERS[num]}):")
            paradigm = self.get_agreement_paradigm(num)
            for class_name, form in paradigm.items():
                report.append(f"  {class_name}: {form}")
        
        # Etymology summary
        report.append("\n5. ETYMOLOGICAL NOTES")
        report.append("-" * 80)
        
        for word in ['emu', 'bbiri', 'ssatu', 'nnya', 'ttaano']:
            meta = self.get_metadata(word) # USES FIXED GET_METADATA
            if meta and meta.etymology and meta.etymology.proto_bantu_root:
                report.append(f"{word} < Proto-Bantu {meta.etymology.proto_bantu_root}")
                if meta.etymology.cognates:
                    cognates = ', '.join([f"{lang}:{form}" for lang, form in meta.etymology.cognates.items()])
                    report.append(f"  Cognates: {cognates}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


# ============================================================================
# CURRENCY NORMALIZATION (from nums1.py)
# ============================================================================

class Currency(Enum):
    UGX = ("Ssiringi", NounClass.CLASS_9)
    USD = ("Ddoola", NounClass.CLASS_9)
    GBP = ("Pawundi", NounClass.CLASS_9)
    EUR = ("Yuro", NounClass.CLASS_9)
    
    @property
    def luganda_name(self) -> str: 
        return self.value[0]
    
    @property
    def noun_class(self) -> NounClass: 
        return self.value[1]


class CurrencyNormalizer:
    CURRENCY_PATTERNS = {
        re.compile(r'(?:UGX|Shs\.?|=/=|=)\s*([\d,]+(?:\.\d{1,2})?)'): Currency.UGX,
        re.compile(r'\$\s*([\d,]+(?:\.\d{1,2})?)'): Currency.USD,
        re.compile(r'£\s*([\d,]+(?:\.\d{1,2})?)'): Currency.GBP,
        re.compile(r'€\s*([\d,]+(?:\.\d{1,2})?)'): Currency.EUR,
    }

    def __init__(self, num_system: LugandaNumericalSystem):
        self.num_system = num_system

    def normalize(self, amount: float, currency: Currency) -> str:
        integer_part = int(amount)
        fractional_part = round((amount - integer_part) * 100)
        
        if 2 <= integer_part <= 5:
            number_str = {2: 'zibiri', 3: 'zisatu', 4: 'zina', 5: 'zitaano'}.get(integer_part)
        elif integer_part == 500:
            number_str = 'tutaano'
        else:
            number_str = self.num_system.number_to_luganda(integer_part, currency.noun_class)
        
        result = f"{currency.luganda_name} {number_str}"
        if fractional_part > 0:
            cents_str = self.num_system.number_to_luganda(fractional_part)
            result += f" n'ennusu {cents_str}"
        return result


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

def run_linguistic_demonstrations():
    """Demonstrate linguistic research capabilities."""
    print("\n" + "="*80)
    print("LUGANDA LINGUISTIC RESEARCH SYSTEM - DEMONSTRATION")
    print("="*80 + "\n")
    
    system = LugandaNumericalSystem()
    
    # 1. Word Analysis
    print("1. DETAILED WORD ANALYSIS")
    print("-" * 80)
    for word in ['bbiri', 'biri', 'mukaaga', 'kkumi', 'kumi']:
        analysis = system.analyze_word(word)
        print(f"\n{word.upper()}:")
        if 'error' in analysis:
            print(f"  ERROR: {analysis['error']}")
            continue
        print(f"  Value: {analysis['numeric_value']}")
        print(f"  Type: {analysis['type']}")
        print(f"  Syllables: {analysis['phonology']['syllable_structure']}")
        print(f"  Tone: {analysis['phonology']['tone_pattern']}")
        print(f"  Morphemes: {' + '.join(analysis['morphology']['breakdown']) if analysis['morphology']['breakdown'] else 'monomorphemic'}")
        if 'etymology' in analysis and analysis['etymology']['proto_bantu']:
            print(f"  Proto-Bantu: {analysis['etymology']['proto_bantu']}")
    
    # 2. Agreement Paradigms
    print("\n\n2. AGREEMENT PARADIGMS")
    print("-" * 80)
    for num in [1, 2, 3]:
        print(f"\nNumber {num} ({system.BASE_NUMBERS[num]}):")
        paradigm = system.get_agreement_paradigm(num)
        for class_info, form in list(paradigm.items())[:3]:
            print(f"  {class_info}: {form}")
    
    # 3. Phonological Features
    print("\n\n3. PHONOLOGICAL FEATURE ANALYSIS")
    print("-" * 80)
    geminate = system.get_by_feature(PhonologicalFeature.GEMINATE_CONSONANT)
    print(f"Words with geminate consonants: {', '.join(geminate[:5])}")
    
    prenasalized = system.get_by_feature(PhonologicalFeature.PRENASALIZED)
    print(f"Words with prenasalization: {', '.join(prenasalized)}")
    
    # 4. Noun Class Distribution
    print("\n\n4. NOUN CLASS DISTRIBUTION")
    print("-" * 80)
    for noun_class in [2, 4, 5, 7]:
        words = system.get_by_noun_class(noun_class)
        if words:
            print(f"Class {noun_class}: {', '.join([f'{w[0]} ({w[1]})' for w in words[:3]])}")
    
    # 5. Morphological Analysis
    print("\n\n5. MORPHOLOGICAL ANALYSIS")
    print("-" * 80)
    for word in ['mukaaga', 'kikumi', 'lukumi']:
        breakdown = system.morphological_breakdown(word)
        if breakdown:
            print(f"{word}: {breakdown['structure']} (root: {breakdown['root']})")
    
    # 6. Frequency Ranking
    print("\n\n6. FREQUENCY RANKING (Top 10)")
    print("-" * 80)
    ranking = system.frequency_ranking()[:10]
    for word, value, rank in ranking:
        print(f"{rank:2d}. {word:<15} = {value}")
    
    # 7. Generate Full Report
    print("\n\n7. GENERATING COMPREHENSIVE RESEARCH REPORT")
    print("-" * 80)
    report = system.generate_research_report()
    # print(report) # Muted for brevity
    print("Report generated (muted for brevity).")
    
    # 8. Test conversions
    print("\n\n8. NUMBER CONVERSION TESTS (L2N & N2L)")
    print("-" * 80)
    test_cases = [
        (1, "emu"),
        (11, "kkumi n'emu"),
        (15, "kkumi na ttaano"),
        (21, "abiri mu emu"),
        (100, "kikumi"),
        (200, "bibiri"),
        (1000, "lukumi"),
        (2000, "nkumi bbiri"),
        (2102, "nkumi bbiri mu kikumi mu bbiri"),
        (10000, "mutwalo gumu"),
        (1000000, "kakadde kamu")
    ]
    
    for num, luganda_str in test_cases:
        # Test N2L
        n2l_result = system.number_to_luganda(num)
        
        # Test L2N
        l2n_result = system.luganda_to_number(luganda_str)
        l2n_status = "✓" if l2n_result == num else "✗"
        print(f"{l2n_status} L2N: {luganda_str:<30} → {l2n_result}")

        if l2n_result != num:
             print(f"    ERROR: Expected {num} but got {l2n_result}")

    # Test complex L2N specifically
    print("\nTesting complex L2N parser:")
    complex_l2n = [
        ("kkumi na ttaano", 15),
        ("abiri mu emu", 21),
        ("nkumi bbiri mu kikumi mu abiri", 2120), # NOTE: 2102 would be '...mu bbiri'
        ("kumi na ttaano", 15), # Ungeminated
    ]
    for luganda_str, num in complex_l2n:
        l2n_result = system.luganda_to_number(luganda_str)
        l2n_status = "✓" if l2n_result == num else "✗"
        print(f"{l2n_status} L2N: {luganda_str:<30} → {l2n_result}")


if __name__ == "__main__":
    run_linguistic_demonstrations()