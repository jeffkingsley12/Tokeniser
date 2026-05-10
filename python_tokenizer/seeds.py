"""
Phonotactic and morpheme seeds for the Luganda tokenizer.
Ported from src/syllabifier_seed.c

These are the fundamental syllable patterns used to bootstrap the syllable table.
"""

# Core CV (consonant-vowel) patterns - the most common Luganda syllables
CV_SEEDS = [
    # Basic CV patterns
    "ba", "be", "bi", "bo", "bu",
    "ca", "ce", "ci", "co", "cu",
    "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu",
    "ga", "ge", "gi", "go", "gu",
    "ja", "je", "ji", "jo", "ju",
    "ka", "ke", "ki", "ko", "ku",
    "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu",
    "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "pu",
    "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su",
    "ta", "te", "ti", "to", "tu",
    "va", "ve", "vi", "vo", "vu",
    "wa", "we", "wi", "wo", "wu",
    "ya", "ye", "yi", "yo", "yu",
    "za", "ze", "zi", "zo", "zu",
]

# Labialized consonants (C + w)
LABIALIZED_SEEDS = [
    "bwa", "bwe", "bwi", "bwo", "bwu",
    "cwa", "cwe", "cwi", "cwo", "cwu",
    "dwa", "dwe", "dwi", "dwo", "dwu",
    "fwa", "fwe", "fwi", "fwo", "fwu",
    "gwa", "gwe", "gwi", "gwo", "gwu",
    "jwa", "jwe", "jwi", "jwo", "jwu",
    "kwa", "kwe", "kwi", "kwo", "kwu",
    "lwa", "lwe", "lwi", "lwo", "lwu",
    "mwa", "mwe", "mwi", "mwo", "mwu",
    "nwa", "nwe", "nwi", "nwo", "nwu",
    "pwa", "pwe", "pwi", "pwo", "pwu",
    "rwa", "rwe", "rwi", "rwo", "rwu",
    "swa", "swe", "swi", "swo", "swu",
    "twa", "twe", "twi", "two", "twu",
    "vwa", "vwe", "vwi", "vwo", "vwu",
    "zwa", "zwe", "zwi", "zwo", "zwu",
]

# Palatalized consonants (C + y)
PALATALIZED_SEEDS = [
    "bya", "bye", "byi", "byo", "byu",
    "cya", "cye", "cyi", "cyo", "cyu",
    "dya", "dye", "dyi", "dyo", "dyu",
    "fya", "fye", "fyi", "fyo", "fyu",
    "gya", "gye", "gyi", "gyo", "gyu",
    "jya", "jye", "jyi", "jyo", "jyu",
    "kya", "kye", "kyi", "kyo", "kyu",
    "lya", "lye", "lyi", "lyo", "lyu",
    "mya", "mye", "myi", "myo", "myu",
    "nya", "nye", "nyi", "nyo", "nyu",
    "pya", "pye", "pyi", "pyo", "pyu",
    "rya", "rye", "ryi", "ryo", "ryu",
    "sya", "sye", "syi", "syo", "syu",
    "tya", "tye", "tyi", "tyo", "tyu",
    "vya", "vye", "vyi", "vyo", "vyu",
    "zya", "zye", "zyi", "zyo", "zyu",
]

# Prenasalized stops and affricates (NC)
PRENASALIZED_SEEDS = [
    "mba", "mbe", "mbi", "mbo", "mbu",
    "mbwa", "mbwe", "mbwi", "mbwo", "mbwu",
    "mbya", "mbye", "mbyi", "mbyo", "mbyu",
    "nda", "nde", "ndi", "ndo", "ndu",
    "ndwa", "ndwe", "ndwi", "ndwo", "ndwu",
    "ndya", "ndye", "ndyi", "ndyo", "ndyu",
    "nga", "nge", "ngi", "ngo", "ngu",
    "ngwa", "ngwe", "ngwi", "ngwo", "ngwu",
    "ngya", "ngye", "ngyi", "ngyo", "ngyu",
    "nka", "nke", "nki", "nko", "nku",
    "nkwa", "nkwe", "nkwi", "nkwo", "nkwu",
    "nkya", "nkye", "nkyi", "nkyo", "nkyu",
    "nca", "nce", "nci", "nco", "ncu",
    "ncwa", "ncwe", "ncwi", "ncwo", "ncwu",
    "ncya", "ncye", "ncyi", "ncyo", "ncyu",
    "nja", "nje", "nji", "njo", "nju",
    "njwa", "njwe", "njwi", "njwo", "njwu",
    "njya", "njye", "njyi", "njyo", "njyu",
    "nsa", "nse", "nsi", "nso", "nsu",
    "nswa", "nswe", "nswi", "nswo", "nswu",
    "nsya", "nsye", "nsyi", "nsyo", "nsyu",
    "nza", "nze", "nzi", "nzo", "nzu",
    "nzwa", "nzwe", "nzwi", "nzwo", "nzwu",
    "nzya", "nzye", "nzyi", "nzyo", "nzyu",
    "nta", "nte", "nti", "nto", "ntu",
    "ntya", "ntye", "ntyi", "ntyo", "ntyu",
]

# Long vowel CVV patterns (vowel sequences)
LONG_VOWEL_SEEDS = [
    "baa", "bee", "bii", "boo", "buu",
    "caa", "cee", "cii", "coo", "cuu",
    "daa", "dee", "dii", "doo", "duu",
    "faa", "fee", "fii", "foo", "fuu",
    "gaa", "gee", "gii", "goo", "guu",
    "jaa", "jee", "jii", "joo", "juu",
    "kaa", "kee", "kii", "koo", "kuu",
    "laa", "lee", "lii", "loo", "luu",
    "maa", "mee", "mii", "moo", "muu",
    "naa", "nee", "nii", "noo", "nuu",
    "paa", "pee", "pii", "poo", "puu",
    "raa", "ree", "rii", "roo", "ruu",
    "saa", "see", "sii", "soo", "suu",
    "taa", "tee", "tii", "too", "tuu",
    "vaa", "vee", "vii", "voo", "vuu",
    "waa", "wee", "wii", "woo", "wuu",
    "yaa", "yee", "yii", "yoo", "yuu",
    "zaa", "zee", "zii", "zoo", "zuu",
    # CVVV for triple vowels (less common)
    "baaa", "beee", "baee", "beea",
]

# Pre-prenasalized with long vowels
PRENASALIZED_LONG_VOWEL_SEEDS = [
    "mbee", "mbii", "mbuu", "mbaa", "mboo",
    "ndee", "ndii", "nduu", "ndaa", "ndoo",
    "ngee", "ngii", "nguu", "ngaa", "ngoo",
    "nkee", "nkii", "nkuu", "nkaa", "nkoo",
    "ncee", "ncii", "ncuu", "ncaa", "ncoo",
    "njee", "njii", "njuu", "njaa", "njoo",
    "nsee", "nsii", "nsuu", "nsaa", "nsoo",
    "nzee", "nzii", "nzuu", "nzaa", "nzoo",
]

# Vowel-only syllables (rare but possible at boundaries)
VOWEL_SEEDS = [
    "a", "e", "i", "o", "u",
    "aa", "ee", "ii", "oo", "uu",
]

# Combine all phonotactic seeds
PHONO_SEEDS = (
    CV_SEEDS + 
    LABIALIZED_SEEDS + 
    PALATALIZED_SEEDS + 
    PRENASALIZED_SEEDS +
    LONG_VOWEL_SEEDS +
    PRENASALIZED_LONG_VOWEL_SEEDS +
    VOWEL_SEEDS
)

# Morpheme seeds - common Luganda morphemes
MORPHEME_SEEDS = [
    # Subject markers
    "n", "o", "a", "tu", "mu", "ba", "a", "si", "te",
    # Object markers  
    "n", "ku", "mu", "wa", "gi", "ki", "bi", "zi", "li", "ya",
    # Tense/aspect markers
    "na", "ka", "sa", "a", "e", "o", "ng'a",
    # Negation
    "si", "ta",
    # Locative
    "mu", "ku", "e", "wa", "wo", "ya", "kyo", "byo", "gyo", "zzo",
    # Common verb roots
    "gul", "lya", "nywa", "zanya", "laba", "tegeera", "kola",
    "genda", "jja", "nya", "fa", "saba", "wa", "ba",
    # Common noun prefixes
    "o", "e", "a", "bu", "gu", "lu", "n", "tu", "mu", "ki", "bi", 
    "ma", "mi", "ka", "ba", "lu", "zi", "ti", "gi",
    # Common roots
    "ntu", "dd", "si", "bb", "jj", "jj", "gg", "dd", "vv", "zz",
    # Question words
    "ki", "ani", "wa", "lwaki", "nga", "bwe", "si", "nti",
    # Connectives
    "na", "n", "ne", "oba", "n""nga", "nti", "bwe", "nga", "singa",
    # Demonstratives
    "o", "nu", "no", "lyo", "ryo", "byo", "gyo", "yo", "zo", "wo",
    "gwo", "bwo", "kyo", "lwa", "bwa", "kwa", "mwa", "nnya",
    # Possessives
    "wa", "we", "we", "ya", "ye", "ye", "lya", "lye", "bye", "bye",
    "gye", "zze", "bbe", "nge", "se", "nge", "mwe",
    # Quantifiers
    "na", "onna", "ke", "nna", "ba", "bba", "ddala",
    # Adjective roots
    "nene", "to", "ggi", "ggya", "mpi", "bbi", "ddala", "ddagala",
]

# Ensure all seeds are lowercase and unique
PHONO_SEEDS = list(dict.fromkeys([s.lower() for s in PHONO_SEEDS]))
MORPHEME_SEEDS = list(dict.fromkeys([s.lower() for s in MORPHEME_SEEDS]))

# Statistics for verification
PHONO_SEED_COUNT = len(PHONO_SEEDS)
MORPHEME_SEED_COUNT = len(MORPHEME_SEEDS)
