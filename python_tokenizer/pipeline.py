"""
Morpheme-Aware Tokenization Pipeline for Luganda
=================================================
This script bridges the gap between the morphological analyzer (_breakdown.py)
and the tokenizer (tokenizer.py), implementing the powerful pre-segmentation
approach.

Key Components:
1. MorphemeVocabExtractor: Converts analyzer patterns to tokenizer format
2. CorpusPreSegmenter: Pre-segments corpus using trained analyzer
3. MorphemeBPETokenizer: Simplified tokenizer for pre-segmented text
4. IntegratedPipeline: End-to-end orchestrator

Usage:
    # Step 1: Train analyzer
    pipeline = IntegratedPipeline()
    pipeline.train_analyzer("luganda_corpus.txt")
    
    # Step 2: Pre-segment corpus
    pipeline.pre_segment_corpus("luganda_corpus.txt", "corpus.segmented.txt")
    
    # Step 3: Train tokenizer
    pipeline.train_tokenizer("corpus.segmented.txt", vocab_size=5000)
    
    # Step 4: Use for inference
    tokens = pipeline.encode("omuntu akola")
"""

import json
import regex as re 
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from collections import Counter
import logging
# Removed unused multiprocessing import

# Assuming breakdown.py classes are available
try:
    from special_tokens import SpecialTokens

    from breakdown import (
        UnifiedMorphologyAnalyzer,
        EnhancedLugandaTextPreprocessor,
        MorphemeSegmenter
    )
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    logging.warning("Could not import analyzer. Install breakdown.py first.")

# Removed preprocessor.py dependency - using standalone implementation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── BPE helpers (module-level to avoid triplication) ─────────────────────────

def _bpe_get_stats(ids: List[int], stats: Optional[Dict] = None) -> Dict:
    """Count adjacent token-pair frequencies in a token sequence."""
    if stats is None:
        stats = {}
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def _bpe_merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace all occurrences of *pair* in *ids* with *new_id*."""
    new_ids: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ============================================================================
# COMPONENT 1: MORPHEME VOCABULARY EXTRACTOR
# ============================================================================

class MorphemeVocabExtractor:
    """
    Extracts morpheme vocabulary from trained analyzer in the format
    expected by the hybrid tokenizer.
    """
    
    @staticmethod
    def extract_from_analyzer(analyzer: 'UnifiedMorphologyAnalyzer',
                              min_confidence: float = 0.3) -> Dict[str, Set[str]]:
        """
        Extract morpheme vocabulary from trained analyzer.
        
        Args:
            analyzer: Trained UnifiedMorphologyAnalyzer instance
            min_confidence: Minimum confidence threshold for patterns
            
        Returns:
            Dictionary with 'prefixes', 'roots', 'suffixes' keys
        """
        morpheme_vocab = {
            'prefixes': set(),
            'roots': set(),
            'suffixes': set()
        }
        
        # Extract prefixes
        for prefix, pattern in analyzer.engine.prefix_patterns.items():
            if pattern.cohesion_score >= min_confidence:
                morpheme_vocab['prefixes'].add(prefix)
        
        # Extract roots
        for root, pattern in analyzer.engine.root_patterns.items():
            if pattern.cohesion_score >= min_confidence:
                morpheme_vocab['roots'].add(root)
        
        # Extract suffixes
        for suffix, pattern in analyzer.engine.suffix_patterns.items():
            if pattern.cohesion_score >= min_confidence:
                morpheme_vocab['suffixes'].add(suffix)
        
        logger.info(f"Extracted vocabulary: {len(morpheme_vocab['prefixes'])} prefixes, "
                   f"{len(morpheme_vocab['roots'])} roots, "
                   f"{len(morpheme_vocab['suffixes'])} suffixes")
        
        return morpheme_vocab
    
    @staticmethod
    def save_morpheme_vocab(morpheme_vocab: Dict[str, Set[str]], 
                           filepath: Path):
        """Save morpheme vocabulary to JSON file."""
        # Convert sets to sorted lists for JSON serialization
        serializable = {
            k: sorted(list(v)) for k, v in morpheme_vocab.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved morpheme vocabulary to {filepath}")
    
    @staticmethod
    def load_morpheme_vocab(filepath: Path) -> Dict[str, Set[str]]:
        """Load morpheme vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert lists back to sets
        return {k: set(v) for k, v in data.items()}


# ============================================================================
# COMPONENT 2: CORPUS PRE-SEGMENTER
# ============================================================================

class CorpusPreSegmenter:
    """Pre-segments corpus using trained morphological analyzer."""
    
    def __init__(self, analyzer: 'UnifiedMorphologyAnalyzer',
                 confidence_threshold: float = 0.3):
        self.analyzer = analyzer
        self.decomposition_cache = {}
        self.preprocessor = EnhancedLugandaTextPreprocessor(
            use_morphology=True,
            morphology_analyzer=analyzer,
            morphology_confidence_threshold=confidence_threshold
        )
        logger.info(f"Initialized CorpusPreSegmenter (threshold={confidence_threshold})")
    
    def segment_text(self, text: str) -> str:
        """Segment text with caching."""
        # BUG FIX: Use text itself as cache key instead of hash(text) (non-deterministic)
        cache_key = text
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        tokens = self.preprocessor.preprocess_text(text)
        result = " ".join(tokens)
        self.decomposition_cache[cache_key] = result
        return result
    
    def segment_file(self, input_path: Path, output_path: Path):
        """Segment corpus file (streaming)."""
        logger.info(f"Pre-segmenting: {input_path} -> {output_path}")
        
        # Add validation
        if not input_path.exists():
            raise FileNotFoundError(f"Input corpus not found: {input_path}")
            
        # Resolve paths
        input_path = input_path.resolve()
        output_path = output_path.resolve()
        
        # Prevent overwriting input
        if input_path == output_path:
            raise ValueError("Input and output paths cannot be the same") 
            
        line_count = 0
        total_tokens = 0
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                segmented = self.segment_text(line)
                f_out.write(segmented + "\n")
                
                line_count += 1
                total_tokens += len(segmented.split())
                
                if line_count % 1000 == 0:
                    f_out.flush()
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count:,} lines...")
        
        logger.info(f"✅ Complete: {line_count:,} lines, {total_tokens:,} tokens")
        if line_count > 0:
            logger.info(f"📊 Average: {total_tokens/line_count:.1f} tokens/line")
        else:
            logger.warning("Corpus produced zero non-empty lines — output is empty.")

# ============================================================================
# COMPONENT 3: MORPHEME-AWARE BPE TOKENIZER
# ============================================================================

class MorphemeBPETokenizer:

    def __init__(self):
        # ✅ FIX 2: Improved pattern that properly captures special tokens
        # Match special tokens (<|...|>) OR non-whitespace sequences
        special_token_pattern = r'<\|[a-zA-Z0-9_]+\|>'
        morpheme_pattern = r'\S+'
        combined_pattern = f'({special_token_pattern}|{morpheme_pattern})'

        self.pattern = combined_pattern
        self.special_tokens = SpecialTokens.get_all_tokens()
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        # Quick access
        self.MORPHEME_SEP = SpecialTokens.MORPH_SEP
        self.SPACE_TOKEN = SpecialTokens.SPACE
        self.NUM_START_TOKEN = SpecialTokens.NUM_START_TOKEN
        self.NUM_END_TOKEN = SpecialTokens.NUM_END_TOKEN
        self.DATE_START_TOKEN = SpecialTokens.DATE_START_TOKEN
        self.DATE_END_TOKEN = SpecialTokens.DATE_END_TOKEN
        self.TIME_START_TOKEN = SpecialTokens.TIME_START_TOKEN
        self.TIME_END_TOKEN = SpecialTokens.TIME_END_TOKEN
        self.UNK_TOKEN = SpecialTokens.UNKNOWN
        self._compiled_pattern = re.compile(self.pattern)
        logger.info("Initialized MorphemeBPETokenizer with improved pattern")

    def train_from_segmented(self, text: str, vocab_size: int, verbose: bool = True):
        """Train on pre-segmented text with proper special token handling."""
        logger.info(f"Training MorphemeBPE (target vocab: {vocab_size})")
        
        # Split into morphemes using improved pattern
        morphemes = re.findall(self._compiled_pattern, text)
        morpheme_freq = Counter(morphemes)
        
        logger.info(f"Found {len(morpheme_freq):,} unique morphemes")
        
        # ✅ FIX 4: Separate special tokens from regular morphemes
        special_token_strings = set(self.special_tokens.keys())
        regular_morphemes = {m: f for m, f in morpheme_freq.items() 
                           if m not in special_token_strings}
        
        logger.info(f"Regular morphemes: {len(regular_morphemes):,}")
        logger.info(f"Special tokens found in text: "
                   f"{len([m for m in morpheme_freq if m in special_token_strings])}")
        
        # Create base vocab from frequent regular morphemes only
        base_vocab_size = min(len(regular_morphemes), vocab_size // 2)
        frequent_morphemes = [
            m for m, _ in Counter(regular_morphemes).most_common(base_vocab_size)
        ]
        
        # Initialize vocabulary
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Add special tokens first (they get fixed high IDs)
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode('utf-8')
            self.inverse_vocab[token] = idx
        
        # Add base morphemes starting from ID 0
        # (This avoids collisions with special token IDs in the 100000 range)
        next_id = 0
        for morpheme in frequent_morphemes:
            # ✅ FIX 5: Skip if it's a special token (shouldn't happen, but safe)
            if morpheme in special_token_strings:
                continue
            
            # Skip if already in vocab (shouldn't happen with proper filtering)
            if morpheme in self.inverse_vocab:
                continue
            
            self.vocab[next_id] = morpheme.encode('utf-8')
            self.inverse_vocab[morpheme] = next_id
            next_id += 1
        
        logger.info(f"Base vocabulary: {len(self.vocab)} tokens")
        
        # Convert text to ID chunks
        text_chunks = text.split('\n')
        unk_id = self.special_tokens[self.UNK_TOKEN]
        
        ids_chunks = []
        for chunk in text_chunks:
            if not chunk.strip():
                continue
            
            chunk_ids = []
            chunk_morphemes = re.findall(self._compiled_pattern, chunk)
            
            for m in chunk_morphemes:
                if m in self.special_tokens:
                    chunk_ids.append(self.special_tokens[m])
                elif m in self.inverse_vocab:
                    chunk_ids.append(self.inverse_vocab[m])
                else:
                    chunk_ids.append(unk_id)
            
            if chunk_ids:
                ids_chunks.append(chunk_ids)
        
        logger.info(f"Converted corpus to {len(ids_chunks):,} chunks")
        
        # Perform BPE merges
        num_merges = vocab_size - len(self.vocab)
        if num_merges > 0:
            logger.info(f"Performing {num_merges} BPE merges...")
            self._run_bpe_merges(ids_chunks, num_merges, verbose)
        
        logger.info(f"✅ Training complete. Final vocab size: {len(self.vocab)}")
    
    def encode_ordinary(self, text: str) -> List[int]:
        """
        ✅ FIX 6: Improved encoding that properly handles special tokens.
        """
        if not text or not text.strip():
            return []

        # Split using improved pattern
        morpheme_chunks = re.findall(self._compiled_pattern, text)
        
        unk_id = self.special_tokens[self.UNK_TOKEN]
        ids = []
        
        # Convert morphemes to base IDs
        for m in morpheme_chunks:
            if m in self.special_tokens:
                # Special tokens get their fixed IDs
                ids.append(self.special_tokens[m])
            elif m in self.inverse_vocab:
                # Regular morphemes get their learned IDs
                ids.append(self.inverse_vocab[m])
            else:
                # Unknown morphemes
                ids.append(unk_id)
        
        # Apply BPE merges (but don't merge across special tokens)
        if not hasattr(self, 'merges') or not self.merges:
            return ids
        
        # ✅ FIX 7: Split on special tokens to avoid merging across boundaries
        special_token_ids = set(self.special_tokens.values())
        result = []
        current_segment = []
        
        for id in ids:
            if id in special_token_ids:
                # Process accumulated segment
                if current_segment:
                    result.extend(self._apply_merges(current_segment))
                    current_segment = []
                # Add special token
                result.append(id)
            else:
                current_segment.append(id)
        
        # Process final segment
        if current_segment:
            result.extend(self._apply_merges(current_segment))
        
        return result
    
    def _apply_merges(self, ids: List[int]) -> List[int]:
        """Apply BPE merges to a segment using module-level helpers."""
        max_iterations = len(ids) * 2  # Reasonable upper bound
        iteration = 0
        
        while len(ids) >= 2 and iteration < max_iterations:
            stats = _bpe_get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break
            
            new_id = self.merges[pair]
            old_len = len(ids)
            ids = _bpe_merge(ids, pair, new_id)
            
            # ✅ FIX 9: Safety check: merge should reduce length
            if len(ids) >= old_len:
                logger.warning(f"Merge didn't reduce sequence length at iteration {iteration}")
                break
            
            iteration += 1
        
        if iteration >= max_iterations:
            logger.warning(f"_apply_merges hit iteration limit ({max_iterations})")
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Improved decode that properly reconstructs text."""
        parts = []
        prev_was_morph_sep = False
        
        for token_id in ids:
            if token_id not in self.vocab:
                if not skip_special_tokens:
                    parts.append(self.UNK_TOKEN)
                continue
            
            token_bytes = self.vocab[token_id]
            token_str = token_bytes.decode('utf-8', errors='replace')
            
            # Check if it's a special token
            is_special = token_id in self.inverse_special_tokens
            
            if is_special:
                if token_str == self.SPACE_TOKEN:
                    parts.append(' ')
                    prev_was_morph_sep = False
                elif token_str == self.MORPHEME_SEP:
                    # ✅ FIX 11: Track morpheme separators but don't add them
                    prev_was_morph_sep = True
                    continue
                elif skip_special_tokens:
                    # Other special tokens are skipped
                    prev_was_morph_sep = False
                    continue
                else:
                    parts.append(token_str)
                    prev_was_morph_sep = False
            else:
                # ✅ FIX 11: Don't add space after morpheme separator
                if parts and not prev_was_morph_sep and parts[-1] != ' ':
                    # This is a new morpheme in same word, needs no separator
                    pass
                parts.append(token_str)
                prev_was_morph_sep = False
        
        # Join and clean up
        text = ''.join(parts)
        
        if skip_special_tokens:
            # Remove any remaining special token strings
            for special in self.special_tokens.keys():
                text = text.replace(special, '')
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    
    def _run_bpe_merges(self, ids_chunks: List[List[int]], num_merges: int, verbose: bool):
        """Perform BPE merging using module-level helpers."""
        self.merges = {}
        
        # Pre-compute reserved ID range
        reserved_ids = set(self.special_tokens.values())
        max_reserved = max(reserved_ids) if reserved_ids else 0
        
        for i in range(num_merges):
            stats: Dict = {}
            for chunk_ids in ids_chunks:
                _bpe_get_stats(chunk_ids, stats)
            
            if not stats:
                if verbose:
                    logger.info(f"No more pairs at iteration {i}")
                break
            
            pair = max(stats, key=stats.get)
            
            # Start from next available ID above current vocab size
            new_id = len(self.vocab)
            # Skip reserved IDs (special tokens live in 100000+ range)
            while new_id in reserved_ids:
                new_id += 1
            
            if new_id > max_reserved + 100000:
                logger.error(f"Token ID space exhausted at merge {i}")
                break
            
            # FIX: Guard against KeyError if pair members are not in vocab
            if pair[0] not in self.vocab or pair[1] not in self.vocab:
                logger.warning(f"Skipping merge {i}: token {pair} not in vocab")
                continue

            left_token  = self.vocab[pair[0]]
            right_token = self.vocab[pair[1]]
            merged_token_bytes = left_token + right_token
            
            self.vocab[new_id] = merged_token_bytes
            self.merges[pair]  = new_id
            
            ids_chunks = [_bpe_merge(chunk_ids, pair, new_id) for chunk_ids in ids_chunks]
            
            if verbose and i % 100 == 0:
                merged_str = merged_token_bytes.decode('utf-8', errors='replace')
                print(f"Merge {i+1}/{num_merges}: {pair} -> {new_id} "
                      f"('{merged_str}') freq={stats[pair]}")

# ============================================================================
# COMPONENT 4: INTEGRATED PIPELINE
# ============================================================================

class IntegratedPipeline:
    """
    End-to-end pipeline orchestrating analyzer training, corpus pre-segmentation,
    and tokenizer training.
    """
    
    def __init__(self, analyzer_config: Optional[Dict] = None):
        """
        Initialize integrated pipeline.
        
        Args:
            analyzer_config: Configuration overrides for analyzer
        """
        self.analyzer_config = analyzer_config or {}
        self.analyzer = None
        self.tokenizer = None
        self.pre_segmenter: Optional[CorpusPreSegmenter] = None
        
        logger.info("Initialized IntegratedPipeline")
    
    def train_analyzer(self, corpus_path: str, 
                      save_dir: Optional[str] = None,
                      inference_confidence: float = 0.3):
        """
        Step 1: Train morphological analyzer.
        
        Args:
            corpus_path: Path to raw Luganda corpus
            save_dir: Optional directory to save trained analyzer
            inference_confidence: Default confidence threshold for .encode()
        """
        if not HAS_ANALYZER:
            raise RuntimeError("Analyzer not available. Install _breakdown.py")
        
        logger.info("=" * 70)
        logger.info("STEP 1: TRAINING MORPHOLOGICAL ANALYZER")
        logger.info("=" * 70)
        
        self.analyzer = UnifiedMorphologyAnalyzer(
            config_overrides=self.analyzer_config,
            deterministic=True
        )
        
        self.analyzer.analyze_corpus_from_file(Path(corpus_path))
        self.analyzer.print_report()
        
        self.pre_segmenter = CorpusPreSegmenter(
            self.analyzer,
            confidence_threshold=inference_confidence
        )
        logger.info(f"✅ Analyzer trained and pre-segmenter initialized "
                   f"(inference confidence={inference_confidence})")
        
        # Optionally save
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            morpheme_vocab = MorphemeVocabExtractor.extract_from_analyzer(self.analyzer)
            MorphemeVocabExtractor.save_morpheme_vocab(
                morpheme_vocab, 
                save_path / "morpheme_vocab.json"
            )
            
            self.analyzer.engine.export_results(
                save_path / "morphology_analysis.json",
                format='json'
            )
            
            logger.info(f"✅ Analyzer saved to {save_dir}")
    
    def pre_segment_corpus(self, input_corpus: str, output_corpus: str,
                          confidence_threshold: float = 0.3):
        """
        Step 2: Pre-segment corpus using trained analyzer.
        
        Args:
            input_corpus: Path to raw corpus
            output_corpus: Path for segmented output
            confidence_threshold: Minimum confidence for morpheme splits
        """
        if not self.analyzer:
            raise RuntimeError("Analyzer not trained. Run train_analyzer() first")
        
        logger.info("=" * 70)
        logger.info("STEP 2: PRE-SEGMENTING CORPUS")
        logger.info("=" * 70)
        
        segmenter = CorpusPreSegmenter(
            self.analyzer,
            confidence_threshold=confidence_threshold
        )
        
        segmenter.segment_file(
            Path(input_corpus),
            Path(output_corpus)
        )
        
        logger.info(f"✅ Segmented corpus saved to {output_corpus}")
    
    def train_tokenizer_streaming(self, segmented_corpus: str, vocab_size: int = 5000,
                              chunk_size: int = 10_000_000):
        """Train tokenizer on large corpus by reading in chunks.

        Note: training itself still requires all text in memory; this method
        reads the file once in bounded chunks to limit peak RSS.
        """
        parts: List[str] = []
        with open(segmented_corpus, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                parts.append(chunk)
        text = ''.join(parts)
        self.tokenizer = MorphemeBPETokenizer()
        self.tokenizer.train_from_segmented(text, vocab_size, verbose=True)

    
    def train_tokenizer(self, segmented_corpus: str, vocab_size: int = 5000,
                       save_path: Optional[str] = None):
        """
        Step 3: Train BPE tokenizer on pre-segmented corpus.
        
        Args:
            segmented_corpus: Path to pre-segmented corpus
            vocab_size: Target vocabulary size
            save_path: Optional path to save trained tokenizer
        """
        logger.info("=" * 70)
        logger.info("STEP 3: TRAINING MORPHEME-BPE TOKENIZER")
        logger.info("=" * 70)

        MAX_CORPUS_SIZE = 1_000_000_000  # 1GB

        file_size = Path(segmented_corpus).stat().st_size
        if file_size > MAX_CORPUS_SIZE:
            raise ValueError(f"Corpus too large: {file_size} bytes")
            
        with open(segmented_corpus, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.tokenizer = MorphemeBPETokenizer()
        self.tokenizer.train_from_segmented(text, vocab_size, verbose=True)

        if save_path:
            # FIX: JSON does not support tuple keys. Convert (a, b) → "a,b" strings
            # and bytes vocab values to UTF-8 strings for serialisation.
            save_data = {
                'vocab': {
                    str(k): (v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v)
                    for k, v in self.tokenizer.vocab.items()
                },
                'inverse_vocab': self.tokenizer.inverse_vocab,
                # Convert tuple pair keys to "id1,id2" strings
                'merges': {
                    f"{pair[0]},{pair[1]}": new_id
                    for pair, new_id in self.tokenizer.merges.items()
                },
            }
            Path(save_path).write_text(json.dumps(save_data, ensure_ascii=False), encoding='utf-8')
            logger.info(f"✅ Tokenizer saved to {save_path}")

    def is_ready(self) -> bool:
        """Check if pipeline is fully trained and ready for encoding."""
        return (
            self.analyzer is not None and
            self.pre_segmenter is not None and
            self.tokenizer is not None
        )

    def encode(self, text: str) -> List[int]:
        """
        Encode text using the trained pipeline. This method requires the
        analyzer and tokenizer to be trained.
        
        Args:
            text: Raw input text
            
        Returns:
            List of token IDs
        """
        if not text or not text.strip():
            return []
        
        if not self.is_ready():
            missing = []
            if not self.analyzer: missing.append("analyzer")
            if not self.pre_segmenter: missing.append("pre_segmenter")
            if not self.tokenizer: missing.append("tokenizer")
            raise RuntimeError(
                f"Pipeline is not ready. Missing components: {', '.join(missing)}. "
                "Please run train_analyzer() and train_tokenizer() first."
            )
        
        segmented_text = self.pre_segmenter.segment_text(text)

        # BUG FIX: MorphemeBPETokenizer has no encode() method, use encode_ordinary()
        return self.tokenizer.encode_ordinary(segmented_text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not trained")
        
        return self.tokenizer.decode(ids)


# ============================================================================
# DEMO & USAGE EXAMPLES
# ============================================================================

def create_demo_corpus():
    """Create a sample Luganda corpus for demonstration."""
    sample_text = """
    omuntu abantu omwana abaana omukazi abakazi omusaija abasaija
    ekitabo ebitabo ekinyabo ebinya ennyumba amayumba
    akola bakola tukola mukola yakola akola bakola tukola
    alaba balaba tulaba mulaba yalaba alabye babye tulabbye
    agenda bagenda tugenda mugenda yagenda agende bagende
    asoma basoma tusoma musoma yasoma asome basome tusome
    anywa banywa tunywa munywa yanywa anywe banywe tunywe
    """ * 50
    
    corpus_path = Path("demo_luganda_corpus.txt")
    corpus_path.write_text(sample_text, encoding='utf-8')
    logger.info(f"✅ Created demo corpus: {corpus_path}")
    return corpus_path


def demo_full_pipeline():
    """Demonstrate the complete integrated pipeline."""
    print("\n" + "=" * 70)
    print("🚀 MORPHEME-AWARE TOKENIZATION PIPELINE DEMO")
    print("=" * 70 + "\n")
    
    corpus_path = create_demo_corpus()
    segmented_path = "demo_corpus.segmented.txt"
    
    try:
        config = {
            'min_word_freq': 2,
            'min_pattern_freq': 2,
            'num_iterations': 3,
            'prefer_long_roots': True,
            'use_pmi': True,
            'max_workers': 1
        }
        
        pipeline = IntegratedPipeline(analyzer_config=config)
        
        pipeline.train_analyzer(
            str(corpus_path),
            save_dir="./demo_analyzer",
            inference_confidence=0.3
        )
        
        pipeline.pre_segment_corpus(
            str(corpus_path),
            segmented_path,
            confidence_threshold=0.3
        )
        
        with open(segmented_path, 'r', encoding='utf-8') as f:
            sample_lines = [f.readline().strip() for _ in range(3)]
        print("\n📝 Sample segmented text:")
        for line in sample_lines:
            print(f"  {line}")
        
        pipeline.train_tokenizer(
            segmented_path,
            vocab_size=1000,
            save_path="./demo_tokenizer"
        )
        
        print("\n" + "=" * 70)
        print("TESTING TOKENIZER")
        print("=" * 70)
        
        test_sentences = [
            "omuntu akola",
            "abantu balaba ekitabo",
            "omwana asoma",
            "" # Test empty string
        ]
        
        for text in test_sentences:
            encoded = pipeline.encode(text)
            decoded = pipeline.decode(encoded)
            
            print(f"\nOriginal: '{text}'")
            print(f"Encoded:  {encoded[:10]}... ({len(encoded)} tokens)")
            print(f"Decoded:  '{decoded}'")
            
            original_norm = text.lower().strip()
            decoded_norm = decoded.lower().strip()
            
            print(f"Match:    {'✅' if original_norm == decoded_norm else '❌'}")
        
        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETE!")
        print("=" * 70)
        
    finally:
        if corpus_path.exists():
            corpus_path.unlink()
        if Path(segmented_path).exists():
            Path(segmented_path).unlink()


if __name__ == "__main__":
    demo_full_pipeline()