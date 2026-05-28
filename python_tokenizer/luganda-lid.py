"""
Luganda Language Identification (LID) Pipeline (Final Correction)
Handles multilingual text common in Ugandan social media (Luganda, English, Swahili, Runyankore)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# If fastText is available (install: pip install fasttext)
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("⚠️  fastText not installed. Using fallback heuristic method.")
    print("   Install with: pip install fasttext")


class LanguageLabel(Enum):
    """Language labels for Ugandan context"""
    LUGANDA = "lg"
    ENGLISH = "en"
    SWAHILI = "sw"
    RUNYANKORE = "nyn"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class LanguageDetectionResult:
    """Result of language detection"""
    text: str
    primary_language: str
    confidence: float
    is_code_switched: bool
    language_distribution: Dict[str, float]
    needs_review: bool
    reason: str


class LugandaLID:
    """Language Identification for Luganda with code-switching detection"""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.7):
        self.threshold = threshold
        self.model = None

        if FASTTEXT_AVAILABLE and model_path:
            try:
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(model_path)
                print(f"✅ Loaded fastText model: {model_path}")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
        
        # --- PATTERN REFINEMENT (THE FINAL FIX) ---
        # Made patterns more specific to avoid cross-language conflicts.

        self.luganda_patterns = {
            # BUG FIX: Removed 'en' prefix (ambiguous with English)
            'prefixes': r'\b(eki|eby|omu|aba|olu|ama|aka|otu|obu|oku)\w+',
            # Added more high-frequency, unambiguous Luganda words.
            'common_words': r'\b(nga|bwe|naye|era|anti|kati|wano|omwana|abantu|nze|ggwe|ku|mu|wa)\b',
            'verbs': r'\b(agenda|ajja|alina|balina|yakola|bakola|tugenda)\b'
        }

        self.english_patterns = {
            'articles': r'\b(the|a|an)\b',
            # 'going' and 'know' are good additions.
            'common': r'\b(is|are|was|were|have|has|do|does|will|would|can|could|going|know|today|market)\b',
            'pronouns': r'\b(I|you|he|she|it|we|they|my|your|his|her)\b'
        }

        self.swahili_patterns = {
            # REMOVED ambiguous prefixes like 'ku', 'mu'. Kept more distinct ones.
            'prefixes': r'\b(wa|ki|vi|mji)\w+',
            # Using more distinct and longer words. REMOVED 'na', 'ni', 'wa' which are too common.
            'common': r'\b(sana|wewe|mzuri|nakupenda|karibu|jina|habari|kwa|lakini)\b'
        }

    # The rest of the script's logic is now correct and does not need changes.
    def detect_language_fasttext(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """ Detect language using fastText model """
        if not self.model: return ("unknown", 0.0, {})
        cleaned = text.replace('\n', ' ').strip()
        if not cleaned: return ("unknown", 0.0, {})
        labels, scores = self.model.predict(cleaned, k=3)
        primary_lang = labels[0].replace('__label__', '')
        confidence = float(scores[0])
        distribution = {l.replace('__label__', ''): float(s) for l, s in zip(labels, scores)}
        return (primary_lang, confidence, distribution)

    def detect_language_heuristic(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """ Heuristic-based detection that returns a full distribution. """
        text_lower = text.lower()
        scores = {'lg': 0.0, 'en': 0.0, 'sw': 0.0}
        for pattern in self.luganda_patterns.values():
            scores['lg'] += len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in self.english_patterns.values():
            scores['en'] += len(re.findall(pattern, text_lower, re.IGNORECASE))
        for pattern in self.swahili_patterns.values():
            scores['sw'] += len(re.findall(pattern, text_lower, re.IGNORECASE))
        total = sum(scores.values())
        if total == 0: return ("unknown", 0.0, {})
        distribution = {lang: score / total for lang, score in scores.items()}
        primary_lang, confidence = max(distribution.items(), key=lambda x: x[1])
        return (primary_lang, confidence, distribution)

    def detect_code_switching(self, text: str, overall_distribution: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """ Detect code-switching using sentence-level analysis or intra-sentence mixing. """
        significant_langs = [lang for lang, score in overall_distribution.items() if score > 0.25]
        if len(significant_langs) > 1:
            return (True, overall_distribution)
        sentences = re.split(r'[.!?\n]+', text)
        if len(sentences) <= 1:
            return (False, overall_distribution)
        lang_counts: Dict[str, int] = {}
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            lang, _, _ = (self.detect_language_fasttext(sentence)
                          if self.model else self.detect_language_heuristic(sentence))
            if lang != 'unknown':
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        is_code_switched = len(lang_counts) > 1
        # FIX: Normalise sentence counts to proportions so callers always
        # receive a Dict[str, float] that sums to ~1.0, consistent with the
        # return type from detect_language_fasttext / detect_language_heuristic.
        # Previously this returned raw integer counts which broke downstream
        # formatting (e.g. `f"{v:.2f}"`) and confidence comparisons.
        total_sentences = sum(lang_counts.values())
        if total_sentences > 0:
            normalised: Dict[str, float] = {
                lang: count / total_sentences
                for lang, count in lang_counts.items()
            }
        else:
            normalised = overall_distribution
        return (is_code_switched, normalised)

    def process_text(self, text: str) -> LanguageDetectionResult:
        """ Complete LID pipeline with code-switching detection """
        if self.model:
            primary_lang, confidence, distribution = self.detect_language_fasttext(text)
        else:
            primary_lang, confidence, distribution = self.detect_language_heuristic(text)
        is_code_switched, final_distribution = self.detect_code_switching(text, distribution)
        needs_review = False
        reason = ""
        if is_code_switched:
            primary_lang = "mixed"
            needs_review = True
            reason = "Code-switching detected"
        elif confidence < self.threshold:
            needs_review = True
            reason = f"Low confidence ({confidence:.2f} < {self.threshold})"
        elif primary_lang not in ['lg', 'en', 'sw', 'nyn']:
            needs_review = True
            reason = f"Unexpected language: {primary_lang}"
        return LanguageDetectionResult(text=text, primary_language=primary_lang, confidence=confidence, is_code_switched=is_code_switched, language_distribution=final_distribution, needs_review=needs_review, reason=reason)

    def filter_luganda_corpus(self, texts: List[str], min_confidence: float = 0.7, allow_code_switching: bool = False) -> Dict[str, List[Tuple[str, LanguageDetectionResult]]]:
        results = {'accepted': [], 'code_switched': [], 'low_confidence': [], 'wrong_language': []}
        for text in texts:
            result = self.process_text(text)
            if result.is_code_switched:
                # Code-switched items go only to code_switched — they are not
                # additionally added to low_confidence.  If the caller needs to
                # find low-confidence code-switched items, they can filter the
                # code_switched bucket themselves.
                results['code_switched'].append((text, result))
            elif result.primary_language == 'lg' and result.confidence >= min_confidence:
                results['accepted'].append((text, result))
            elif result.confidence < min_confidence:
                results['low_confidence'].append((text, result))
            else:
                results['wrong_language'].append((text, result))
        return results


def main():
    """Demonstration of the LID system"""
    print("=" * 80)
    print("LUGANDA LANGUAGE IDENTIFICATION PIPELINE (FINAL CORRECTED)")
    print("=" * 80)
    print()
    lid = LugandaLID(threshold=0.7)
    test_samples = [
        "Omwana wange agenda ku ssomero buli lunaku.",
        "I am going to the market today to buy some food.",
        "Nze I'm going ku market naye simanyi whether bazze.",
        "Tugenda ku party nga Friday era tujja ku club.",
        "Nakupenda sana na wewe ni mzuri.",
        "You know nze I really love Kampala city kuba it has abantu abangi.",
        "Eggulu lirabika nga limyufu era enkuba ejja kutonnya mu budde obutonotono.",
    ]
    print("PROCESSING INDIVIDUAL SAMPLES:")
    print("-" * 80)
    for i, text in enumerate(test_samples, 1):
        result = lid.process_text(text)
        print(f"\nSample {i}:")
        print(f"Text: {text}")
        print(f"Primary Language: {result.primary_language}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Code-switched: {result.is_code_switched}")
        if result.language_distribution:
            dist_str = ", ".join([f"{k}: {v:.2f}" for k, v in result.language_distribution.items()])
            print(f"Distribution: {{{dist_str}}}")
        if result.needs_review:
            print(f"⚠️  NEEDS REVIEW: {result.reason}")
        else:
            print("✅ ACCEPTED")
    print("\n" + "=" * 80)
    print("CORPUS FILTERING:")
    print("-" * 80)
    filtered = lid.filter_luganda_corpus(test_samples, min_confidence=0.7)
    print(f"\n✅ Accepted (Pure Luganda): {len(filtered['accepted'])}")
    for text, result in filtered['accepted']:
        print(f"   - {text[:60]}... (conf: {result.confidence:.2f})")
    print(f"\n🔄 Code-switched: {len(filtered['code_switched'])}")
    for text, result in filtered['code_switched']:
        dist_str = ", ".join([f"{k}: {v:.2f}" for k, v in result.language_distribution.items()])
        print(f"   - {text[:60]}... {{{dist_str}}}")
    print(f"\n⚠️  Low Confidence: {len(filtered['low_confidence'])}")
    for text, result in filtered['low_confidence']:
        print(f"   - {text[:60]}... (conf: {result.confidence:.2f})")
    print(f"\n❌ Wrong Language: {len(filtered['wrong_language'])}")
    for text, result in filtered['wrong_language']:
        print(f"   - {text[:60]}... (detected: {result.primary_language})")

if __name__ == "__main__":
    main()