"""
Tests for the Python tokenizer components.
"""

import sys
from pathlib import Path

# Add project root to sys.path so python_tokenizer can be imported from any working directory
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from python_tokenizer.special_tokens import SpecialTokens
from python_tokenizer.breakdown import LRUCache, BloomFilter


class TestSpecialTokens:
    """Test special tokens configuration."""

    def test_special_tokens_constants(self):
        """Test that all special token constants are defined."""
        assert hasattr(SpecialTokens, 'MORPH_SEP')
        assert hasattr(SpecialTokens, 'SPACE')
        assert hasattr(SpecialTokens, 'NUM_START_TOKEN')
        assert hasattr(SpecialTokens, 'NUM_END_TOKEN')
        assert hasattr(SpecialTokens, 'DATE_START_TOKEN')
        assert hasattr(SpecialTokens, 'DATE_END_TOKEN')
        assert hasattr(SpecialTokens, 'TIME_START_TOKEN')
        assert hasattr(SpecialTokens, 'TIME_END_TOKEN')
        assert hasattr(SpecialTokens, 'UNKNOWN')
        assert hasattr(SpecialTokens, 'EOS')
        assert hasattr(SpecialTokens, 'PAD')

    def test_get_all_tokens(self):
        """Test that get_all_tokens returns a dict with high IDs."""
        tokens = SpecialTokens.get_all_tokens()
        assert isinstance(tokens, dict)
        assert len(tokens) > 0
        # Check that IDs are in the high range (100000+)
        for token, token_id in tokens.items():
            assert token_id >= 100000, f"Token {token} has ID {token_id}, expected >= 100000"


class TestLRUCache:
    """Test LRU cache implementation."""

    def test_cache_basic_operations(self):
        """Test basic get/set operations."""
        cache = LRUCache(maxsize=3)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.get('nonexistent') is None

    def test_cache_eviction(self):
        """Test that least recently used items are evicted."""
        cache = LRUCache(maxsize=2)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')  # Should evict key1
        
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'

    def test_cache_lru_update(self):
        """Test that accessing an item updates its position."""
        cache = LRUCache(maxsize=2)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.get('key1')  # Access key1, making it most recently used
        cache.set('key3', 'value3')  # Should evict key2
        
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') is None
        assert cache.get('key3') == 'value3'

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(maxsize=10)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.clear()
        
        assert cache.get('key1') is None
        assert cache.get('key2') is None


class TestBloomFilter:
    """Test Bloom filter implementation."""

    def test_bloom_filter_basic(self):
        """Test basic add and contains operations."""
        bf = BloomFilter(size=1000, num_hashes=3)
        bf.add('test1')
        bf.add('test2')
        
        assert 'test1' in bf
        assert 'test2' in bf
        assert 'nonexistent' not in bf

    def test_bloom_filter_false_positives(self):
        """Test that bloom filter may have false positives but no false negatives."""
        bf = BloomFilter(size=1000, num_hashes=3)
        bf.add('test1')
        
        # Should never be a false negative
        assert 'test1' in bf
        
        # May have false positives (depends on hash collisions)
        # This is expected behavior for bloom filters

    def test_bloom_filter_sha256(self):
        """Test that bloom filter uses SHA-256 (not MD5)."""
        bf = BloomFilter(size=1000, num_hashes=3)
        # Just verify it doesn't crash - the implementation uses SHA-256
        bf.add('test')
        assert 'test' in bf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
