#!/usr/bin/env python3
"""
Example usage of the Luganda Tokenizer Python bindings.

This example demonstrates loading and using a pre-trained tokenizer.
To train a new model, use the C tools:

    $ cd /path/to/Tokeniser
    $ make tokenizer_demo
    $ ./tokenizer_demo --train corpus.txt --save model.bin

Then load the model in Python:

    $ python example.py
"""

from python_tokenizer import LugandaTokenizer
import sys
import os


def main():
    print("=" * 60)
    print("Luganda Tokenizer - Python FFI Example")
    print("=" * 60)

    # Find or create a model file
    model_paths = [
        "production_model.bin",
        "model.bin",
        "luganda_tok.bin",
    ]

    model_path = None
    for p in model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("\nNo model file found. Please train one first:")
        print("    $ make tokenizer_demo")
        print("    $ ./tokenizer_demo --train corpus.txt --save model.bin")
        print("\nOr download a pre-trained model to one of these locations:")
        print(f"    {', '.join(model_paths)}")
        sys.exit(1)

    print(f"\n1. Loading pre-trained model from: {model_path}")
    tokenizer = LugandaTokenizer.load(model_path)
    print(f"   Model loaded successfully")
    
    print("\n2. Encoding text...")
    text = "bw'osoma oluganda luno"
    try:
        tokens = tokenizer.encode(text)
        print(f"   Input: '{text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Token count: {len(tokens)}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    print("\n3. Decoding back to text...")
    try:
        decoded = tokenizer.decode(tokens)
        print(f"   Decoded: '{decoded}'")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n4. Without special tokens...")
    try:
        tokens_no_special = tokenizer.encode(text, add_special_tokens=False)
        print(f"   Tokens: {tokens_no_special}")
        decoded_no_special = tokenizer.decode(tokens_no_special, skip_special_tokens=False)
        print(f"   Decoded: '{decoded_no_special}'")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n5. Batch encoding...")
    texts = [
        "nze ndi mu Uganda",
        "tuli wano",
        "oluganda lwe ggwanga",
    ]
    try:
        token_lists = tokenizer.encode_batch(texts, num_workers=2)
        for t, toks in zip(texts, token_lists):
            print(f"   '{t}' -> {len(toks)} tokens")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n6. Vocabulary inspection...")
    try:
        vocab = tokenizer.get_vocab()
        special_tokens = {k: v for k, v in vocab.items() if k.startswith('<') or k.startswith('[')}
        print(f"   Special tokens: {special_tokens}")
        sample_keys = list(vocab.keys())[5:10]
        print(f"   Sample tokens: {sample_keys}")
    except Exception as e:
        print(f"   Note: get_vocab() may be slow or unavailable: {e}")

    print("\n7. Round-trip verification...")
    test_text = "abantu"
    try:
        original_tokens = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(original_tokens)
        print(f"   Original: '{test_text}'")
        print(f"   Tokens: {original_tokens}")
        print(f"   Decoded: '{decoded_text}'")
        # Note: perfect round-trip is not guaranteed due to tokenization boundaries
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n8. Performance test...")
    import time

    corpus = [
        "oluganda lwe ggwanga lyaffe",
        "nze mbaagala okusoma olupapula",
        "abantu ba Uganda balina enjawulo",
        "omwana asoma ebitabo by'essomero",
        "tulina emmere n'ebintu by'okunywa",
    ]
    test_texts = corpus * 20  # 100 sentences

    start = time.time()
    try:
        for _ in range(10):  # 10 iterations
            for text in test_texts:
                tokenizer.encode(text, add_special_tokens=False)
        elapsed = time.time() - start

        total_chars = sum(len(t) for t in test_texts) * 10
        ops_per_sec = total_chars / elapsed if elapsed > 0 else 0
        print(f"   Processed {total_chars:,} characters in {elapsed:.2f}s")
        print(f"   Throughput: {ops_per_sec:,.0f} chars/sec")
    except Exception as e:
        print(f"   Error during performance test: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print("\nTo train your own model, run:")
    print("    $ make tokenizer_demo")
    print("    $ ./tokenizer_demo --train corpus.txt --save model.bin")
    print("\nThen load it in Python:")
    print("    >>> from python_tokenizer import LugandaTokenizer")
    print("    >>> t = LugandaTokenizer.load('model.bin')")


if __name__ == "__main__":
    main()
