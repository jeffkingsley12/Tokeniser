#!/usr/bin/env python3
"""
unicode_table_generator.py

Generate compact two-level Unicode property tables for fast O(1) lookup.
Produces C header with:
  - unicode_page_index[256] : maps high byte to page index
  - unicode_props[][256]    : property bytes per codepoint

Properties (packed into uint8_t):
  U_PROP_LETTER   = 1 << 0  # Lu, Ll, Lt, Lm, Lo
  U_PROP_VOWEL    = 1 << 1  # Luganda-specific (a,e,i,o,u + accented)
  U_PROP_EXTEND   = 1 << 2  # Mn, Mc, Me (combining marks)
  U_PROP_ZWJ      = 1 << 3  # U+200D Zero Width Joiner
  U_PROP_EMOJI    = 1 << 4  # Emoji base from emoji-data.txt
  U_PROP_RI       = 1 << 5  # Regional Indicator (U+1F1E6..U+1F1FF)
  U_PROP_SYMBOL   = 1 << 6  # Sm, Sc, Sk, So (math, currency, etc.)
  U_PROP_PUNCT    = 1 << 7  # Pc, Pd, Ps, Pe, Pi, Pf, Po
"""

import sys
import urllib.request
import re
from pathlib import Path

# Property bit flags
U_PROP_LETTER = 1 << 0
U_PROP_VOWEL = 1 << 1
U_PROP_EXTEND = 1 << 2
U_PROP_ZWJ = 1 << 3
U_PROP_EMOJI = 1 << 4
U_PROP_RI = 1 << 5
U_PROP_SYMBOL = 1 << 6
U_PROP_PUNCT = 1 << 7

# Luganda vowels (ASCII + accented variants)
LUGANDA_VOWELS = set('aeiouAEIOU')
LUGANDA_VOWEL_ACCENTS = {
    0x00E1,  # á
    0x00E9,  # é
    0x00ED,  # í
    0x00F3,  # ó
    0x00FA,  # ú
}


def fetch_unicode_data():
    """Fetch UnicodeData.txt from unicode.org or use local cache."""
    urls = [
        "https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt",
        "https://unicode.org/Public/UCD/latest/ucd/UnicodeData.txt",
    ]
    cache_path = Path("/tmp/UnicodeData.txt")
    
    if cache_path.exists():
        return cache_path.read_text()
    
    last_error = None
    for url in urls:
        try:
            print(f"Fetching {url}...", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            cache_path.write_text(data)
            return data
        except Exception as e:
            last_error = e
            continue
    
    # All URLs failed
    print(f"❌ Could not fetch UnicodeData.txt from any source", file=sys.stderr)
    print(f"   Last error: {last_error}", file=sys.stderr)
    print(f"   Cache location: {cache_path}", file=sys.stderr)
    print(f"   To fix: manually download and cache the file to {cache_path}", file=sys.stderr)
    return None


def fetch_emoji_data():
    """Fetch emoji-data.txt from unicode.org or use local cache."""
    # Try multiple URLs in order of preference
    urls = [
        "https://www.unicode.org/Public/UNIDATA/emoji/emoji-data.txt",
        "https://unicode.org/Public/UNIDATA/emoji/emoji-data.txt",
        "https://www.unicode.org/Public/emoji/latest/emoji-data.txt",
    ]
    cache_path = Path("/tmp/emoji-data.txt")
    
    if cache_path.exists():
        return cache_path.read_text()
    
    last_error = None
    for url in urls:
        try:
            print(f"Fetching {url}...", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            cache_path.write_text(data)
            return data
        except Exception as e:
            last_error = e
            continue
    
    # All URLs failed
    print(f"⚠️  Could not fetch emoji-data.txt from any source", file=sys.stderr)
    print(f"   Last error: {last_error}", file=sys.stderr)
    print(f"   Tried URLs:", file=sys.stderr)
    for url in urls:
        print(f"     - {url}", file=sys.stderr)
    print(f"   Emoji properties will not be included.", file=sys.stderr)
    print(f"   Cache location: {cache_path}", file=sys.stderr)
    return None


def parse_unicode_data(text):
    """Parse UnicodeData.txt and return dict of codepoint -> property bits."""
    props = {}
    
    for line in text.strip().split('\n'):
        if not line or line.startswith('#'):
            continue
        
        fields = line.split(';')
        if len(fields) < 3:
            continue
        
        cp_str, name, cat = fields[0], fields[1], fields[2]
        cp = int(cp_str, 16)
        
        # Determine properties from category
        prop = 0
        
        # Letter categories
        if cat in ('Lu', 'Ll', 'Lt', 'Lm', 'Lo'):
            prop |= U_PROP_LETTER
        
        # Mark categories (combining)
        if cat in ('Mn', 'Mc', 'Me'):
            prop |= U_PROP_EXTEND
        
        # Symbol categories
        if cat in ('Sm', 'Sc', 'Sk', 'So'):
            prop |= U_PROP_SYMBOL
        
        # Punctuation categories
        if cat in ('Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po'):
            prop |= U_PROP_PUNCT
        
        # Special cases
        if cp == 0x200D:  # ZWJ
            prop |= U_PROP_ZWJ
        
        # Regional Indicators
        if 0x1F1E6 <= cp <= 0x1F1FF:
            prop |= U_PROP_RI
        
        # Luganda vowels (ASCII)
        if chr(cp) in LUGANDA_VOWELS:
            prop |= U_PROP_VOWEL
        
        # Luganda accented vowels
        if cp in LUGANDA_VOWEL_ACCENTS:
            prop |= U_PROP_VOWEL | U_PROP_LETTER
        
        props[cp] = prop
    
    return props


def parse_emoji_data(text, props):
    """Parse emoji-data.txt and add EMOJI property to relevant codepoints."""
    for line in text.strip().split('\n'):
        if not line or line.startswith('#'):
            continue
        
        # Format: codepoint(s) ; property # name
        parts = line.split('#')[0].strip()
        if not parts:
            continue
        
        fields = parts.split(';')
        if len(fields) < 2:
            continue
        
        cp_field = fields[0].strip()
        prop_field = fields[1].strip()
        
        # We care about Emoji and Emoji_Presentation
        if 'Emoji' not in prop_field:
            continue
        
        # Parse codepoint or range
        if '..' in cp_field:
            start, end = cp_field.split('..')
            cp_start = int(start, 16)
            cp_end = int(end, 16)
            for cp in range(cp_start, cp_end + 1):
                props[cp] = props.get(cp, 0) | U_PROP_EMOJI
        else:
            cp = int(cp_field, 16)
            props[cp] = props.get(cp, 0) | U_PROP_EMOJI


def generate_tables(props):
    """Generate compact two-level lookup tables."""
    # Group by high byte (page)
    pages = {}
    for cp, prop in props.items():
        page_idx = cp >> 8
        offset = cp & 0xFF
        if page_idx not in pages:
            pages[page_idx] = [0] * 256
        pages[page_idx][offset] = prop
    
    # Create page index table
    # Strategy: most pages are empty or sparse; use a compact representation
    # where we deduplicate identical pages
    
    unique_pages = []
    page_to_idx = {}
    page_index = [0] * 256  # Maps high byte to unique page index
    
    for high_byte in range(256):
        if high_byte in pages:
            page_data = tuple(pages[high_byte])
            if page_data not in page_to_idx:
                page_to_idx[page_data] = len(unique_pages)
                unique_pages.append(page_data)
            page_index[high_byte] = page_to_idx[page_data]
        else:
            # Empty page - use page 0 (which should be all zeros)
            page_index[high_byte] = 0
    
    # Ensure page 0 is the zero page
    if not unique_pages or unique_pages[0] != tuple([0] * 256):
        # Insert zero page at beginning and shift indices
        unique_pages.insert(0, tuple([0] * 256))
        new_page_index = []
        for idx in page_index:
            new_page_index.append(idx + 1 if idx > 0 else 0)
        page_index = new_page_index
    
    return page_index, unique_pages


def emit_c_header(page_index, unique_pages, output_path):
    """Emit C header file with the tables."""
    lines = [
        "/* Auto-generated Unicode property tables */",
        "/* Generated by unicode_table_generator.py */",
        "#ifndef UNICODE_PROPS_H",
        "#define UNICODE_PROPS_H",
        "#include <stdint.h>",
        "",
        "/* Property bit flags */",
        "#define U_PROP_LETTER   (1u << 0)  /* Lu, Ll, Lt, Lm, Lo */",
        "#define U_PROP_VOWEL    (1u << 1)  /* Luganda vowels */",
        "#define U_PROP_EXTEND   (1u << 2)  /* Mn, Mc, Me (combining marks) */",
        "#define U_PROP_ZWJ      (1u << 3)  /* U+200D Zero Width Joiner */",
        "#define U_PROP_EMOJI    (1u << 4)  /* Emoji base/presentation */",
        "#define U_PROP_RI       (1u << 5)  /* Regional Indicator U+1F1E6..FF */",
        "#define U_PROP_SYMBOL   (1u << 6)  /* Sm, Sc, Sk, So */",
        "#define U_PROP_PUNCT    (1u << 7)  /* Punctuation */",
        "",
        f"#define U_PAGE_COUNT {len(unique_pages)}",
        "",
        "/* Level 1: Page index table (maps high byte to page index) */",
        "static const uint16_t unicode_page_index[256] = {",
    ]
    
    # Page index table (16 entries per line)
    for i in range(0, 256, 16):
        vals = ', '.join(str(page_index[j]) for j in range(i, min(i + 16, 256)))
        lines.append(f"    {vals},")
    lines[-1] = lines[-1].rstrip(',')  # Remove trailing comma from last line
    lines.append("};")
    lines.append("")
    
    # Property pages
    lines.append("/* Level 2: Property pages (each page is 256 uint8_t entries) */")
    lines.append(f"static const uint8_t unicode_props[{len(unique_pages)}][256] = {{")
    
    for page_idx, page_data in enumerate(unique_pages):
        lines.append(f"    /* Page {page_idx} */ {{")
        # 16 values per line
        for i in range(0, 256, 16):
            vals = ', '.join(f'0x{page_data[j]:02x}' for j in range(i, min(i + 16, 256)))
            lines.append(f"        {vals},")
        lines.append("    },")
    
    lines.append("};")
    lines.append("")
    
    # Inline lookup function
    lines.extend([
        "/* Fast O(1) Unicode property lookup */",
        "static inline uint8_t uprop(uint32_t cp) {",
        "    if (cp > 0x10FFFF) return 0;  /* Out of Unicode range */",
        "    uint16_t page = unicode_page_index[cp >> 8];",
        "    return unicode_props[page][cp & 0xFF];",
        "}",
        "",
        "#endif /* UNICODE_PROPS_H */",
    ])
    
    Path(output_path).write_text('\n'.join(lines))
    print(f"Written: {output_path}", file=sys.stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Unicode property tables')
    parser.add_argument('--output', '-o', default='unicode_props.h',
                       help='Output header file path')
    parser.add_argument('--offline', action='store_true',
                       help='Use cached files only, no network')
    args = parser.parse_args()
    
    # Fetch data
    unicode_text = fetch_unicode_data() if not args.offline else None
    emoji_text = fetch_emoji_data() if not args.offline else None
    
    # Try to use cached files if offline or if fetches failed
    if not unicode_text and args.offline:
        cache_path = Path("/tmp/UnicodeData.txt")
        if cache_path.exists():
            print(f"✓ Using cached UnicodeData.txt", file=sys.stderr)
            unicode_text = cache_path.read_text()
    
    if not emoji_text and args.offline:
        cache_path = Path("/tmp/emoji-data.txt")
        if cache_path.exists():
            print(f"✓ Using cached emoji-data.txt", file=sys.stderr)
            emoji_text = cache_path.read_text()
    
    # Build properties
    props = parse_unicode_data(unicode_text) if unicode_text else {}
    
    if emoji_text:
        parse_emoji_data(emoji_text, props)
        print(f"✓ Added emoji properties", file=sys.stderr)
    
    # Hardcode essential properties if we couldn't fetch data
    if not unicode_text:
        print(f"⚠️  Using fallback properties for essential codepoints", file=sys.stderr)
        # Essential Luganda vowels with accents
        for cp in LUGANDA_VOWEL_ACCENTS:
            props[cp] = U_PROP_LETTER | U_PROP_VOWEL
        # ZWJ
        props[0x200D] = U_PROP_ZWJ
        # Regional indicators
        for cp in range(0x1F1E6, 0x1F1FF + 1):
            props[cp] = U_PROP_RI | U_PROP_EMOJI
    
    # Generate tables
    page_index, unique_pages = generate_tables(props)
    print(f"✓ Generated {len(unique_pages)} unique pages from {len(props)} codepoints", 
          file=sys.stderr)
    
    # Emit header
    emit_c_header(page_index, unique_pages, args.output)
    print(f"✓ Complete: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
