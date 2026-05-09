#!/usr/bin/env python3
"""
Luganda Tokenizer Benchmark Analysis Tool

This script analyzes the output from benchmark_luganda and generates:
- Performance plots (fast-path ratio, throughput)
- Statistical summaries by input type
- Recommendations for optimization

Usage:
    ./benchmark_luganda tokenizer_model.bin tsv > results.tsv
    python3 analyze_benchmark.py results.tsv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_benchmark_data(filepath):
    """Load benchmark TSV output into pandas DataFrame."""
    try:
        # Read TSV with proper handling for mixed data types
        df = pd.read_csv(filepath, sep='\t', dtype={'description': str})
        # Drop any rows with missing data
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def analyze_performance_by_flag(df):
    """Analyze performance metrics by input type flags."""
    # Extract flag information from case names and descriptions
    flag_analysis = {
        'EMOJI': [],
        'MIXED_LANG': [],
        'CRLF': [],
        'UNICODE_WS': [],
        'LOANWORDS': [],
        'MISSPELL': [],
        'LONG_INPUT': [],
        'EDGE_SPACE': []
    }
    
    # Simple heuristic: detect flags from case names and descriptions
    for _, row in df.iterrows():
        name = str(row['case_name']).upper()
        desc = str(row['description']).upper()
        
        if 'EMOJI' in name or 'EMOJI' in desc:
            flag_analysis['EMOJI'].append(row)
        if 'MIXED' in name or 'ENGLISH' in desc or 'MIXED' in desc:
            flag_analysis['MIXED_LANG'].append(row)
        if 'CRLF' in name or 'WINDOWS' in desc:
            flag_analysis['CRLF'].append(row)
        if 'WHITESPACE' in name or 'SPACE' in desc or 'NBSP' in desc:
            flag_analysis['UNICODE_WS'].append(row)
        if 'LOAN' in name or 'FRENCH' in desc:
            flag_analysis['LOANWORDS'].append(row)
        if 'MISSPELL' in name or 'SPELL' in desc:
            flag_analysis['MISSPELL'].append(row)
        if 'LONG' in name or 'MESSAGE' in desc:
            flag_analysis['LONG_INPUT'].append(row)
        if 'EDGE' in name or 'LEADING' in desc or 'TRAILING' in desc:
            flag_analysis['EDGE_SPACE'].append(row)
    
    return flag_analysis

def create_performance_plots(df, output_dir='plots'):
    """Generate performance visualization plots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Fast-path ratio by case
    plt.figure(figsize=(12, 6))
    cases = df['case_name']
    fast_path_pct = df['fast_path_pct']
    
    plt.bar(range(len(cases)), fast_path_pct, color='steelblue', alpha=0.7)
    plt.xlabel('Benchmark Case')
    plt.ylabel('Fast-path Ratio (%)')
    plt.title('Fast-path Performance by Benchmark Case')
    plt.xticks(range(len(cases)), cases, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fast_path_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Throughput (tokens/ms) by case
    plt.figure(figsize=(12, 6))
    throughput = df['tokens_per_ms']
    
    plt.bar(range(len(cases)), throughput, color='coral', alpha=0.7)
    plt.xlabel('Benchmark Case')
    plt.ylabel('Throughput (tokens/ms)')
    plt.title('Tokenization Throughput by Benchmark Case')
    plt.xticks(range(len(cases)), cases, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Fast-path vs Throughput scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(df['fast_path_pct'], df['tokens_per_ms'], 
                alpha=0.7, s=60, color='purple')
    plt.xlabel('Fast-path Ratio (%)')
    plt.ylabel('Throughput (tokens/ms)')
    plt.title('Fast-path Ratio vs Throughput')
    
    # Add case labels
    for i, case in enumerate(df['case_name']):
        plt.annotate(case, (df['fast_path_pct'].iloc[i], df['tokens_per_ms'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fast_path_vs_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_flag_performance(flag_analysis):
    """Generate analysis report for each input type."""
    print("\n=== PERFORMANCE ANALYSIS BY INPUT TYPE ===")
    
    for flag, rows in flag_analysis.items():
        if not rows:
            continue
            
        df_flag = pd.DataFrame(rows)
        avg_fast_path = df_flag['fast_path_pct'].mean()
        avg_throughput = df_flag['tokens_per_ms'].mean()
        avg_cycles = df_flag['cycles_per_token'].mean()
        
        print(f"\n{flag}:")
        print(f"  Cases: {len(rows)}")
        print(f"  Avg fast-path ratio: {avg_fast_path:.1f}%")
        print(f"  Avg throughput: {avg_throughput:.2f} tokens/ms")
        print(f"  Avg cycles per token: {avg_cycles:.0f}")
        
        # Performance tier classification
        if avg_fast_path >= 70:
            tier = "EXCELLENT"
        elif avg_fast_path >= 50:
            tier = "GOOD"
        elif avg_fast_path >= 30:
            tier = "MODERATE"
        else:
            tier = "NEEDS_OPTIMIZATION"
        
        print(f"  Performance tier: {tier}")

def generate_optimization_recommendations(df, flag_analysis):
    """Generate specific optimization recommendations."""
    print("\n=== OPTIMIZATION RECOMMENDATIONS ===")
    
    # Overall performance assessment
    overall_fast_path = df['fast_path_pct'].mean()
    overall_throughput = df['tokens_per_ms'].mean()
    
    print(f"\nOverall Performance:")
    print(f"  Average fast-path ratio: {overall_fast_path:.1f}%")
    print(f"  Average throughput: {overall_throughput:.2f} tokens/ms")
    
    # Specific recommendations
    recommendations = []
    
    # Check emoji performance
    emoji_rows = flag_analysis.get('EMOJI', [])
    if emoji_rows:
        emoji_fast_path = pd.DataFrame(emoji_rows)['fast_path_pct'].mean()
        if emoji_fast_path < 40:
            recommendations.append(
                "EMOJI: Consider adding emoji-specific fast-path optimization"
            )
    
    # Check mixed language performance
    mixed_rows = flag_analysis.get('MIXED_LANG', [])
    if mixed_rows:
        mixed_fast_path = pd.DataFrame(mixed_rows)['fast_path_pct'].mean()
        if mixed_fast_path < 50:
            recommendations.append(
                "MIXED_LANG: English insertions reduce fast-path hits - consider hybrid approach"
            )
    
    # Check misspellings
    misspell_rows = flag_analysis.get('MISSPELL', [])
    if misspell_rows:
        misspell_fast_path = pd.DataFrame(misspell_rows)['fast_path_pct'].mean()
        if misspell_fast_path < 30:
            recommendations.append(
                "MISSPELL: Misspellings force trie fallback - consider fuzzy matching"
            )
    
    # Check whitespace handling
    ws_rows = flag_analysis.get('UNICODE_WS', [])
    if ws_rows:
        ws_fast_path = pd.DataFrame(ws_rows)['fast_path_pct'].mean()
        if ws_fast_path < 60:
            recommendations.append(
                "UNICODE_WS: Unicode whitespace reduces performance - expand fast-path"
            )
    
    # General recommendations
    if overall_fast_path < 60:
        recommendations.append(
            "GENERAL: Fast-path ratio below 60% - consider expanding cv_to_token table"
        )
    
    if overall_throughput < 100:
        recommendations.append(
            "THROUGHPUT: Below 100 tokens/ms - consider SIMD optimizations"
        )
    
    # Check variance
    fast_path_variance = df['fast_path_pct'].var()
    if fast_path_variance > 400:  # High variance
        recommendations.append(
            "CONSISTENCY: High performance variance - investigate edge cases"
        )
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✅ Performance is optimal across all test cases!")

def create_summary_report(df, flag_analysis, output_file='benchmark_report.md'):
    """Generate a comprehensive markdown report."""
    with open(output_file, 'w') as f:
        f.write("# Luganda Tokenizer Benchmark Report\n\n")
        
        # Executive summary
        f.write("## Executive Summary\n\n")
        overall_fast_path = df['fast_path_pct'].mean()
        overall_throughput = df['tokens_per_ms'].mean()
        
        f.write(f"- **Average Fast-path Ratio**: {overall_fast_path:.1f}%\n")
        f.write(f"- **Average Throughput**: {overall_throughput:.2f} tokens/ms\n")
        f.write(f"- **Total Test Cases**: {len(df)}\n")
        f.write(f"- **Performance Tier**: ")
        
        if overall_fast_path >= 70:
            f.write("🟢 EXCELLENT\n")
        elif overall_fast_path >= 50:
            f.write("🟡 GOOD\n")
        else:
            f.write("🔴 NEEDS OPTIMIZATION\n")
        
        # Detailed results table
        f.write("\n## Detailed Results\n\n")
        f.write("| Case | Expected | Actual | Fast-path % | Throughput | Description |\n")
        f.write("|------|----------|--------|-------------|------------|-------------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['case_name']} | {row['expected_tokens']} | "
                   f"{row['actual_tokens']} | {row['fast_path_pct']:.1f}% | "
                   f"{row['tokens_per_ms']:.2f} | {row['description']} |\n")
        
        # Performance by input type
        f.write("\n## Performance by Input Type\n\n")
        for flag, rows in flag_analysis.items():
            if not rows:
                continue
            df_flag = pd.DataFrame(rows)
            avg_fast_path = df_flag['fast_path_pct'].mean()
            avg_throughput = df_flag['tokens_per_ms'].mean()
            
            f.write(f"### {flag}\n")
            f.write(f"- Cases: {len(rows)}\n")
            f.write(f"- Avg Fast-path: {avg_fast_path:.1f}%\n")
            f.write(f"- Avg Throughput: {avg_throughput:.2f} tokens/ms\n\n")
    
    print(f"\n📄 Summary report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze Luganda tokenizer benchmark results')
    parser.add_argument('input_file', help='TSV output from benchmark_luganda')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--report', default='benchmark_report.md', help='Output report file')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    # Load data
    df = load_benchmark_data(args.input_file)
    if df is None:
        sys.exit(1)
    
    print(f"Loaded {len(df)} benchmark results from {args.input_file}")
    
    # Analyze by input type
    flag_analysis = analyze_performance_by_flag(df)
    
    # Generate plots
    if not args.no_plots:
        try:
            create_performance_plots(df, args.output_dir)
            print(f"📊 Plots saved to: {args.output_dir}/")
        except ImportError:
            print("⚠️  Matplotlib not available - skipping plots")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
    
    # Print analysis
    analyze_flag_performance(flag_analysis)
    generate_optimization_recommendations(df, flag_analysis)
    
    # Create summary report
    create_summary_report(df, flag_analysis, args.report)
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
