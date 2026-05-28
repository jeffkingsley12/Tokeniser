#!/usr/bin/env python3
"""
demo_closed_loop.py
===================
Simulates a web crawler digesting a feed of Luganda text, using the
LugandaClosedLoopEngineAgent to monitor training turbulence and dynamically
regulate crawling delay and ingestion weights.
Optimized to stream large files (like luganda_corpus.txt) lazily.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from python_tokenizer import LugandaTokenizer, LugandaClosedLoopEngineAgent


def stream_lines(file_path: Path):
    """Lazily yield non-empty lines from file to conserve memory."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped


def run_simulation():
    parser = argparse.ArgumentParser(description="Luganda Closed-Loop Ingestion Simulation")
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to Luganda corpus file (defaults to luganda_corpus.txt if present, else test_corpus.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20000,
        help="Max number of lines to ingest (default: 20000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of lines to process before triggering an epoch transition",
    )
    args = parser.parse_args()

    model_path = project_root / "gemini_cognitive_snapshot.bin"
    tok_path = project_root / "production_model.bin"

    # Auto-detect corpus path
    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        corpus_path = project_root / "luganda_corpus.txt"
        if not corpus_path.exists():
            corpus_path = project_root / "test_corpus.txt"

    if not model_path.exists():
        print(f"Error: Model snapshot {model_path} not found.")
        sys.exit(1)
    if not tok_path.exists():
        print(f"Error: Tokenizer model {tok_path} not found.")
        sys.exit(1)
    if not corpus_path.exists():
        print(f"Error: Corpus {corpus_path} not found.")
        sys.exit(1)

    print("======================================================================")
    print("        Luganda Closed-Loop Feedback Ingestion Simulation")
    print("======================================================================")
    print(f"Loading Tokenizer: {tok_path.name}")
    tokenizer = LugandaTokenizer(tok_path)
    
    print(f"Loading Gemini Engine Context: {model_path.name}")
    agent = LugandaClosedLoopEngineAgent(model_path, writable=True)

    print(f"Streaming Corpus : {corpus_path.name}")
    print(f"Ingestion Limit  : {args.limit} lines")
    print(f"Epoch Batch Size : {args.batch_size} lines")
    print("\nStarting ingestion loop...")
    print("------------------------------------------------------------------------------------------------")
    print(f"{'Batch':<8} | {'Lines Ingested':<14} | {'Stability (rho)':<16} | {'Entropy Delta':<14} | {'Delay Mult':<10} | {'Promoted':<8} | {'Total Sym':<9}")
    print("------------------------------------------------------------------------------------------------")

    base_delay = 0.05
    tokens_processed = 0
    lines_processed = 0
    batch_idx = 0
    cumulative_promoted = 0
    
    start_time = time.time()
    
    # Process corpus lines lazily
    batch_buffer = []
    
    for line in stream_lines(corpus_path):
        batch_buffer.append(line)
        lines_processed += 1
        
        # When batch is full, ingest it
        if len(batch_buffer) >= args.batch_size:
            for sentence in batch_buffer:
                token_ids = tokenizer.encode(sentence)
                prev_node = 0xFFFFFFFF
                for tid in token_ids:
                    prev_node = agent.process_token(tid, prev_node)
                    tokens_processed += 1
            
            # Reset batch buffer
            batch_buffer = []
            batch_idx += 1
            
            # Trigger an epoch transition to recalculate metrics in the engine
            agent.begin_epoch()

            # Evaluate active candidates and promote stable SCCs to DAWG symbols
            promoted = agent.promote_eligible()
            cumulative_promoted += promoted
            
            # Gather telemetry
            stability = agent.scc_stability
            entropy_d = agent.entropy_delta
            
            # Compute the feedback crawl delay (this dynamically sets the C-level modifier too)
            adjusted_delay = agent.compute_crawl_delay(base_delay)
            multiplier = adjusted_delay / base_delay

            # Output telemetry every 10 batches (to avoid flooding console for large run)
            if batch_idx % 10 == 0 or lines_processed >= args.limit:
                print(f"{batch_idx:<8} | {lines_processed:<14} | {stability:<16.6f} | {entropy_d:<14.6f} | {multiplier:<9.2f}x | {promoted:<8} | {agent.symbol_count:<9}")
                
            # Perform micro-sleep (keep it minimal to process large corpus quickly)
            if multiplier > 1.1:
                # Sleep is proportional to delay multiplier (but scaled down for fast simulation)
                time.sleep((multiplier - 1.0) * 0.0001)

        if lines_processed >= args.limit:
            break

    # Process any leftover lines
    if batch_buffer:
        for sentence in batch_buffer:
            token_ids = tokenizer.encode(sentence)
            prev_node = 0xFFFFFFFF
            for tid in token_ids:
                prev_node = agent.process_token(tid, prev_node)
                tokens_processed += 1
        agent.begin_epoch()
        promoted = agent.promote_eligible()
        cumulative_promoted += promoted

    elapsed = time.time() - start_time
    print("----------------------------------------------------------------------")
    print(f"\nSimulation complete in {elapsed:.2f} seconds!")
    print(f"Total lines processed : {lines_processed}")
    print(f"Total tokens processed: {tokens_processed}")
    print(f"Throughput            : {tokens_processed / elapsed:.1f} tokens/second")
    print(f"Final Engine Stats:")
    print(f"  Nodes  : {agent.node_count}")
    print(f"  Edges  : {agent.edge_count}")
    print(f"  Symbols: {agent.symbol_count}")
    print(f"  SCCs   : {agent.scc_count}")
    
    agent.close()
    tokenizer.close()


if __name__ == "__main__":
    run_simulation()
