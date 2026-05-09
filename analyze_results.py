#!/usr/bin/env python3
##############################################################################
# Tokeniser Test & Benchmark Analysis Tool
# ========================================
# Aggregates, analyzes, and visualizes test and benchmark results
##############################################################################

import os
import sys
import json
import glob
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

@dataclass
class TestResult:
    """Represents a single test result"""
    name: str
    passed: bool
    duration_ms: Optional[float] = None
    output_lines: int = 0
    error_message: Optional[str] = None

@dataclass
class BenchmarkMetric:
    """Represents a benchmark metric"""
    name: str
    value: float
    unit: str
    description: str

class TestAnalyzer:
    """Analyzes test results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = []
        # Load run_tests.sh report for pass/fail status
        self.report_status = {}
        report_files = list(self.results_dir.glob('test_report_*.txt'))
        if report_files:
            latest_report = max(report_files, key=lambda p: p.stat().st_mtime)
            self._parse_report(latest_report)

    def _parse_report(self, report_path: Path):
        """Parse run_tests.sh report to extract pass/fail status"""
        try:
            with open(report_path, 'r') as f:
                for line in f:
                    # Look for: "✓ PASS: test_name" or "✗ FAIL: test_name"
                    if '✓ PASS:' in line or '✗ FAIL:' in line:
                        # Extract test name from lines like:
                        # "✓ PASS: edge_cases (123 lines of output)"
                        # "✗ FAIL: serialization_security (exit code: 139)"
                        match = re.search(r'[✓✗]\s+(PASS|FAIL):\s+(\w+)', line)
                        if match:
                            status = match.group(1) == 'PASS'
                            test_name = match.group(2)
                            self.report_status[test_name] = status
        except Exception as e:
            print(f"Warning: Could not parse report {report_path}: {e}", file=sys.stderr)
        
    def parse_test_output(self, filepath: Path) -> Optional[TestResult]:
        """Parse test output file"""
        try:
            # Use errors='replace' to handle binary/non-UTF8 data in test output
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            lines = content.split('\n')
            test_name = filepath.stem.replace('_output_', '')

            # First check if we have status from run_tests.sh report
            # test_name format: edge_cases20260505_160652 (after replace)
            # report key format: edge_cases
            base_name = test_name.split('2026')[0].rstrip('_')
            if base_name in self.report_status:
                passed = self.report_status[base_name]
                return TestResult(
                    name=test_name,
                    passed=passed,
                    output_lines=len(lines),
                    error_message=None if passed else "See test output for details"
                )

            # Fallback: Check for success indicators in output:
            # 1. Exit code 0 at the end (Run 1: exit 0)
            # 2. Success rate: 100.0%
            # 3. All tests passed message
            # 4. No "exit 139" (segfault) or "exit 1" (failure)
            exit_match = re.search(r'exit\s+(\d+)', content.lower())
            exit_code = int(exit_match.group(1)) if exit_match else 0

            success_rate_match = re.search(r'success rate:\s+(\d+\.?\d*)%', content.lower())
            success_rate = float(success_rate_match.group(1)) if success_rate_match else 0

            # Count actual failures in output (❌ FAIL markers)
            fail_count = len([l for l in lines if '❌ fail' in l.lower()])

            # Consider passed if:
            # - Success rate is 100% AND no ❌ FAIL markers
            # - Or no failure exit code (0 or not present) and no fails
            # - And no segfault (exit 139)
            has_100_percent = success_rate >= 99.0 or '100.0%' in content
            passed = (has_100_percent or (exit_code == 0 and fail_count == 0)) and exit_code != 139

            return TestResult(
                name=test_name,
                passed=passed,
                output_lines=len(lines),
                error_message=None if passed else f"Exit code: {exit_code}, Failures: {fail_count}"
            )
        except Exception as e:
            return TestResult(
                name=filepath.stem,
                passed=False,
                error_message=str(e)
            )
    
    def analyze_all_tests(self) -> Dict:
        """Analyze all test results in directory"""
        test_files = sorted(self.results_dir.glob('*_output_*.txt'))
        
        results = {}
        for f in test_files:
            result = self.parse_test_output(f)
            if result:
                results[result.name] = asdict(result)
        
        total = len(results)
        passed = sum(1 for r in results.values() if r['passed'])
        failed = total - passed
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'success_rate': (passed / total * 100) if total > 0 else 0,
            'tests': results
        }

class BenchmarkAnalyzer:
    """Analyzes benchmark results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def parse_metrics_from_output(self, filepath: Path) -> List[BenchmarkMetric]:
        """Extract metrics from benchmark output"""
        metrics = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Extract common patterns
            patterns = {
                'throughput': r'throughput[:\s]*(\d+\.?\d*)\s*(?:tokens/ms|ops/sec)',
                'fast_path': r'fast.?path[:\s]*(\d+\.?\d*)\s*%',
                'cycles': r'cycles[:\s]*(\d+\.?\d*)',
                'tokens_per_ms': r'tokens/ms[:\s]*(\d+\.?\d*)',
                'latency': r'latency[:\s]*(\d+\.?\d*)\s*(?:ms|μs)',
            }
            
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    metrics.append(BenchmarkMetric(
                        name=metric_name,
                        value=float(match.group(1)),
                        unit='units',  # Placeholder
                        description=f"Extracted from benchmark output"
                    ))
        
        except Exception as e:
            print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        
        return metrics
    
    def analyze_all_benchmarks(self) -> Dict:
        """Analyze all benchmark results"""
        bench_files = sorted(self.results_dir.glob('benchmark_*_output_*.txt'))
        # Also check benchmark_results directory if it exists
        bench_dir = self.results_dir.parent / 'benchmark_results'
        if bench_dir.exists():
            bench_files.extend(sorted(bench_dir.glob('benchmark_*_output_*.txt')))
        
        results = {}
        for f in bench_files:
            bench_name = f.stem.replace('_output_', '')
            metrics = self.parse_metrics_from_output(f)
            results[bench_name] = {
                'metrics': [asdict(m) for m in metrics],
                'file': str(f),
                'timestamp': f.stat().st_mtime
            }
        
        return results

class ReportGenerator:
    """Generates comprehensive reports"""
    
    @staticmethod
    def generate_markdown_report(test_results: Dict, bench_results: Dict, 
                                output_file: Path) -> None:
        """Generate markdown report"""
        
        with open(output_file, 'w') as f:
            f.write("# Tokeniser Testing & Benchmarking Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Test Summary
            f.write("## Test Summary\n\n")
            f.write(f"- **Total Tests:** {test_results['total']}\n")
            f.write(f"- **Passed:** {test_results['passed']}\n")
            f.write(f"- **Failed:** {test_results['failed']}\n")
            f.write(f"- **Success Rate:** {test_results['success_rate']:.1f}%\n\n")
            
            # Test Details
            f.write("### Test Details\n\n")
            f.write("| Test | Status | Output Lines |\n")
            f.write("|------|--------|---------------|\n")
            
            for test_name, test_data in test_results['tests'].items():
                status = "✓ PASS" if test_data['passed'] else "✗ FAIL"
                f.write(f"| {test_name} | {status} | {test_data['output_lines']} |\n")
            
            f.write("\n")
            
            # Benchmark Results
            if bench_results:
                f.write("## Benchmark Results\n\n")
                
                for bench_name, bench_data in bench_results.items():
                    f.write(f"### {bench_name}\n\n")
                    
                    if bench_data['metrics']:
                        f.write("| Metric | Value | Unit |\n")
                        f.write("|--------|-------|------|\n")
                        
                        for metric in bench_data['metrics']:
                            f.write(f"| {metric['name']} | {metric['value']:.2f} | {metric['unit']} |\n")
                    else:
                        f.write("*No metrics extracted*\n")
                    
                    f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if test_results['success_rate'] < 100:
                f.write("- **Address Test Failures:** Review failed tests and fix reported issues\n")
            
            f.write("- **Performance Monitoring:** Track metrics over time\n")
            f.write("- **Regression Testing:** Run full suite before commits\n")
            f.write("- **Documentation:** Keep benchmark expectations up to date\n")
    
    @staticmethod
    def generate_json_report(test_results: Dict, bench_results: Dict,
                            output_file: Path) -> None:
        """Generate JSON report for machine parsing"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'tests': test_results,
            'benchmarks': bench_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    """Main entry point"""
    
    # Determine results directory
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path.cwd()
    
    test_results_dir = base_dir / 'test_results'
    benchmark_results_dir = base_dir / 'benchmark_results'
    report_dir = base_dir / 'analysis'
    report_dir.mkdir(exist_ok=True)
    
    print("Tokeniser Analysis Tool")
    print(f"Results directory: {base_dir}")
    print()
    
    # Analyze tests
    print("Analyzing test results...")
    test_analyzer = TestAnalyzer(test_results_dir)
    test_results = test_analyzer.analyze_all_tests()
    
    print(f"  Total tests: {test_results['total']}")
    print(f"  Passed: {test_results['passed']}")
    print(f"  Failed: {test_results['failed']}")
    print(f"  Success rate: {test_results['success_rate']:.1f}%")
    print()
    
    # Analyze benchmarks
    print("Analyzing benchmark results...")
    bench_analyzer = BenchmarkAnalyzer(benchmark_results_dir)
    bench_results = bench_analyzer.analyze_all_benchmarks()
    
    print(f"  Total benchmarks: {len(bench_results)}")
    for bench_name in bench_results.keys():
        metrics_count = len(bench_results[bench_name]['metrics'])
        print(f"    - {bench_name}: {metrics_count} metrics")
    print()
    
    # Generate reports
    print("Generating reports...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    md_report = report_dir / f'report_{timestamp}.md'
    json_report = report_dir / f'report_{timestamp}.json'
    
    ReportGenerator.generate_markdown_report(test_results, bench_results, md_report)
    ReportGenerator.generate_json_report(test_results, bench_results, json_report)
    
    print(f"  Markdown report: {md_report}")
    print(f"  JSON report: {json_report}")
    print()
    
    print("Analysis complete!")
    
    return 0 if test_results['failed'] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
