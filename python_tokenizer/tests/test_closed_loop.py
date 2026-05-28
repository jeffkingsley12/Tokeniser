"""
Tests for the Python Closed-Loop Ingestion Agent using Python's built-in unittest library.
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path so python_tokenizer can be imported from any working directory
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from python_tokenizer import LugandaClosedLoopEngineAgent


class TestClosedLoopAgent(unittest.TestCase):
    """Test LugandaClosedLoopEngineAgent operations."""

    @classmethod
    def setUpClass(cls):
        """Find a valid model path or skip tests if none exist."""
        paths = [
            Path("gemini_cognitive_snapshot.bin"),
            Path("../gemini_cognitive_snapshot.bin"),
            Path("test_engine_tmp.bin"),
        ]
        
        cls.model_path = None
        for p in paths:
            if p.exists():
                cls.model_path = p
                break
        if cls.model_path is None:
            raise unittest.SkipTest("No valid Gemini cognitive snapshot file found to run tests")
        print("TEST MODEL PATH:", cls.model_path.resolve(), "SIZE:", cls.model_path.stat().st_size)

    def setUp(self):
        """Create a temporary copy of the snapshot for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.temp_model = Path(self.tmpdir) / "model.bin"
        shutil.copy(self.model_path, self.temp_model)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.tmpdir)

    def test_load_and_unload_agent(self):
        """Test basic loading and unloading of the agent."""
        agent = LugandaClosedLoopEngineAgent(self.temp_model, writable=True)
        try:
            self.assertTrue(agent.node_count > 0)
            self.assertTrue(agent.symbol_count >= 0)
            self.assertTrue(agent.edge_count >= 0)
            self.assertIn("LugandaClosedLoopEngineAgent", repr(agent))
        finally:
            agent.close()

    def test_context_manager(self):
        """Test context manager lifecycle of the agent."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            self.assertTrue(agent.node_count > 0)
            
        # Verify it is closed and raises RuntimeError on access
        with self.assertRaises(RuntimeError):
            _ = agent.node_count

    def test_telemetry_getters(self):
        """Test telemetry properties (stability, entropy, active regions)."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            stability = agent.scc_stability
            entropy_delta = agent.entropy_delta
            active_regions = agent.active_region_count
            
            # Assert types and basic value ranges
            self.assertIsInstance(stability, float)
            self.assertTrue(0.0 <= stability <= 1.0)
            
            self.assertIsInstance(entropy_delta, float)
            
            self.assertIsInstance(active_regions, int)
            self.assertTrue(active_regions >= 0)

    def test_config_properties(self):
        """Test EngineConfig property getters and setters."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            # Test default getters
            self.assertIsInstance(agent.rho_min, float)
            self.assertIsInstance(agent.h_max, float)
            self.assertIsInstance(agent.min_freq, int)
            self.assertIsInstance(agent.promotion_epochs, int)
            
            # Test setters
            agent.rho_min = 0.5
            self.assertEqual(agent.rho_min, 0.5)
            
            agent.h_max = 3.0
            self.assertEqual(agent.h_max, 3.0)
            
            agent.min_freq = 5
            self.assertEqual(agent.min_freq, 5)
            
            agent.promotion_epochs = 3
            self.assertEqual(agent.promotion_epochs, 3)

    def test_set_weight_modifier(self):
        """Test that weight modifier can be set successfully."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            # Modifying weight should not raise any exceptions
            agent.set_weight_modifier(0.5)
            agent.set_weight_modifier(1.5)
            agent.set_weight_modifier(0.0)

    def test_process_token(self):
        """Test token ingestion and incremental steps."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            orig_node_count = agent.node_count
            
            # Ingest a new token
            node_id = agent.process_token(token_id=9999)
            self.assertNotEqual(node_id, 0xFFFFFFFF)
            
            # Node count should increase or stay same if it already existed
            self.assertTrue(agent.node_count >= orig_node_count)

    def test_compute_crawl_delay(self):
        """Test adaptive closed-loop delay controller."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            base_delay = 1.0
            delay = agent.compute_crawl_delay(base_delay)
            
            # Since stability/entropy is queried, delay should be a positive float
            self.assertIsInstance(delay, float)
            self.assertTrue(delay >= base_delay)
            self.assertTrue(delay <= base_delay * 10.0)

    def test_promote_eligible(self):
        """Test DAWG promotion execution."""
        with LugandaClosedLoopEngineAgent(self.temp_model, writable=True) as agent:
            promoted = agent.promote_eligible()
            self.assertIsInstance(promoted, int)
            self.assertTrue(promoted >= 0)


if __name__ == "__main__":
    unittest.main()
