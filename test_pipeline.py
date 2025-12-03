"""
Comprehensive test suite for the Jira scraping pipeline
Tests edge cases, fault tolerance, and data quality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
from pathlib import Path
import time

from scraper import JiraScraper, ScraperConfig, RateLimiter, CircuitBreaker, CheckpointManager
from transformer import LLMDataTransformer, TextCleaner, TaskGenerator, DataValidator


class TestRateLimiter(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def test_rate_limiter_enforces_delay(self):
        """Test that rate limiter enforces minimum delay"""
        limiter = RateLimiter(requests_per_second=10.0)
        
        start = time.time()
        for _ in range(3):
            limiter.acquire()
        elapsed = time.time() - start
        
        # Should take at least 0.2 seconds (3 requests at 10 rps)
        self.assertGreater(elapsed, 0.2)
    
    def test_rate_limiter_token_refill(self):
        """Test that tokens refill over time"""
        limiter = RateLimiter(requests_per_second=10.0)
        
        # Exhaust tokens
        for _ in range(10):
            limiter.acquire()
        
        # Wait for refill
        time.sleep(0.5)
        
        # Should be able to acquire again
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start
        
        # Should be almost immediate
        self.assertLess(elapsed, 0.15)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker pattern"""
    
    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        def failing_function():
            raise Exception("Simulated failure")
        
        # Cause failures
        for _ in range(3):
            with self.assertRaises(Exception):
                breaker.call(failing_function)
        
        # Circuit should be open
        self.assertEqual(breaker.state, "OPEN")
    
    def test_circuit_closes_after_timeout(self):
        """Test that circuit transitions to half-open after timeout"""
        breaker = CircuitBreaker(failure_threshold=2, timeout=1)
        
        def failing_function():
            raise Exception("Failure")
        
        # Open circuit
        for _ in range(2):
            try:
                breaker.call(failing_function)
            except:
                pass
        
        self.assertEqual(breaker.state, "OPEN")
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should transition to HALF_OPEN on next call
        def success_function():
            return "success"
        
        result = breaker.call(success_function)
        self.assertEqual(breaker.state, "CLOSED")


class TestCheckpointManager(unittest.TestCase):
    """Test checkpoint persistence"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir)
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        state = {
            'start_at': 100,
            'total_scraped': 50,
            'last_issue_key': 'KAFKA-1234'
        }
        
        self.checkpoint_manager.save_checkpoint('KAFKA', state)
        loaded_state = self.checkpoint_manager.load_checkpoint('KAFKA')
        
        self.assertEqual(loaded_state['start_at'], 100)
        self.assertEqual(loaded_state['last_issue_key'], 'KAFKA-1234')
    
    def test_load_nonexistent_checkpoint(self):
        """Test loading checkpoint that doesn't exist"""
        result = self.checkpoint_manager.load_checkpoint('NONEXISTENT')
        self.assertIsNone(result)
    
    def test_clear_checkpoint(self):
        """Test clearing checkpoints"""
        state = {'start_at': 50}
        self.checkpoint_manager.save_checkpoint('TEST', state)
        
        self.checkpoint_manager.clear_checkpoint('TEST')
        result = self.checkpoint_manager.load_checkpoint('TEST')
        
        self.assertIsNone(result)


class TestTextCleaner(unittest.TestCase):
    """Test text cleaning functionality"""
    
    def test_clean_jira_markup(self):
        """Test removal of Jira markup"""
        text = "Some text {code}print('hello'){code} more text"
        cleaned = TextCleaner.clean_text(text)
        
        self.assertIn("[CODE_BLOCK]", cleaned)
        self.assertNotIn("{code}", cleaned)
    
    def test_clean_user_mentions(self):
        """Test removal of user mentions"""
        text = "Thanks [~johndoe] for the fix"
        cleaned = TextCleaner.clean_text(text)
        
        self.assertNotIn("[~johndoe]", cleaned)
        self.assertIn("Thanks", cleaned)
    
    def test_clean_links(self):
        """Test link cleaning"""
        text = "See [documentation|http://example.com]"
        cleaned = TextCleaner.clean_text(text)
        
        self.assertIn("documentation", cleaned)
        self.assertNotIn("http://", cleaned)
    
    def test_extract_code_blocks(self):
        """Test code block extraction"""
        text = "{code:java}public class Test{}{code} and {code:python}print('hi'){code}"
        blocks = TextCleaner.extract_code_blocks(text)
        
        self.assertEqual(len(blocks), 2)
        self.assertIn("public class Test", blocks[0])
    
    def test_truncate_text(self):
        """Test text truncation"""
        long_text = "a" * 3000
        truncated = TextCleaner.truncate_text(long_text, max_length=100)
        
        self.assertLessEqual(len(truncated), 103)  # 100 + "..."
        self.assertTrue(truncated.endswith("..."))
    
    def test_handle_none_input(self):
        """Test handling of None input"""
        result = TextCleaner.clean_text(None)
        self.assertEqual(result, "")


class TestDataValidator(unittest.TestCase):
    """Test data validation"""
    
    def test_valid_issue(self):
        """Test validation of valid issue"""
        issue = {
            'key': 'KAFKA-1234',
            'summary': 'Fix bug',
            'description': 'This is a bug description'
        }
        
        self.assertTrue(DataValidator.is_valid_issue(issue))
    
    def test_invalid_issue_no_key(self):
        """Test rejection of issue without key"""
        issue = {
            'summary': 'Fix bug',
            'description': 'Description'
        }
        
        self.assertFalse(DataValidator.is_valid_issue(issue))
    
    def test_invalid_issue_no_content(self):
        """Test rejection of issue without description or comments"""
        issue = {
            'key': 'KAFKA-1234',
            'summary': 'Fix bug',
            'description': '',
            'comments': []
        }
        
        self.assertFalse(DataValidator.is_valid_issue(issue))
    
    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        high_quality_issue = {
            'description': 'Detailed description here',
            'comments': [{'body': 'comment 1'}, {'body': 'comment 2'}],
            'labels': ['bug', 'performance'],
            'status': 'Resolved'
        }
        
        score = DataValidator.calculate_quality_score(high_quality_issue)
        self.assertGreater(score, 0.8)
        
        low_quality_issue = {
            'description': '',
            'comments': [],
            'labels': [],
            'status': 'Open'
        }
        
        score = DataValidator.calculate_quality_score(low_quality_issue)
        self.assertLess(score, 0.2)


class TestTaskGenerator(unittest.TestCase):
    """Test training task generation"""
    
    def setUp(self):
        self.sample_issue = {
            'key': 'KAFKA-1234',
            'summary': 'Improve broker performance',
            'description': 'The broker performance needs improvement in high-load scenarios.',
            'status': 'Resolved',
            'priority': 'Major',
            'assignee': 'John Doe',
            'reporter': 'Jane Smith',
            'labels': ['performance', 'broker'],
            'components': ['core'],
            'comments': [
                {'author': 'John', 'body': 'Working on a fix', 'created': '2024-01-01'},
                {'author': 'Jane', 'body': 'Fixed in PR #123', 'created': '2024-01-02'}
            ]
        }
    
    def test_generate_summarization_task(self):
        """Test summarization task generation"""
        task = TaskGenerator.generate_summarization_task(self.sample_issue)
        
        self.assertEqual(task['task'], 'summarization')
        self.assertIn('broker performance', task['input'])
        self.assertEqual(task['output'], 'Improve broker performance')
        self.assertEqual(task['metadata']['issue_key'], 'KAFKA-1234')
    
    def test_generate_classification_task(self):
        """Test classification task generation"""
        task = TaskGenerator.generate_classification_task(self.sample_issue)
        
        self.assertEqual(task['task'], 'classification')
        self.assertIn('Improve broker performance', task['input'])
        self.assertEqual(task['output']['status'], 'Resolved')
        self.assertEqual(task['output']['priority'], 'Major')
    
    def test_generate_qa_pairs(self):
        """Test Q&A pair generation"""
        qa_pairs = TaskGenerator.generate_qa_pairs(self.sample_issue)
        
        self.assertGreater(len(qa_pairs), 0)
        self.assertTrue(all(qa['task'] == 'qa' for qa in qa_pairs))
        
        # Check for expected questions
        questions = [qa['input'] for qa in qa_pairs]
        self.assertTrue(any('status' in q.lower() for q in questions))
    
    def test_generate_resolution_prediction(self):
        """Test resolution prediction task"""
        task = TaskGenerator.generate_resolution_prediction(self.sample_issue)
        
        self.assertIsNotNone(task)
        self.assertEqual(task['task'], 'resolution_prediction')
        self.assertIn('Improve broker performance', task['input'])
    
    def test_no_resolution_for_open_issue(self):
        """Test that open issues don't generate resolution tasks"""
        open_issue = self.sample_issue.copy()
        open_issue['status'] = 'Open'
        
        task = TaskGenerator.generate_resolution_prediction(open_issue)
        self.assertIsNone(task)


class TestLLMDataTransformer(unittest.TestCase):
    """Test complete data transformation pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.transformer = LLMDataTransformer(
            output_dir=self.temp_dir,
            min_quality_score=0.3
        )
        
        self.sample_issue = {
            'key': 'SPARK-5678',
            'summary': 'Add new SQL function',
            'description': 'We need a new SQL function for data processing.',
            'status': 'Closed',
            'priority': 'Minor',
            'assignee': 'Alice',
            'reporter': 'Bob',
            'labels': ['sql'],
            'components': ['sql'],
            'comments': [{'author': 'Alice', 'body': 'Implemented', 'created': '2024-01-01'}]
        }
    
    def test_transform_valid_issue(self):
        """Test transformation of valid issue"""
        examples = self.transformer.transform_issue('SPARK', self.sample_issue)
        
        self.assertGreater(len(examples), 0)
        self.assertTrue(all('task' in ex for ex in examples))
        self.assertTrue(all('project' in ex for ex in examples))
    
    def test_filter_low_quality_issues(self):
        """Test filtering of low-quality issues"""
        low_quality_issue = {
            'key': 'TEST-1',
            'summary': 'Test',
            'description': '',
            'comments': [],
            'status': 'Open'
        }
        
        examples = self.transformer.transform_issue('TEST', low_quality_issue)
        self.assertEqual(len(examples), 0)
    
    def test_deduplication(self):
        """Test that duplicate examples are filtered"""
        # Process same issue twice
        examples1 = self.transformer.transform_issue('SPARK', self.sample_issue)
        examples2 = self.transformer.transform_issue('SPARK', self.sample_issue)
        
        # Second processing should produce no examples (all duplicates)
        self.assertEqual(len(examples2), 0)
    
    def test_write_jsonl(self):
        """Test JSONL file writing"""
        examples = self.transformer.transform_issue('SPARK', self.sample_issue)
        self.transformer.write_jsonl(examples, 'test_output.jsonl')
        
        output_file = Path(self.temp_dir) / 'test_output.jsonl'
        self.assertTrue(output_file.exists())
        
        # Verify format
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.assertIn('task', data)
                self.assertIn('input', data)
                self.assertIn('output', data)
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        self.transformer.transform_issue('SPARK', self.sample_issue)
        
        stats = self.transformer.get_statistics()
        self.assertGreater(stats['issues_processed'], 0)
        self.assertGreater(stats.get('summarization', 0) + stats.get('qa', 0), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_response(self):
        """Test handling of empty API response"""
        config = ScraperConfig(
            base_url="https://issues.apache.org/jira",
            projects=["TEST"]
        )
        scraper = JiraScraper(config)
        
        # Mock empty response
        with patch.object(scraper, '_make_request', return_value=None):
            issues = list(scraper.scrape_project('TEST'))
            self.assertEqual(len(issues), 0)
    
    def test_malformed_issue_data(self):
        """Test handling of malformed issue data"""
        config = ScraperConfig(
            base_url="https://issues.apache.org/jira",
            projects=["TEST"]
        )
        scraper = JiraScraper(config)
        
        malformed_data = {'key': 'TEST-1'}  # Missing fields
        result = scraper._parse_issue(malformed_data)
        
        # Should still return something, not crash
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], 'TEST-1')
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters"""
        issue = {
            'key': 'TEST-1',
            'summary': 'Fix emoji bug 🐛',
            'description': 'Test with 中文字符 and émojis 🚀',
            'status': 'Open',
            'comments': []
        }
        
        cleaned = TextCleaner.clean_text(issue['description'])
        self.assertIn('🚀', cleaned)
        self.assertIn('中文字符', cleaned)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)