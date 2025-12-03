"""
Data Transformer for LLM Training
Converts raw Jira data into structured JSONL format with derived tasks
"""

import json
import re
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and normalizes text data"""
    
    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove Jira markup
        text = re.sub(r'\{code(?::[a-z]+)?\}.*?\{code\}', '[CODE_BLOCK]', text, flags=re.DOTALL)
        text = re.sub(r'\{quote\}.*?\{quote\}', '[QUOTE]', text, flags=re.DOTALL)
        text = re.sub(r'\{noformat\}.*?\{noformat\}', '[PREFORMATTED]', text, flags=re.DOTALL)
        
        # Remove Jira links
        text = re.sub(r'\[~[^\]]+\]', '', text)  # User mentions
        text = re.sub(r'\[([^\]|]+)\|[^\]]+\]', r'\1', text)  # Links
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_code_blocks(text: Optional[str]) -> List[str]:
        """Extract code blocks from Jira text"""
        if not text:
            return []
        
        code_pattern = r'\{code(?::[a-z]+)?\}(.*?)\{code\}'
        return re.findall(code_pattern, str(text), flags=re.DOTALL)
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 2000) -> str:
        """Truncate text to maximum length"""
        if not text:
            return ""
        text = str(text)
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class TaskGenerator:
    """Generates LLM training tasks from issue data"""
    
    @staticmethod
    def generate_summarization_task(issue: Dict) -> Dict:
        """Generate summarization task"""
        description = TextCleaner.clean_text(issue.get('description', ''))
        comments = issue.get('comments', [])
        
        # Combine description and top comments
        full_text = description
        if comments:
            comment_texts = [TextCleaner.clean_text(c.get('body', '')) for c in comments[:3]]
            comment_texts = [c for c in comment_texts if c]  # Remove empty
            if comment_texts:
                full_text += "\n\nComments:\n" + "\n".join(comment_texts)
        
        return {
            'task': 'summarization',
            'input': TextCleaner.truncate_text(full_text, 2000),
            'output': issue.get('summary', ''),
            'metadata': {
                'issue_key': issue.get('key', ''),
                'comment_count': len(comments)
            }
        }
    
    @staticmethod
    def generate_classification_task(issue: Dict) -> Dict:
        """Generate issue classification task"""
        description = TextCleaner.clean_text(issue.get('description', ''))
        summary = issue.get('summary', '')
        
        input_text = f"Title: {summary}\n\nDescription: {description}"
        
        return {
            'task': 'classification',
            'input': TextCleaner.truncate_text(input_text, 1500),
            'output': {
                'status': issue.get('status', ''),
                'priority': issue.get('priority', ''),
                'labels': issue.get('labels', []),
                'components': issue.get('components', [])
            },
            'metadata': {
                'issue_key': issue.get('key', '')
            }
        }
    
    @staticmethod
    def generate_qa_pairs(issue: Dict) -> List[Dict]:
        """Generate Q&A pairs from issue and comments"""
        qa_pairs = []
        
        # Q: What is the issue? A: Summary + Description
        description = TextCleaner.clean_text(issue.get('description', ''))
        if description:
            qa_pairs.append({
                'task': 'qa',
                'input': f"What is issue {issue.get('key', '')} about?",
                'output': f"{issue.get('summary', '')}. {description[:500]}",
                'metadata': {'issue_key': issue.get('key', '')}
            })
        
        # Q: What is the status? A: Status
        qa_pairs.append({
            'task': 'qa',
            'input': f"What is the current status of {issue.get('key', '')}?",
            'output': f"The issue is currently {issue.get('status', 'Unknown')}.",
            'metadata': {'issue_key': issue.get('key', '')}
        })
        
        # Q: Who is working on this? A: Assignee
        if issue.get('assignee') and issue.get('assignee') != 'Unassigned':
            qa_pairs.append({
                'task': 'qa',
                'input': f"Who is assigned to {issue.get('key', '')}?",
                'output': f"{issue.get('assignee', 'Unknown')} is assigned to this issue.",
                'metadata': {'issue_key': issue.get('key', '')}
            })
        
        return qa_pairs
    
    @staticmethod
    def generate_resolution_prediction(issue: Dict) -> Optional[Dict]:
        """Generate task to predict resolution from description"""
        if issue.get('status') not in ['Resolved', 'Closed']:
            return None
        
        description = TextCleaner.clean_text(issue.get('description', ''))
        if not description:
            return None
        
        # Use first comment as resolution hint
        resolution_text = ""
        comments = issue.get('comments', [])
        if comments:
            resolution_text = TextCleaner.clean_text(comments[-1].get('body', ''))[:500]
        
        return {
            'task': 'resolution_prediction',
            'input': f"Issue: {issue.get('summary', '')}\n\n{description[:1000]}",
            'output': resolution_text if resolution_text else "Issue was resolved.",
            'metadata': {
                'issue_key': issue.get('key', ''),
                'status': issue.get('status', '')
            }
        }


class DataValidator:
    """Validates and filters issue data"""
    
    @staticmethod
    def is_valid_issue(issue: Dict) -> bool:
        """Check if issue has minimum required data"""
        if not issue.get('key'):
            return False
        
        if not issue.get('summary'):
            return False
        
        # Must have either description or comments
        description = issue.get('description')
        has_description = bool(description and str(description).strip())
        has_comments = len(issue.get('comments', [])) > 0
        
        return has_description or has_comments
    
    @staticmethod
    def calculate_quality_score(issue: Dict) -> float:
        """Calculate quality score for issue (0-1)"""
        score = 0.0
        
        # Has description
        description = issue.get('description')
        if description and str(description).strip():
            score += 0.3
        
        # Has comments
        comment_count = len(issue.get('comments', []))
        if comment_count > 0:
            score += min(0.3, comment_count * 0.05)
        
        # Has labels/components
        if issue.get('labels') or issue.get('components'):
            score += 0.2
        
        # Is resolved/closed
        if issue.get('status') in ['Resolved', 'Closed']:
            score += 0.2
        
        return min(1.0, score)


class LLMDataTransformer:
    """Main transformer class"""
    
    def __init__(self, output_dir: str, min_quality_score: float = 0.3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_quality_score = min_quality_score
        self.task_generator = TaskGenerator()
        self.validator = DataValidator()
        self.seen_hashes = set()
        self.stats = Counter()
    
    def _generate_hash(self, data: Dict) -> str:
        """Generate unique hash for deduplication"""
        key = data.get('issue_key', '') + data.get('task', '') + str(data.get('input', ''))
        return hashlib.md5(key.encode()).hexdigest()
    
    def transform_issue(self, project: str, issue: Dict) -> List[Dict]:
        """Transform single issue into multiple training examples"""
        
        try:
            # Validate issue
            if not self.validator.is_valid_issue(issue):
                self.stats['invalid'] += 1
                return []
            
            # Calculate quality score
            quality_score = self.validator.calculate_quality_score(issue)
            if quality_score < self.min_quality_score:
                self.stats['low_quality'] += 1
                return []
            
            training_examples = []
            
            # Add project context
            issue['project'] = project
            
            # Generate summarization task
            try:
                summary_task = self.task_generator.generate_summarization_task(issue)
                summary_task['quality_score'] = quality_score
                summary_task['project'] = project
                training_examples.append(summary_task)
            except Exception as e:
                logger.error(f"Error generating summarization task for {issue.get('key')}: {e}")
            
            # Generate classification task
            try:
                classification_task = self.task_generator.generate_classification_task(issue)
                classification_task['quality_score'] = quality_score
                classification_task['project'] = project
                training_examples.append(classification_task)
            except Exception as e:
                logger.error(f"Error generating classification task for {issue.get('key')}: {e}")
            
            # Generate Q&A pairs
            try:
                qa_pairs = self.task_generator.generate_qa_pairs(issue)
                for qa in qa_pairs:
                    qa['quality_score'] = quality_score
                    qa['project'] = project
                    training_examples.append(qa)
            except Exception as e:
                logger.error(f"Error generating Q&A tasks for {issue.get('key')}: {e}")
            
            # Generate resolution prediction (if applicable)
            try:
                resolution_task = self.task_generator.generate_resolution_prediction(issue)
                if resolution_task:
                    resolution_task['quality_score'] = quality_score
                    resolution_task['project'] = project
                    training_examples.append(resolution_task)
            except Exception as e:
                logger.error(f"Error generating resolution task for {issue.get('key')}: {e}")
            
            # Deduplicate
            unique_examples = []
            for example in training_examples:
                example_hash = self._generate_hash(example)
                if example_hash not in self.seen_hashes:
                    self.seen_hashes.add(example_hash)
                    unique_examples.append(example)
                    self.stats[example['task']] += 1
            
            self.stats['issues_processed'] += 1
            
            return unique_examples
            
        except Exception as e:
            logger.error(f"Error transforming issue {issue.get('key', 'UNKNOWN')}: {e}")
            self.stats['transform_errors'] += 1
            return []
    
    def write_jsonl(self, examples: List[Dict], filename: str):
        """Write examples to JSONL file"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get transformation statistics"""
        return dict(self.stats)
    
    def save_statistics(self):
        """Save statistics to file"""
        stats_file = self.output_dir / "transformation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.get_statistics(), f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")


if __name__ == "__main__":
    # Example usage
    transformer = LLMDataTransformer(output_dir="./output")
    
    # Sample issue data
    sample_issue = {
        'key': 'KAFKA-1234',
        'summary': 'Improve broker performance',
        'description': 'Current broker performance needs optimization...',
        'status': 'Resolved',
        'priority': 'Major',
        'assignee': 'John Doe',
        'reporter': 'Jane Smith',
        'labels': ['performance', 'broker'],
        'components': ['core'],
        'comments': [
            {'author': 'John Doe', 'body': 'Working on this...', 'created': '2024-01-01'},
            {'author': 'Jane Smith', 'body': 'Fixed in PR #123', 'created': '2024-01-02'}
        ]
    }
    
    examples = transformer.transform_issue('KAFKA', sample_issue)
    transformer.write_jsonl(examples, 'kafka_training_data.jsonl')
    
    print(f"Generated {len(examples)} training examples")
    print(f"Statistics: {transformer.get_statistics()}")