"""
Jira Data Scraper for LLM Training Pipeline
Handles data extraction with fault tolerance and state recovery
"""

import requests
import time
import json
import logging
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Configuration for the scraper"""
    base_url: str
    projects: List[str]
    batch_size: int = 50
    max_workers: int = 5
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 2.0
    rate_limit_delay: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./output"


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_second: float = 10.0):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.min_delay = 1.0 / requests_per_second
    
    def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens < 1:
            sleep_time = (1 - self.tokens) / self.rate
            time.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1
        
        time.sleep(self.min_delay)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failures} failures")
            raise e


class CheckpointManager:
    """Manages scraping state persistence"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, project: str, state: Dict):
        """Save current scraping state"""
        checkpoint_file = self.checkpoint_dir / f"{project}_checkpoint.json"
        state['timestamp'] = datetime.now().isoformat()
        
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Checkpoint saved for {project}: {state.get('last_issue_key', 'N/A')}")
    
    def load_checkpoint(self, project: str) -> Optional[Dict]:
        """Load last checkpoint for project"""
        checkpoint_file = self.checkpoint_dir / f"{project}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Resumed from checkpoint: {project}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, project: str):
        """Clear checkpoint after successful completion"""
        checkpoint_file = self.checkpoint_dir / f"{project}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()


class JiraScraper:
    """Main scraper class for Apache Jira"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = RateLimiter(requests_per_second=1.0 / config.rate_limit_delay)
        self.circuit_breaker = CircuitBreaker()
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _create_session(self) -> requests.Session:
        """Create configured requests session"""
        session = requests.Session()
        session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Apache-Jira-LLM-Scraper/1.0'
        })
        return session
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """Make HTTP request with retry logic"""
        self.rate_limiter.acquire()
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.circuit_breaker.call(
                    self.session.get,
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                elif response.status_code >= 500:
                    wait_time = self.config.backoff_factor ** attempt
                    logger.warning(f"Server error {response.status_code}. Retry in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"Request failed: {response.status_code} - {response.text[:200]}")
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout. Attempt {attempt + 1}/{self.config.max_retries}")
                time.sleep(self.config.backoff_factor ** attempt)
            
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error: {e}")
                time.sleep(self.config.backoff_factor ** attempt)
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
        
        return None
    
    def _fetch_issues(self, project: str, start_at: int = 0) -> Optional[Dict]:
        """Fetch issues for a project with pagination"""
        # Use the correct Jira REST API endpoint
        url = f"{self.config.base_url}/rest/api/2/search"
        
        params = {
            'jql': f'project = {project} ORDER BY created ASC',
            'startAt': start_at,
            'maxResults': self.config.batch_size,
            'fields': 'summary,description,status,priority,assignee,reporter,labels,created,updated,comment,issuelinks,components,versions'
        }
        
        logger.debug(f"Fetching: {url} with params: {params}")
        
        return self._make_request(url, params)
    
    def _parse_issue(self, issue_data: Dict) -> Dict:
        """Parse and normalize issue data"""
        try:
            fields = issue_data.get('fields', {})
            
            # Extract comments
            comments = []
            comment_data = fields.get('comment', {})
            if comment_data:
                for comment in comment_data.get('comments', []):
                    comments.append({
                        'author': comment.get('author', {}).get('displayName', 'Unknown'),
                        'body': comment.get('body', ''),
                        'created': comment.get('created', '')
                    })
            
            # Extract linked issues
            links = []
            for link in fields.get('issuelinks', []):
                link_type = link.get('type', {}).get('name', '')
                if 'outwardIssue' in link:
                    links.append({
                        'type': link_type,
                        'key': link['outwardIssue'].get('key', ''),
                        'direction': 'outward'
                    })
                elif 'inwardIssue' in link:
                    links.append({
                        'type': link_type,
                        'key': link['inwardIssue'].get('key', ''),
                        'direction': 'inward'
                    })
            
            # Safe extraction of nested fields
            assignee_name = 'Unassigned'
            if fields.get('assignee'):
                assignee_name = fields['assignee'].get('displayName', 'Unassigned')
            
            reporter_name = 'Unknown'
            if fields.get('reporter'):
                reporter_name = fields['reporter'].get('displayName', 'Unknown')
            
            status_name = 'Unknown'
            if fields.get('status'):
                status_name = fields['status'].get('name', 'Unknown')
            
            priority_name = 'None'
            if fields.get('priority'):
                priority_name = fields['priority'].get('name', 'None')
            
            return {
                'key': issue_data.get('key', ''),
                'summary': fields.get('summary', ''),
                'description': fields.get('description', ''),
                'status': status_name,
                'priority': priority_name,
                'assignee': assignee_name,
                'reporter': reporter_name,
                'labels': fields.get('labels', []),
                'components': [c.get('name', '') for c in fields.get('components', [])],
                'versions': [v.get('name', '') for v in fields.get('versions', [])],
                'created': fields.get('created', ''),
                'updated': fields.get('updated', ''),
                'comments': comments,
                'linked_issues': links,
                'comment_count': len(comments)
            }
        
        except Exception as e:
            logger.error(f"Error parsing issue {issue_data.get('key', 'UNKNOWN')}: {e}")
            return None
    
    def scrape_project(self, project: str, resume: bool = False) -> Generator[Dict, None, None]:
        """Scrape all issues from a project"""
        logger.info(f"Starting scrape for project: {project}")
        
        # Load checkpoint if resuming
        start_at = 0
        if resume:
            checkpoint = self.checkpoint_manager.load_checkpoint(project)
            if checkpoint:
                start_at = checkpoint.get('start_at', 0)
        
        total_scraped = 0
        
        while True:
            response = self._fetch_issues(project, start_at)
            
            if not response:
                logger.error(f"Failed to fetch issues for {project} at offset {start_at}")
                break
            
            issues = response.get('issues', [])
            total = response.get('total', 0)
            
            if not issues:
                break
            
            logger.info(f"Processing {len(issues)} issues ({start_at}/{total})")
            
            for issue_data in issues:
                parsed = self._parse_issue(issue_data)
                if parsed:
                    yield parsed
                    total_scraped += 1
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(project, {
                'start_at': start_at + len(issues),
                'total_scraped': total_scraped,
                'last_issue_key': issues[-1].get('key', '')
            })
            
            start_at += len(issues)
            
            if start_at >= total:
                break
        
        logger.info(f"Completed scraping {project}: {total_scraped} issues")
        self.checkpoint_manager.clear_checkpoint(project)
    
    def scrape_all_projects(self, resume: bool = False):
        """Scrape all configured projects"""
        for project in self.config.projects:
            try:
                for issue in self.scrape_project(project, resume):
                    yield project, issue
            except Exception as e:
                logger.error(f"Failed to scrape project {project}: {e}")
                continue


if __name__ == "__main__":
    # Example usage
    config = ScraperConfig(
        base_url="https://issues.apache.org/jira",
        projects=["KAFKA", "SPARK", "HADOOP"],
        batch_size=50,
        rate_limit_delay=1.0
    )
    
    scraper = JiraScraper(config)
    
    for project, issue in scraper.scrape_all_projects(resume=False):
        print(f"[{project}] {issue['key']}: {issue['summary'][:50]}...")