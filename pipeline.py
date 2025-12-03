"""
Main Pipeline Orchestrator
Coordinates scraping, transformation, and output generation
"""

import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict
import sys

# Import our modules
from scraper import JiraScraper, ScraperConfig
from transformer import LLMDataTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.scraper = None
        self.transformer = None
        self.start_time = None
        self.metrics = {
            'total_issues_scraped': 0,
            'total_examples_generated': 0,
            'projects_processed': 0,
            'errors': 0
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize scraper and transformer"""
        
        # Initialize scraper
        scraper_config = ScraperConfig(
            base_url=self.config['jira']['base_url'],
            projects=self.config['jira']['projects'],
            batch_size=self.config['scraper']['batch_size'],
            max_workers=self.config['scraper']['max_workers'],
            timeout=self.config['scraper']['timeout'],
            max_retries=self.config['scraper']['max_retries'],
            backoff_factor=self.config['scraper']['backoff_factor'],
            rate_limit_delay=self.config['scraper']['rate_limit_delay'],
            checkpoint_dir=self.config['output']['checkpoint_dir'],
            output_dir=self.config['output']['directory']
        )
        
        self.scraper = JiraScraper(scraper_config)
        
        # Initialize transformer
        min_quality = self.config.get('transformer', {}).get('min_quality_score', 0.3)
        self.transformer = LLMDataTransformer(
            output_dir=self.config['output']['directory'],
            min_quality_score=min_quality
        )
        
        logger.info("Pipeline components initialized")
    
    def run(self, resume: bool = False, project_filter: str = None):
        """Run the complete pipeline"""
        
        self.start_time = datetime.now()
        logger.info("=" * 80)
        logger.info("Starting Jira LLM Training Data Pipeline")
        logger.info(f"Projects: {self.config['jira']['projects']}")
        logger.info(f"Resume mode: {resume}")
        logger.info("=" * 80)
        
        try:
            self._initialize_components()
            
            # Filter projects if specified
            projects = self.config['jira']['projects']
            if project_filter:
                projects = [p for p in projects if p == project_filter]
                logger.info(f"Filtering to project: {project_filter}")
            
            # Process each project
            for project in projects:
                self._process_project(project, resume)
            
            # Save final statistics
            self._save_final_report()
            
            logger.info("=" * 80)
            logger.info("Pipeline completed successfully!")
            logger.info(f"Total issues scraped: {self.metrics['total_issues_scraped']}")
            logger.info(f"Total examples generated: {self.metrics['total_examples_generated']}")
            logger.info(f"Duration: {datetime.now() - self.start_time}")
            logger.info("=" * 80)
            
        except KeyboardInterrupt:
            logger.warning("\nPipeline interrupted by user. Progress has been saved.")
            self._save_final_report()
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.metrics['errors'] += 1
            self._save_final_report()
            raise
    
    def _process_project(self, project: str, resume: bool):
        """Process a single project"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing project: {project}")
        logger.info(f"{'='*60}")
        
        project_issues = 0
        project_examples = 0
        batch_examples = []
        batch_size = 100
        
        try:
            # Scrape and transform issues
            for issue in self.scraper.scrape_project(project, resume):
                project_issues += 1
                self.metrics['total_issues_scraped'] += 1
                
                # Transform issue into training examples
                examples = self.transformer.transform_issue(project, issue)
                batch_examples.extend(examples)
                project_examples += len(examples)
                self.metrics['total_examples_generated'] += len(examples)
                
                # Write batch to disk
                if len(batch_examples) >= batch_size:
                    filename = f"{project.lower()}_training_data.jsonl"
                    self.transformer.write_jsonl(batch_examples, filename)
                    logger.info(f"Wrote batch of {len(batch_examples)} examples to {filename}")
                    batch_examples = []
                
                # Log progress
                if project_issues % 50 == 0:
                    logger.info(f"Progress: {project_issues} issues, {project_examples} examples")
            
            # Write remaining examples
            if batch_examples:
                filename = f"{project.lower()}_training_data.jsonl"
                self.transformer.write_jsonl(batch_examples, filename)
                logger.info(f"Wrote final batch of {len(batch_examples)} examples")
            
            self.metrics['projects_processed'] += 1
            
            logger.info(f"\nProject {project} completed:")
            logger.info(f"  - Issues processed: {project_issues}")
            logger.info(f"  - Training examples: {project_examples}")
            
        except Exception as e:
            logger.error(f"Error processing project {project}: {e}")
            self.metrics['errors'] += 1
            # Continue with next project
    
    def _save_final_report(self):
        """Save final pipeline report"""
        
        report = {
            'pipeline_run': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            },
            'metrics': self.metrics,
            'transformer_stats': self.transformer.get_statistics() if self.transformer else {},
            'config': self.config
        }
        
        report_path = Path(self.config['output']['directory']) / 'pipeline_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nPipeline report saved to: {report_path}")
        
        # Also save transformer statistics
        if self.transformer:
            self.transformer.save_statistics()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Jira to LLM Training Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python pipeline.py --config config.yaml
  
  # Resume from checkpoint
  python pipeline.py --config config.yaml --resume
  
  # Process specific project
  python pipeline.py --config config.yaml --project KAFKA
  
  # Custom output directory
  python pipeline.py --config config.yaml --output ./my_output
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        help='Process only specified project (e.g., KAFKA)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Override output directory from config'
    )
    
    args = parser.parse_args()
    
    # Load and optionally override config
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    pipeline = Pipeline(args.config)
    
    # Override output directory if specified
    if args.output:
        pipeline.config['output']['directory'] = args.output
        logger.info(f"Output directory overridden to: {args.output}")
    
    # Run pipeline
    pipeline.run(resume=args.resume, project_filter=args.project)


if __name__ == "__main__":
    main()