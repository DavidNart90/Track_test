"""
Direct script for running data ingestion.

This script provides a simple command-line interface for ingesting
market data and property listings without using the full CLI.
"""

import asyncio
import sys
import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from .data.ingestion import DataIngestionEngine
from .data.enhanced_ingestion import EnhancedDataIngestionEngine, ListingType
from .data.validation import DataValidator


class DataType(str, Enum):
    """Data types for ingestion."""
    MARKET = "market"
    PROPERTY = "property"


async def run_ingestion(args):
    """Run the ingestion process with the provided arguments."""
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load data
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]  # Convert single object to list
            
        logger.info(f"Loaded {len(data)} records from {args.file}")
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        return 1
    
    # Auto-detect data type if not specified
    data_type = args.type
    if not data_type:
        if any('region_id' in item or 'median_price' in item or 'region_name' in item for item in data):
            data_type = DataType.MARKET
        else:
            data_type = DataType.PROPERTY
        logger.info(f"Auto-detected data type: {data_type}")
    
    # Validate data if dry run
    if args.dry_run:
        validator = DataValidator()
        validation_type = 'market_data' if data_type == DataType.MARKET else 'property_listing'
        
        logger.info(f"Validating {len(data)} records as {validation_type}...")
        validation_result = await validator.validate_batch(data, validation_type)
        
        print(f"\nValidation Results:")
        print(f"  Total records: {validation_result['total_records']}")
        print(f"  Valid records: {validation_result['valid_records']}")
        print(f"  Invalid records: {validation_result['invalid_records']}")
        print(f"  Success rate: {validation_result['validation_success_rate']:.2%}")
        
        if validation_result['errors']:
            print(f"\nFirst 10 errors:")
            for error in validation_result['errors'][:10]:
                print(f"  - {error}")
        
        return 0
    
    # Initialize the appropriate engine
    if args.enhanced:
        logger.info("Using enhanced ingestion engine")
        engine = EnhancedDataIngestionEngine(batch_size=args.batch_size)
        
        # Enable RAG functionality if requested
        if args.enable_rag:
            logger.info("RAG functionality enabled - will generate embeddings and update knowledge graph")
            engine.rag_enabled = True
            
            # Set up more detailed logging for RAG components
            rag_logger = logging.getLogger("trackrealties.rag")
            rag_logger.setLevel(logging.DEBUG)
            
            # Add a console handler with a specific format
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            rag_logger.addHandler(console_handler)
            
            # Ensure OpenAI API key is set in environment
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                logger.info("OpenAI API key found in environment")
                # Update settings to ensure embedding service uses the key
                from src.trackrealties.core.config import get_settings
                settings = get_settings()
                settings.embedding_api_key = openai_api_key
            else:
                logger.warning("OpenAI API key not found in environment - RAG functionality may not work properly")
    else:
        logger.info("Using standard ingestion engine")
        engine = DataIngestionEngine(batch_size=args.batch_size)
    
    # Run ingestion
    try:
        if data_type == DataType.MARKET:
            logger.info(f"Starting market data ingestion from {args.file}")
            job = await engine.ingest_market_data(args.source, data)
        else:
            # Convert listing type string to enum if using enhanced engine
            listing_type_enum = None
            if args.enhanced:
                listing_type_enum = ListingType.BOTH
                if args.listing_type == 'sale':
                    listing_type_enum = ListingType.SALE
                elif args.listing_type == 'rental':
                    listing_type_enum = ListingType.RENTAL
                logger.info(f"Using listing type filter: {listing_type_enum.value}")
            
            logger.info(f"Starting property listings ingestion from {args.file}")
            if args.enhanced and listing_type_enum:
                job = await engine.ingest_property_listings(args.source, data, listing_type=listing_type_enum)
            else:
                job = await engine.ingest_property_listings(args.source, data)
        
        # Monitor progress
        while job.status.value in ['pending', 'running']:
            await asyncio.sleep(2)
            progress = await engine.get_job_progress(job.id)
            if progress:
                percent = progress.processed_records / progress.total_records * 100 if progress.total_records > 0 else 0
                print(f"Progress: {progress.processed_records}/{progress.total_records} ({percent:.1f}%)", end='\r')
        
        # Final status
        print("\n")  # Clear the progress line
        print(f"Ingestion completed with status: {job.status.value}")
        print(f"Processed: {job.processed_records}/{job.total_records} records")
        
        if job.failed_records > 0:
            print(f"Warning: {job.failed_records} records failed to process")
            
        if job.error_message:
            print(f"Error: {job.error_message}")
            
        # Show validation summary if using enhanced engine
        if args.enhanced and hasattr(job, 'validation_stats'):
            print("\nValidation Summary:")
            print(f"  Valid Records: {job.validation_stats.valid_records}")
            print(f"  Invalid Records: {job.validation_stats.invalid_records}")
            print(f"  Records with Warnings: {job.validation_stats.records_with_warnings}")
            print(f"  Total Errors: {job.validation_stats.error_count}")
            print(f"  Total Warnings: {job.validation_stats.warning_count}")
        
        return 0
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        return 1


async def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(description="TrackRealties Data Ingestion Tool")
    
    # Required arguments
    parser.add_argument("file", help="Path to the data file (JSON)")
    parser.add_argument("source", help="Data source identifier")
    
    # Optional arguments
    parser.add_argument("--type", "-t", choices=[t.value for t in DataType], 
                        help="Data type (market or property, default: auto-detect)")
    parser.add_argument("--listing-type", choices=["sale", "rental", "both"], default="both",
                        help="Type of property listings to ingest (default: both)")
    parser.add_argument("--batch-size", "-b", type=int, default=1000, 
                        help="Batch size for processing (default: 1000)")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Validate data without saving")
    parser.add_argument("--enhanced", action="store_true", 
                        help="Use enhanced ingestion pipeline")
    parser.add_argument("--enable-rag", action="store_true", 
                        help="Enable RAG functionality (vector embeddings and knowledge graph)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Validate file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return 1
    
    return await run_ingestion(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))