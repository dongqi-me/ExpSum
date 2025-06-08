import argparse
import os
import sys
import json
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from models import PlanOutputModel, PlanInputModel, ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories(output_dir: str):
    """Create necessary directories"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir}")

def train_plan_output(args):
    """Train Plan-Output model"""
    logger.info("Starting Plan-Output model training...")
    
    # Create configuration
    config = ModelConfig(
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        random_seed=args.seed
    )
    
    # Adjust max new tokens based on dataset
    if 'scinews' in args.train_data.lower():
        config.max_new_tokens = 1024
    elif 'elife' in args.train_data.lower():
        config.max_new_tokens = 512
    elif 'plos' in args.train_data.lower():
        config.max_new_tokens = 256
    else:
        config.max_new_tokens = 512  # Default value
    
    # Create model
    model = PlanOutputModel(config)
    
    # Train model
    output_dir = os.path.join(args.output_dir, "plan_output_model")
    setup_directories(output_dir)
    
    model.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=output_dir
    )
    
    logger.info("Plan-Output model training completed!")

def train_plan_input(args):
    """Train Plan-Input model"""
    logger.info("Starting Plan-Input model training...")
    
    # Create configuration
    config = ModelConfig(
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        random_seed=args.seed
    )
    
    # Adjust max new tokens based on dataset
    if 'scinews' in args.train_data.lower():
        config.max_new_tokens = 1024
    elif 'elife' in args.train_data.lower():
        config.max_new_tokens = 512
    elif 'plos' in args.train_data.lower():
        config.max_new_tokens = 256
    else:
        config.max_new_tokens = 512  # Default value
    
    # Create model
    model = PlanInputModel(config)
    
    # Set output directory
    output_dir = os.path.join(args.output_dir, "plan_input_model")
    setup_directories(output_dir)
    
    # Train PG module
    logger.info("Training PG module...")
    model.train_pg(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=output_dir
    )
    
    # Train SG module
    logger.info("Training SG module...")
    model.train_sg(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=output_dir
    )
    
    logger.info("Plan-Input model training completed!")

def main():
    parser = argparse.ArgumentParser(description="Train explanatory summarization models")
    
    # Basic parameters
    parser.add_argument("--model_type", type=str, choices=["plan_output", "plan_input"], 
                       required=True, help="Model type")
    parser.add_argument("--train_data", type=str, required=True, 
                       help="Training data path")
    parser.add_argument("--val_data", type=str, required=True, 
                       help="Validation data path")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Model output directory")
    
    # Model hyperparameters
    parser.add_argument("--max_length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=15, 
                       help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=2024, 
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data file not found: {args.train_data}")
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data file not found: {args.val_data}")
    
    # Train based on model type
    if args.model_type == "plan_output":
        train_plan_output(args)
    elif args.model_type == "plan_input":
        train_plan_input(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

if __name__ == "__main__":
    main() 
