import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict
import torch

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from models import PlanOutputModel, PlanInputModel, ModelConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInference:
    """Model inference class"""
    
    def __init__(self, model_type: str, model_path: str, config: ModelConfig):
        self.model_type = model_type
        self.model_path = model_path
        self.config = config
        
        if model_type == "plan_output":
            self.model = self._load_plan_output_model()
        elif model_type == "plan_input":
            self.model = self._load_plan_input_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_plan_output_model(self) -> PlanOutputModel:
        """Load Plan-Output model"""
        logger.info(f"Loading Plan-Output model: {self.model_path}")
        
        # Create model instance
        model = PlanOutputModel(self.config)
        
        # Load trained model weights
        model.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
        return model
    
    def _load_plan_input_model(self) -> PlanInputModel:
        """Load Plan-Input model"""
        logger.info(f"Loading Plan-Input model: {self.model_path}")
        
        # Create model instance
        model = PlanInputModel(self.config)
        
        # Load PG and SG modules
        pg_path = os.path.join(self.model_path, "pg")
        sg_path = os.path.join(self.model_path, "sg")
        
        if not os.path.exists(pg_path) or not os.path.exists(sg_path):
            raise FileNotFoundError(f"Plan-Input model path incomplete, requires pg and sg subdirectories: {self.model_path}")
        
        model.load_trained_models(pg_path, sg_path)
        
        return model
    
    def generate_summary(self, document: str, plan_questions: List[str] = None) -> str:
        """Generate summary"""
        return self.model.generate_summary(document, plan_questions)
    
    def generate_plan(self, document: str) -> List[str]:
        """Generate plan (only applicable to Plan-Input model)"""
        if self.model_type != "plan_input":
            raise ValueError("generate_plan method only applicable to Plan-Input model")
        return self.model.generate_plan(document)

def load_test_data(data_path: str) -> List[Dict]:
    """Load test data"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Test data should be a list containing dictionaries")
    
    return data

def save_results(results: List[Dict], output_path: str):
    """Save inference results"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")

def run_inference(args):
    """Run inference"""
    logger.info(f"Starting inference, model type: {args.model_type}")
    
    # Create configuration
    config = ModelConfig(
        max_length=args.max_length,
        batch_size=1,  # Use single sample for inference
        random_seed=args.seed
    )
    
    # Adjust generation parameters based on dataset
    if 'scinews' in args.test_data.lower():
        config.max_new_tokens = 1024
    elif 'elife' in args.test_data.lower():
        config.max_new_tokens = 512
    elif 'plos' in args.test_data.lower():
        config.max_new_tokens = 256
    else:
        config.max_new_tokens = 512
    
    # Create inference engine
    inference = ModelInference(args.model_type, args.model_path, config)
    
    # Load test data
    test_data = load_test_data(args.test_data)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Perform inference
    results = []
    for i, item in enumerate(test_data):
        logger.info(f"Processing sample {i+1}/{len(test_data)}")
        
        document = item['document']
        reference_summary = item.get('summary', '')
        plan_questions = item.get('plan_questions', [])
        
        try:
            # Generate summary
            if args.use_reference_plan and plan_questions:
                # Use reference plan
                generated_summary = inference.generate_summary(document, plan_questions)
                used_plan = plan_questions
            else:
                # Let model generate its own plan (or use no plan)
                generated_summary = inference.generate_summary(document)
                used_plan = []
                
                # If Plan-Input model, can get generated plan
                if args.model_type == "plan_input":
                    used_plan = inference.generate_plan(document)
            
            result = {
                'id': i,
                'document': document,
                'reference_summary': reference_summary,
                'generated_summary': generated_summary,
                'reference_plan': plan_questions,
                'used_plan': used_plan
            }
            
            results.append(result)
            
            # Print sample results
            if args.verbose:
                logger.info(f"Sample {i+1} results:")
                logger.info(f"Generated summary: {generated_summary[:200]}...")
                if used_plan:
                    logger.info(f"Used plan: {used_plan[:3]}...")
        
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {str(e)}")
            result = {
                'id': i,
                'document': document,
                'reference_summary': reference_summary,
                'generated_summary': "",
                'reference_plan': plan_questions,
                'used_plan': [],
                'error': str(e)
            }
            results.append(result)
    
    # Save results
    save_results(results, args.output_path)
    
    logger.info("Inference completed!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Explanatory summarization model inference")
    
    # Basic parameters
    parser.add_argument("--model_type", type=str, choices=["plan_output", "plan_input"],
                       required=True, help="Model type")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Trained model path")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Test data path")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output results path")
    
    # Inference parameters
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--use_reference_plan", action="store_true",
                       help="Whether to use reference plan (if available)")
    parser.add_argument("--verbose", action="store_true",
                       help="Whether to show detailed output")
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    # Run inference
    run_inference(args)

if __name__ == "__main__":
    main() 
