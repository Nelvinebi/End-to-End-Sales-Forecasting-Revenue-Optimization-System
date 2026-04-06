import argparse
import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import Config
from data_preprocessing import run_preprocessing
from feature_engineering import run_feature_engineering
from train import run_training
from evaluate import run_evaluation
from predict import run_prediction


class Pipeline:
    """End-to-end ML pipeline orchestrator."""
    
    def __init__(self):
        self.config = Config()
        self.start_time = None
        
    def log_stage(self, name):
        """Print stage header."""
        print("\n" + "="*70)
        print(f" STAGE: {name}")
        print("="*70)
        self.start_time = time.time()
        
    def log_complete(self):
        """Print stage completion."""
        elapsed = time.time() - self.start_time
        print(f"\n✅ Stage completed in {elapsed:.2f}s")
        
    def run_full_pipeline(self):
        """Execute all stages."""
        print("""
        
             SALES FORECASTING ML PIPELINE                            
             Rossmann Store Sales Prediction                         
        """)
        
        # Stage 1: Preprocessing
        self.log_stage("1. DATA PREPROCESSING")
        run_preprocessing(self.config)
        self.log_complete()
        
        # Stage 2: Feature Engineering
        self.log_stage("2. FEATURE ENGINEERING")
        run_feature_engineering(self.config)
        self.log_complete()
        
        # Stage 3: Training
        self.log_stage("3. MODEL TRAINING")
        run_training(self.config)
        self.log_complete()
        
        # Stage 4: Evaluation
        self.log_stage("4. MODEL EVALUATION")
        run_evaluation(self.config)
        self.log_complete()
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETE")
        print("="*70)
        print(f"\n📁 Outputs:")
        print(f"   Models:      {self.config.MODELS_DIR}/")
        print(f"   Visuals:     {self.config.VIZ_DIR}/")
        print(f"   Predictions: Use: python main.py --stage predict")
        
    def run_single_stage(self, stage):
        """Run specific stage."""
        stages = {
            'preprocess': (run_preprocessing, "DATA PREPROCESSING"),
            'features': (run_feature_engineering, "FEATURE ENGINEERING"),
            'train': (run_training, "MODEL TRAINING"),
            'evaluate': (run_evaluation, "MODEL EVALUATION"),
            'predict': (run_prediction, "PREDICTION")
        }
        
        if stage not in stages:
            print(f"❌ Unknown stage: {stage}")
            print(f"Available: {', '.join(stages.keys())}")
            return
            
        func, name = stages[stage]
        self.log_stage(name)
        func(self.config)
        self.log_complete()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sales Forecasting ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage all          # Run full pipeline
  python main.py --stage preprocess   # Only preprocessing
  python main.py --stage train        # Only training
  python main.py --stage predict      # Make predictions
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'preprocess', 'features', 'train', 'evaluate', 'predict'],
        default='all',
        help='Which pipeline stage to run (default: all)'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use small sample for quick testing'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Run
    if args.stage == 'all':
        pipeline.run_full_pipeline()
    else:
        pipeline.run_single_stage(args.stage)


if __name__ == "__main__":
    main()