import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import FlowPhotoDataset
from modules import ResNetRankNet
from utils import load_images_from_csv, evaluate_rank_predictions

import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Target device
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    params = checkpoint["params"]
    
    model = ResNetRankNet(
        input_shape=(3, params["input_shape"][0], params["input_shape"][1]),
        transforms=checkpoint["transforms"],
        resnet_size=18,
        truncate=2,
    )
    model = nn.DataParallel(model, device_ids=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully")

    return model.to(device).eval()

def predict_batch(
    model: nn.Module,
    dataset: FlowPhotoDataset,
    device: torch.device
) -> np.ndarray:
    """Generate predictions for a dataset.
    
    Args:
        model: Loaded model
        dataset: Dataset to predict
        device: Target device
        
    Returns:
        Array of predictions
    """
    predictions = []
    
    with torch.no_grad():
        for image, _ in tqdm(dataset, desc="Generating predictions"):
            # Move image to device and get prediction
            image = image.to(device)
            transformed = model.module.transforms['eval'](image)
            output = model.module.forward_single(transformed.unsqueeze(0))
            score = output.detach().cpu().numpy().item()
            predictions.append(score)
            
    return np.array(predictions)

def setup_mlflow(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """Setup MLflow tracking if available.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI of the MLflow tracking server
        run_name: Name of the run to associate metrics with
        
    Returns:
        Tuple of (whether MLflow is enabled, run ID if found)
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info("MLflow tracking URI set to: %s", tracking_uri)
        
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experiment set to: %s", experiment_name)
            
            if run_name:
                # Try to find existing run
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[client.get_experiment_by_name(experiment_name).experiment_id],
                    filter_string=f"tags.mlflow.runName = '{run_name}'"
                )
                if runs:
                    run_id = runs[0].info.run_id
                    logger.info(f"Found existing run: {run_id}")
                    return True, run_id
            
            return True, None
            
        return False, None
    except ImportError:
        logger.info("MLflow not available - metrics will not be logged")
        return False, None

def test(args: argparse.Namespace) -> Dict[str, Any]:
    """Run inference and evaluation on test dataset."""
    test_start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup MLflow if requested
    use_mlflow, run_id = setup_mlflow(
        experiment_name=args.mlflow_experiment_name,
        tracking_uri=args.mlflow_tracking_uri,
        run_name=args.run_name
    )

    # Start MLflow run if enabled
    if use_mlflow:
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=args.run_name)

    try:
        # Load test data
        data_path = Path(args.data_dir) / args.data_file
        logger.info(f"Loading test data from: {data_path}")
        df = load_images_from_csv(data_path)

        # remove rows with missing values
        n_missing = df['value'].isna().sum()
        df = df.dropna(subset=['value'])

        logger.info(f"Loaded {len(df)} test samples (removed {n_missing} missing values)")

        # Create dataset
        dataset = FlowPhotoDataset(df, args.images_dir)

        # Load model
        model_path = Path(args.model_dir) / "model.pth"
        model = load_model(model_path, device)

        # Generate predictions
        scores = predict_batch(model, dataset, device)
        df['score'] = scores

        # Compute evaluation metrics
        metrics = evaluate_rank_predictions(
            scores=scores,
            values=df['value'].values,
            n_buckets=args.num_buckets
        )
        logger.info("Evaluation metrics:")
        metric_keys = ['kendall_tau', 'spearman_rho', 'ndcg', 'rank_mae', 'map_high', 'map_low']
        for metric in metric_keys:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")

        # Log metrics to MLflow
        if use_mlflow:
            # Add 'test_' prefix to distinguish from training metrics
            mlflow.log_metrics({
                f"test_{k}": metrics[k] for k in metric_keys
            })

            mlflow.log_metrics({
                "test_n_samples": len(df)
            })
            
            # Save predictions file as artifact
            predictions_path = Path(args.output_dir) / "predictions.csv"
            df.to_csv(predictions_path, index=False)
            mlflow.log_artifact(str(predictions_path))

        # Calculate timing metrics
        total_test_time = time.time() - test_start_time
        images_per_second = len(df) / total_test_time

        # After computing predictions and metrics
        output_dict = {
            'version': '1.0',
            
            'summary': {
                'kendall_tau': float(f"{metrics['kendall_tau']:.4f}"),
                'spearman_rho': float(f"{metrics['spearman_rho']:.4f}"),
                'ndcg': float(f"{metrics['ndcg']:.4f}"),
                'rank_mae': float(f"{metrics['rank_mae']:.4f}"),
                'map_high': float(f"{metrics['map_high']:.4f}"),
                'map_low': float(f"{metrics['map_low']:.4f}")
            },
            
            'timing': {
                'total_seconds': float(f"{total_test_time:.2f}"),
                'total_formatted': format_time(total_test_time),
                'images_per_second': float(f"{images_per_second:.2f}"),
                'total_images': len(df)
            },
            
            'data': {
                'sizes': {
                    'samples': int(len(df)),
                    'missing': int(n_missing),
                    'total': int(len(df)) + int(n_missing)
                },
                'timestamps': {
                    'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'values': {
                    'min': float(f"{df['value'].min():.4f}"),
                    'max': float(f"{df['value'].max():.4f}"),
                    'mean': float(f"{df['value'].mean():.4f}"),
                    'median': float(f"{df['value'].median():.4f}")
                },
                'scores': {
                    'min': float(f"{scores.min():.4f}"),
                    'max': float(f"{scores.max():.4f}"),
                    'mean': float(f"{scores.mean():.4f}"),
                    'median': float(f"{np.median(scores):.4f}")
                }
            },
            
            'detailed_metrics': {
                'correlation': {
                    'kendall_tau': float(f"{metrics['kendall_tau']:.4f}"),
                    'spearman_rho': float(f"{metrics['spearman_rho']:.4f}"),
                    'pearson_r': float(f"{metrics['pearson_r']:.4f}") if 'pearson_r' in metrics else None
                },
                'ranking': {
                    'ndcg': float(f"{metrics['ndcg']:.4f}"),
                    'rank_mae': float(f"{metrics['rank_mae']:.4f}"),
                    'map_high': float(f"{metrics['map_high']:.4f}"),
                    'map_low': float(f"{metrics['map_low']:.4f}"),
                    'map_high_precisions': [
                        {
                            'percentile': p['percentile'],
                            'threshold': float(f"{p['threshold']:.4f}"),
                            'precision': float(f"{p['precision']:.4f}")
                        } for p in metrics['map_high_precisions']
                    ],
                    'map_low_precisions': [
                        {
                            'percentile': p['percentile'],
                            'threshold': float(f"{p['threshold']:.4f}"),
                            'precision': float(f"{p['precision']:.4f}")
                        } for p in metrics['map_low_precisions']
                    ]
                }
            },
            
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': str(model_path),
                'data_file': str(data_path)
            }
        }

        return {
            "predictions": df,
            "output": output_dict
        }
    finally:
        if use_mlflow:
            mlflow.end_run()

def format_time(seconds: float) -> str:
    """Format time in seconds to hours:minutes:seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing model checkpoint",
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results",
        default=os.environ["SM_OUTPUT_DIR"]
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images",
        default=os.environ["SM_CHANNEL_IMAGES"]
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing test CSV",
        default=os.environ["SM_CHANNEL_DATA"]
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="test-images.csv",
        help="Filename of test CSV file"
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=10,
        help="Number of buckets for NDCG calculation"
    )

    # MLflow parameters
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name to associate metrics with training run"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run test
    results = test(args)
    
    # Save predictions
    predictions_path = output_dir / "data" / "predictions.csv"
    logger.info(f"Saving predictions to: {predictions_path}")
    results["predictions"].to_csv(predictions_path, index=False)
    
    # Save metrics
    output_path = output_dir / "data" / "output.json"
    logger.info(f"Saving output to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results["output"], f, indent=2)
