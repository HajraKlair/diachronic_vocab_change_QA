"""
Utility functions for DATR project.
"""

import os
import yaml
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np
import torch


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs", experiment_name: str = None) -> logging.Logger:
    """Setup logging configuration with immediate flushing for real-time monitoring."""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")

    # Create handlers with immediate flushing
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Ensure immediate flushing
    import sys
    sys.stdout.flush()

    return logging.getLogger(__name__)


def get_device(config: Dict[str, Any] = None) -> torch.device:
    """Get the appropriate device (CUDA or CPU)."""
    if config and config.get("hardware", {}).get("device") == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(data: List[Dict], filepath: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_date(date_str: str) -> Optional[int]:
    """
    Parse publication date string to extract year.
    Handles various formats found in ChroniclingAmericaQA.
    """
    if not date_str:
        return None
    
    try:
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y"]:
            try:
                dt = datetime.strptime(date_str[:10], fmt)
                return dt.year
            except ValueError:
                continue
        
        # Try to extract year directly
        year = int(date_str[:4])
        if 1700 <= year <= 2000:
            return year
    except (ValueError, TypeError):
        pass
    
    return None


def get_era_from_year(year: int, era_bins: List[Dict]) -> str:
    """
    Map a year to its corresponding era based on configuration bins.
    """
    if year is None:
        return "unknown"
    
    for era in era_bins:
        if era["start"] <= year <= era["end"]:
            return era["name"]
    
    return "unknown"


def compute_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def batch_iterator(data: List[Any], batch_size: int):
    """Iterate over data in batches."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary for logging."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{prefix}{key}: {value:.4f}")
        else:
            parts.append(f"{prefix}{key}: {value}")
    return " | ".join(parts)
