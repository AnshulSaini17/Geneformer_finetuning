"""
Geneformer Cell Type Classification Training Script
Fine-tunes Geneformer V1 model for cardiomyocyte subtype classification
"""

import os
import sys
import pickle
from datetime import datetime
from datasets import Dataset
from geneformer import Classifier


# Configuration
BASE_DIR = "/Users/anshul/Desktop/Seminar"
MODEL_PATH = "ctheodoris/Geneformer"
DATA_FILE = f"{BASE_DIR}/dataset.arrow"
OUTPUT_PREFIX = "cardiomyocyte_classifier"
MAX_CELLS = 50000

# Cell types to classify
CELL_TYPES = ["Cardiomyocyte1", "Cardiomyocyte2", "Cardiomyocyte3"]

# Training hyperparameters
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 500,
    "weight_decay": 0.001,
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 1000,
    "eval_steps": 1000,
    "evaluation_strategy": "steps",
    "load_best_model_at_end": True,
    "report_to": "none",
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
}


def setup_environment():
    os.environ['WANDB_DISABLED'] = 'true'
    
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir = f"{BASE_DIR}/output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def prepare_dataset(data_file, output_dir):
    """Load and prepare dataset for training"""
    print(f"Loading dataset from {data_file}")
    dataset = Dataset.from_file(data_file)
    print(f"Loaded {len(dataset)} samples")
    
    dataset_dir = f"{BASE_DIR}/dataset_directory"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset.save_to_disk(dataset_dir)
    print(f"Dataset saved to {dataset_dir}")
    
    return dataset_dir


def initialize_classifier(output_dir, training_config):
    training_config["output_dir"] = output_dir
    
    classifier = Classifier(
        classifier="cell",
        cell_state_dict={
            "state_key": "cell_type",
            "states": CELL_TYPES
        },
        training_args=training_config,
        max_ncells=MAX_CELLS,
        freeze_layers=0,
        model_version="V1",
        nproc=1,
    )
    
    return classifier


def train_model(classifier, dataset_dir, output_dir, output_prefix):
    print("\nPreparing training data...")
    classifier.prepare_data(
        input_data_file=dataset_dir,
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    id_class_dict_file = f"{output_dir}/{output_prefix}_id_class_dict.pkl"
    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)
    print(f"Prepared {len(id_class_dict)} samples across {len(set(id_class_dict.values()))} classes")
    
    print("\nStarting training...")
    metrics = classifier.validate(
        model_directory=MODEL_PATH,
        prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled_train.dataset",
        id_class_dict_file=id_class_dict_file,
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    return metrics


def evaluate_model(output_dir, output_prefix):
    print("\nEvaluating model...")
    
    evaluator = Classifier(classifier="cell", model_version="V1")
    
    test_metrics = evaluator.evaluate_saved_model(
        model_directory=f"{output_dir}/ksplit1/",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        test_data_file=f"{output_dir}/{output_prefix}_labeled_test.dataset",
        output_directory=output_dir,
        output_prefix=output_prefix,
        predict=True,
    )
    
    evaluator.plot_conf_mat(
        conf_mat_dict=test_metrics,
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    evaluator.plot_predictions(
        predictions_file=f"{output_dir}/{output_prefix}_ksplit1_predictions.pkl",
        id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    return test_metrics


def main():
    print("="*60)
    print("Geneformer Cell Type Classification")
    print("="*60)
    
    output_dir = setup_environment()
    print(f"Output directory: {output_dir}")
    
    dataset_dir = prepare_dataset(DATA_FILE, output_dir)
    
    classifier = initialize_classifier(output_dir, TRAINING_CONFIG)
    print("Classifier initialized")
    
    train_metrics = train_model(classifier, dataset_dir, output_dir, OUTPUT_PREFIX)
    print(f"\nTraining complete. Validation metrics: {train_metrics}")
    
    #test_metrics = evaluate_model(output_dir, OUTPUT_PREFIX)
    #print(f"\nTest metrics: {test_metrics}")
    
    #print(f"\nAll outputs saved to: {output_dir}")
    #print("="*60)


if __name__ == "__main__":
    main()

