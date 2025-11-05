#!/usr/bin/env python
"""
VulBERTa Training Script
Supports both MLP and CNN model architectures
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import re
import torch
import sklearn
import os
import random
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Import custom modules
import custom
import models
import clang
from clang import *
from clang import cindex

from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
from transformers.modeling_outputs import SequenceClassifierOutput
from custom import CustomDataCollatorForLanguageModeling

# For CNN model
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from transformers import RobertaModel

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_tokenizer():
    """Initialize and return the custom tokenizer"""
    from tokenizers.pre_tokenizers import PreTokenizer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers import NormalizedString, PreTokenizedString
    from typing import List
    from tokenizers import Tokenizer
    from tokenizers import normalizers, decoders
    from tokenizers.normalizers import StripAccents, Replace
    from tokenizers.processors import TemplateProcessing
    from tokenizers import processors, pre_tokenizers
    from tokenizers.models import BPE

    class MyTokenizer:
        cidx = cindex.Index.create()

        def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
            tok = []
            tu = self.cidx.parse('tmp.c',
                           args=[''],
                           unsaved_files=[('tmp.c', str(normalized_string.original))],
                           options=0)
            for t in tu.get_tokens(extent=tu.cursor.extent):
                spelling = t.spelling.strip()
                if spelling == '':
                    continue
                tok.append(NormalizedString(spelling))
            return tok

        def pre_tokenize(self, pretok: PreTokenizedString):
            pretok.split(self.clang_split)

    # Load pre-trained tokenizers
    vocab, merges = BPE.read_file(vocab="./tokenizer/drapgh-vocab.json",
                                  merges="./tokenizer/drapgh-merges.txt")
    my_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

    my_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
    my_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
    my_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    my_tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", 0),
            ("<pad>", 1),
            ("</s>", 2),
            ("<unk>", 3),
            ("<mask>", 4)
        ]
    )

    my_tokenizer.enable_truncation(max_length=1024)
    my_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0,
                               pad_token='<pad>', length=None, pad_to_multiple_of=None)

    return my_tokenizer

def cleaner(code):
    """Remove code comments"""
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub('\n', '', code)
    code = re.sub('\t', '', code)
    return code

def process_encodings(encodings):
    """Process tokenizer encodings"""
    input_ids = []
    attention_mask = []
    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

class MyCustomDataset(Dataset):
    """Custom dataset for MLP model"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_dataset(dataset_name, my_tokenizer):
    """Load and prepare dataset"""
    if dataset_name == 'devign':
        train_index = set()
        valid_index = set()

        with open('data/finetune/devign/train.txt') as f:
            for line in f:
                line = line.strip()
                train_index.add(int(line))

        with open('data/finetune/devign/valid.txt') as f:
            for line in f:
                line = line.strip()
                valid_index.add(int(line))

        mydata = pd.read_json('data/finetune/devign/Devign.json')
        m1 = mydata.iloc[list(train_index)]
        m2 = mydata.iloc[list(valid_index)]

        mydata = None
        del mydata

        m1.func = m1.func.apply(cleaner)
        m2.func = m2.func.apply(cleaner)

        train_encodings = my_tokenizer.encode_batch(m1.func)
        train_encodings = process_encodings(train_encodings)

        val_encodings = my_tokenizer.encode_batch(m2.func)
        val_encodings = process_encodings(val_encodings)

        train_labels = m1.target.tolist()
        val_labels = m2.target.tolist()

    elif dataset_name == 'd2a':
        task = 'function'
        m1 = pd.read_csv(f'data/finetune/{dataset_name}/{task}/d2a_lbv1_{task}_train.csv')
        m2 = pd.read_csv(f'data/finetune/{dataset_name}/{task}/d2a_lbv1_{task}_dev.csv')

        m1.code = m1.code.apply(cleaner)
        train_encodings = my_tokenizer.encode_batch(m1.code)
        train_encodings = process_encodings(train_encodings)

        m2.code = m2.code.apply(cleaner)
        val_encodings = my_tokenizer.encode_batch(m2.code)
        val_encodings = process_encodings(val_encodings)

        train_labels = m1.label.tolist()
        val_labels = m2.label.tolist()

    else:
        # draper, reveal, mvd, vuldeepecker
        m1 = pd.read_pickle(f'data/finetune/{dataset_name}/{dataset_name}_train.pkl')
        m2 = pd.read_pickle(f'data/finetune/{dataset_name}/{dataset_name}_val.pkl')

        m1.functionSource = m1.functionSource.apply(cleaner)
        m2.functionSource = m2.functionSource.apply(cleaner)

        if dataset_name == 'draper':
            m1['target'] = m1['combine'] * 1
            m2['target'] = m2['combine'] * 1

        train_encodings = my_tokenizer.encode_batch(m1.functionSource)
        train_encodings = process_encodings(train_encodings)

        val_encodings = my_tokenizer.encode_batch(m2.functionSource)
        val_encodings = process_encodings(val_encodings)

        try:
            train_labels = m1.target.tolist()
            val_labels = m2.target.tolist()
        except:
            train_labels = m1.label.tolist()
            val_labels = m2.label.tolist()

    return train_encodings, val_encodings, train_labels, val_labels, m1

def train_mlp(args):
    """Train VulBERTa-MLP model"""
    print("=" * 80)
    print("Training VulBERTa-MLP Model")
    print("=" * 80)

    # Set device
    if args.cpu:
        device = torch.device("cpu")
        print("Device: cpu (forced by --cpu flag)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU: {gpu_name}")
            except:
                pass

    # Setup seed
    setup_seed(args.seed)

    # Disable wandb
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'dryrun'

    # Get tokenizer
    print("Loading tokenizer...")
    my_tokenizer = get_tokenizer()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    train_encodings, val_encodings, train_labels, val_labels, m1 = load_dataset(args.dataset, my_tokenizer)

    # Create datasets
    train_dataset = MyCustomDataset(train_encodings, train_labels)
    val_dataset = MyCustomDataset(val_encodings, val_labels)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Load pretrained model
    print("Loading pretrained VulBERTa model...")
    model = RobertaForSequenceClassification.from_pretrained('./models/VulBERTa/')
    print(f"Model parameters: {model.num_parameters()}")

    # Compute class weights
    try:
        cw = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=[0, 1], y=train_labels)
    except:
        cw = sklearn.utils.class_weight.compute_class_weight(
            class_weight='balanced', classes=[0, 1], y=m1.target.tolist())

    c_weights = torch.FloatTensor([cw[0], cw[1]])
    criterion = torch.nn.CrossEntropyLoss(weight=c_weights)
    criterion.to(device)

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Training arguments
    output_dir = f"models/VB-MLP_{args.dataset}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=20,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        report_to=None,
        load_best_model_at_end=True
    )

    # Create trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")

    train_result = trainer.train()

    # Save final model
    trainer.save_model()

    return train_result, trainer

def train_cnn(args):
    """Train VulBERTa-CNN model"""
    print("=" * 80)
    print("Training VulBERTa-CNN Model")
    print("=" * 80)
    print("CNN training not yet implemented in this script")
    print("Please use the Jupyter notebook: Finetuning+evaluation_VulBERTa-CNN.ipynb")
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Train VulBERTa models')

    # Required arguments
    parser.add_argument('-n', '--model_name', type=str, required=True,
                       choices=['mlp', 'cnn'],
                       help='Model architecture (mlp or cnn)')

    parser.add_argument('-d', '--dataset', type=str, required=True,
                       choices=['devign', 'draper', 'reveal', 'mvd', 'vuldeepecker', 'd2a'],
                       help='Dataset to train on')

    # Optional arguments with defaults matching original code
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: 2 for MLP, 128 for CNN)')

    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: 10 for MLP, 20 for CNN)')

    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (default: 3e-05 for MLP, 0.0005 for CNN)')

    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay (default: 0.0)')

    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: 42 for MLP, 1234 for CNN)')

    parser.add_argument('--fp16', action='store_true', default=None,
                       help='Use mixed precision training (default: True for MLP, False for CNN)')

    parser.add_argument('--cpu', action='store_true',
                       help='Force training on CPU (useful for CUDA compatibility issues)')

    args = parser.parse_args()

    # Set model-specific defaults
    if args.model_name == 'mlp':
        if args.batch_size is None:
            args.batch_size = 2  # Recommended: 2 to avoid OOM on RTX 3080. Original: 4
        if args.epochs is None:
            args.epochs = 10
        if args.learning_rate is None:
            args.learning_rate = 3e-05
        if args.weight_decay is None:
            args.weight_decay = 0.0
        if args.seed is None:
            args.seed = 42
        if args.fp16 is None:
            args.fp16 = True if not args.cpu else False  # Disable fp16 on CPU
    else:  # cnn
        if args.batch_size is None:
            args.batch_size = 128  # Original: 128, may need to reduce
        if args.epochs is None:
            args.epochs = 20
        if args.learning_rate is None:
            args.learning_rate = 0.0005
        if args.weight_decay is None:
            args.weight_decay = 0.0
        if args.seed is None:
            args.seed = 1234
        if args.fp16 is None:
            args.fp16 = False

    # Force disable fp16 if using CPU
    if args.cpu and args.fp16:
        print("Warning: FP16 training is not supported on CPU. Disabling fp16.")
        args.fp16 = False

    # Record start time
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Train model
    errors = []
    try:
        if args.model_name == 'mlp':
            train_result, trainer = train_mlp(args)
        else:
            train_result, trainer = train_cnn(args)
    except Exception as e:
        errors.append(str(e))
        import traceback
        errors.append(traceback.format_exc())
        print(f"\nError during training: {e}")
        traceback.print_exc()
        train_result = None
        trainer = None

    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time

    # Print training report
    print("\n" + "=" * 80)
    print("TRAINING REPORT")
    print("=" * 80)
    print(f"Model: VulBERTa-{args.model_name.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\nHyperparameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Seed: {args.seed}")
    print(f"  FP16: {args.fp16}")

    if errors:
        print(f"\n{'='*80}")
        print("ERRORS ENCOUNTERED:")
        print("="*80)
        for error in errors:
            print(error)
    else:
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)

        if train_result and trainer:
            print(f"\nFinal training loss: {train_result.training_loss:.4f}")

            # Get evaluation metrics
            eval_results = trainer.evaluate()
            print(f"\nValidation Metrics:")
            for key, value in eval_results.items():
                print(f"  {key}: {value}")

    print("=" * 80)

if __name__ == '__main__':
    main()
