#!/usr/bin/env python
"""
Training wrapper script for DNN and RVSM bug localization models.
Supports command-line arguments for hyperparameter tuning.
"""

import argparse
import sys
import os
import time
import traceback
from datetime import datetime

# Add src directory to Python path and change to src directory for correct relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)
os.chdir(src_dir)

from dnn_model import dnn_model_kfold, train_dnn, kfold_split_indexes, oversample, features_and_labels
from rvsm_model import rsvm_model
from util import csv2dict, helper_collections
from sklearn.neural_network import MLPRegressor
from joblib import Parallel, delayed
import numpy as np


def train_dnn_custom(i, num_folds, samples, start, finish, sample_dict, bug_reports, br2files_dict,
                     hidden_sizes, alpha, max_iter, n_iter_no_change, solver):
    """Custom DNN training function with configurable hyperparameters"""
    print("Fold: {} / {}".format(i + 1, num_folds), end="\r")

    from dnn_model import kfold_split, topk_accuarcy, oversample, features_and_labels

    train_samples, test_bug_reports = kfold_split(bug_reports, samples, start, finish)
    train_samples = oversample(train_samples)
    np.random.shuffle(train_samples)
    X_train, y_train = features_and_labels(train_samples)

    clf = MLPRegressor(
        solver=solver,
        alpha=alpha,
        hidden_layer_sizes=hidden_sizes,
        random_state=1,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
    )
    clf.fit(X_train, y_train.ravel())

    acc_dict = topk_accuarcy(test_bug_reports, sample_dict, br2files_dict, clf=clf)

    return acc_dict


def dnn_model_custom(k=10, hidden_sizes=(300,), alpha=1e-5, max_iter=10000,
                     n_iter_no_change=30, solver="sgd", n_jobs=-2):
    """Run kfold cross validation with custom hyperparameters"""
    samples = csv2dict("../data/features.csv")
    sample_dict, bug_reports, br2files_dict = helper_collections(samples)
    np.random.shuffle(samples)

    # K-fold Cross Validation in parallel
    acc_dicts = Parallel(n_jobs=n_jobs)(
        delayed(train_dnn_custom)(
            i, k, samples, start, step, sample_dict, bug_reports, br2files_dict,
            hidden_sizes, alpha, max_iter, n_iter_no_change, solver
        )
        for i, (start, step) in enumerate(kfold_split_indexes(k, len(samples)))
    )

    # Calculating the average accuracy from all folds
    avg_acc_dict = {}
    for key in acc_dicts[0].keys():
        avg_acc_dict[key] = round(sum([d[key] for d in acc_dicts]) / len(acc_dicts), 3)

    return avg_acc_dict


def main():
    parser = argparse.ArgumentParser(description='Train bug localization models (DNN or RVSM)')
    parser.add_argument('-n', '--model_name', type=str, required=True,
                        choices=['dnn', 'rvsm'],
                        help='Model to train: dnn or rvsm')

    # DNN hyperparameters
    parser.add_argument('--kfold', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[300],
                        help='Hidden layer sizes (default: 300)')
    parser.add_argument('--alpha', type=float, default=1e-5,
                        help='L2 penalty parameter (default: 1e-5)')
    parser.add_argument('--max_iter', type=int, default=10000,
                        help='Maximum number of iterations (default: 10000)')
    parser.add_argument('--n_iter_no_change', type=int, default=30,
                        help='Max epochs with no improvement (default: 30)')
    parser.add_argument('--solver', type=str, default='sgd',
                        choices=['sgd', 'adam', 'lbfgs'],
                        help='Optimizer (default: sgd)')
    parser.add_argument('--n_jobs', type=int, default=-2,
                        help='Number of parallel jobs (-2 = all cores but one, default: -2)')

    args = parser.parse_args()

    # Training report
    report = {
        'model_name': args.model_name,
        'start_time': None,
        'end_time': None,
        'duration': None,
        'results': None,
        'errors': []
    }

    print("=" * 80)
    print(f"Bug Localization Model Training - {args.model_name.upper()}")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    report['start_time'] = datetime.now()
    start_timestamp = time.time()

    try:
        if args.model_name == 'dnn':
            print("DNN Hyperparameters:")
            print(f"  - K-fold: {args.kfold}")
            print(f"  - Hidden layer sizes: {tuple(args.hidden_sizes)}")
            print(f"  - Alpha (L2 penalty): {args.alpha}")
            print(f"  - Max iterations: {args.max_iter}")
            print(f"  - Early stopping patience: {args.n_iter_no_change}")
            print(f"  - Solver: {args.solver}")
            print(f"  - Parallel jobs: {args.n_jobs}")
            print()
            print("Training DNN model with k-fold cross-validation...")
            print("-" * 80)

            results = dnn_model_custom(
                k=args.kfold,
                hidden_sizes=tuple(args.hidden_sizes),
                alpha=args.alpha,
                max_iter=args.max_iter,
                n_iter_no_change=args.n_iter_no_change,
                solver=args.solver,
                n_jobs=args.n_jobs
            )
            report['results'] = results

        elif args.model_name == 'rvsm':
            print("RVSM Model:")
            print("  - Using rVSM cosine similarity baseline")
            print()
            print("Training RVSM model...")
            print("-" * 80)

            results = rsvm_model()
            report['results'] = results

        print()
        print("Training completed successfully!")

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        print(f"\n{error_msg}")
        print("\nTraceback:")
        traceback.print_exc()
        report['errors'].append(error_msg)
        report['errors'].append(traceback.format_exc())

    report['end_time'] = datetime.now()
    report['duration'] = time.time() - start_timestamp

    # Print training report
    print()
    print("=" * 80)
    print("TRAINING REPORT")
    print("=" * 80)
    print(f"Model: {report['model_name'].upper()}")
    print(f"Start time: {report['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {report['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {report['duration']:.2f} seconds ({report['duration']/60:.2f} minutes)")
    print()

    if report['errors']:
        print("ERRORS ENCOUNTERED:")
        for error in report['errors']:
            print(error)
        print()
        return 1

    if report['results']:
        print("MODEL PERFORMANCE (Top-k Accuracy):")
        print("-" * 80)
        for k in [1, 5, 10, 20]:
            if k in report['results']:
                print(f"  Top-{k:2d} Accuracy: {report['results'][k]:.3f} ({report['results'][k]*100:.1f}%)")
        print()

        print("Detailed Results (All k values):")
        for k, acc in sorted(report['results'].items()):
            print(f"  Top-{k:2d}: {acc:.3f}")

    print("=" * 80)

    return 0 if not report['errors'] else 1


if __name__ == "__main__":
    sys.exit(main())
