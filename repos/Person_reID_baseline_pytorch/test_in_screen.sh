#!/bin/bash
# Quick test script to run in screen session

cd /home/green/energy_dl/test/Person_reID_baseline_pytorch

# Run a very short training for testing (5 epochs)
./train.sh -n quick_test --total_epoch 5 2>&1 | tee test_training.log
