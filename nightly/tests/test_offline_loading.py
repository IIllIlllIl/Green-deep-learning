#!/usr/bin/env python3
"""
Quick offline model loading test

This script tests if all pretrained models can be loaded in offline mode
without any network connection.

Usage:
    python3 tests/test_offline_loading.py
"""

import os
import sys

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

print("=" * 80)
print("OFFLINE MODEL LOADING TEST")
print("=" * 80)
print("\nTesting with HF_HUB_OFFLINE=1 (forced offline mode)")
print("=" * 80)

success = True

# Test 1: HRNet18
print("\n[Test 1/3] Loading HRNet18...")
try:
    import timm
    model = timm.create_model('hrnet_w18', pretrained=True)
    print("✅ HRNet18 loaded successfully in offline mode")
    del model
except Exception as e:
    print(f"❌ HRNet18 failed to load: {e}")
    success = False

# Test 2: ResNet50
print("\n[Test 2/3] Loading ResNet50...")
try:
    import torchvision.models as models
    try:
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except ImportError:
        model = models.resnet50(pretrained=True)
    print("✅ ResNet50 loaded successfully in offline mode")
    del model
except Exception as e:
    print(f"❌ ResNet50 failed to load: {e}")
    success = False

# Test 3: DenseNet121
print("\n[Test 3/3] Loading DenseNet121...")
try:
    import torchvision.models as models
    try:
        from torchvision.models import DenseNet121_Weights
        model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    except ImportError:
        model = models.densenet121(pretrained=True)
    print("✅ DenseNet121 loaded successfully in offline mode")
    del model
except Exception as e:
    print(f"❌ DenseNet121 failed to load: {e}")
    success = False

print("\n" + "=" * 80)
if success:
    print("✅ ALL MODELS LOADED SUCCESSFULLY IN OFFLINE MODE")
    print("=" * 80)
    print("\nOffline training is ready!")
    print("You can now run experiments without network connection:")
    print("  export HF_HUB_OFFLINE=1")
    print("  sudo -E python3 mutation.py settings/your_config.json")
else:
    print("❌ SOME MODELS FAILED TO LOAD")
    print("=" * 80)
    print("\nPlease run the download script again:")
    print("  python3 scripts/download_pretrained_models.py")
print("=" * 80)

sys.exit(0 if success else 1)
