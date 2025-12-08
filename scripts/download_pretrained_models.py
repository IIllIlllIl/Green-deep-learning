#!/usr/bin/env python3
"""
Pre-download script for HuggingFace model weights

This script downloads all required pretrained model weights to local cache,
enabling offline training experiments.

Models that require pretrained weights:
- Person_reID_baseline_pytorch/hrnet18: timm/hrnet_w18
- Person_reID_baseline_pytorch/densenet121: torchvision densenet121
- Person_reID_baseline_pytorch/resnet50: torchvision resnet50

Usage:
    python3 scripts/download_pretrained_models.py

Environment:
    - Requires internet connection
    - Should be run in the reid_baseline conda environment
    - Models will be cached in ~/.cache/huggingface/ and ~/.cache/torch/

Author: Claude Code
Date: 2025-11-18
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))


def download_timm_models():
    """Download timm models (HRNet)"""
    print("\n" + "=" * 80)
    print("Downloading timm models...")
    print("=" * 80)

    try:
        import timm

        # List of timm models needed
        timm_models = [
            'hrnet_w18',  # Used by Person_reID_baseline_pytorch/hrnet18
        ]

        for model_name in timm_models:
            print(f"\nDownloading: {model_name}")
            print(f"{'─' * 80}")

            try:
                # Create model with pretrained=True to download weights
                model = timm.create_model(model_name, pretrained=True)
                print(f"✅ Successfully downloaded: {model_name}")

                # Clean up
                del model

            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                return False

        print(f"\n✅ All timm models downloaded successfully")
        return True

    except ImportError:
        print("❌ Error: timm library not found")
        print("   Please install: pip install timm")
        return False


def download_torchvision_models():
    """Download torchvision models (ResNet, DenseNet)"""
    print("\n" + "=" * 80)
    print("Downloading torchvision models...")
    print("=" * 80)

    try:
        import torchvision.models as models

        # List of torchvision models needed
        torchvision_models = [
            ('resnet50', models.resnet50),      # Used by Person_reID_baseline_pytorch
            ('densenet121', models.densenet121), # Used by Person_reID_baseline_pytorch
        ]

        for model_name, model_fn in torchvision_models:
            print(f"\nDownloading: {model_name}")
            print(f"{'─' * 80}")

            try:
                # Create model with pretrained=True to download weights
                # Note: torchvision v0.13+ uses weights parameter
                try:
                    # New API (torchvision >= 0.13)
                    from torchvision.models import ResNet50_Weights, DenseNet121_Weights
                    if model_name == 'resnet50':
                        model = model_fn(weights=ResNet50_Weights.IMAGENET1K_V1)
                    elif model_name == 'densenet121':
                        model = model_fn(weights=DenseNet121_Weights.IMAGENET1K_V1)
                except ImportError:
                    # Old API (torchvision < 0.13)
                    model = model_fn(pretrained=True)

                print(f"✅ Successfully downloaded: {model_name}")

                # Clean up
                del model

            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
                return False

        print(f"\n✅ All torchvision models downloaded successfully")
        return True

    except ImportError:
        print("❌ Error: torchvision library not found")
        print("   Please install: pip install torchvision")
        return False


def verify_cache():
    """Verify downloaded models in cache"""
    print("\n" + "=" * 80)
    print("Verifying model cache...")
    print("=" * 80)

    # Check HuggingFace cache
    hf_cache = Path.home() / ".cache" / "huggingface"
    if hf_cache.exists():
        hf_size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file())
        print(f"\n✅ HuggingFace cache: {hf_cache}")
        print(f"   Size: {hf_size / (1024**3):.2f} GB")
    else:
        print(f"\n⚠️  HuggingFace cache not found: {hf_cache}")

    # Check PyTorch cache
    torch_cache = Path.home() / ".cache" / "torch"
    if torch_cache.exists():
        torch_size = sum(f.stat().st_size for f in torch_cache.rglob('*') if f.is_file())
        print(f"\n✅ PyTorch cache: {torch_cache}")
        print(f"   Size: {torch_size / (1024**3):.2f} GB")
    else:
        print(f"\n⚠️  PyTorch cache not found: {torch_cache}")

    # List cached models
    if hf_cache.exists():
        print(f"\nCached HuggingFace models:")
        hub_dir = hf_cache / "hub"
        if hub_dir.exists():
            models = [d.name for d in hub_dir.iterdir() if d.is_dir()]
            for model in sorted(models):
                print(f"  - {model}")
        else:
            print(f"  (none)")

    if torch_cache.exists():
        print(f"\nCached PyTorch models:")
        checkpoints_dir = torch_cache / "hub" / "checkpoints"
        if checkpoints_dir.exists():
            models = [f.name for f in checkpoints_dir.iterdir() if f.is_file()]
            for model in sorted(models):
                print(f"  - {model}")
        else:
            print(f"  (none)")


def test_offline_loading():
    """Test loading models in offline mode"""
    print("\n" + "=" * 80)
    print("Testing offline model loading...")
    print("=" * 80)

    # Temporarily disable network to test offline loading
    print("\nℹ️  Testing with HF_HUB_OFFLINE=1 environment variable")
    os.environ['HF_HUB_OFFLINE'] = '1'

    try:
        import timm
        print("\nTesting timm hrnet_w18...")
        try:
            model = timm.create_model('hrnet_w18', pretrained=True)
            print("✅ Successfully loaded hrnet_w18 in offline mode")
            del model
        except Exception as e:
            print(f"❌ Failed to load hrnet_w18 in offline mode: {e}")

        import torchvision.models as models
        print("\nTesting torchvision models...")
        try:
            model = models.resnet50(pretrained=True)
            print("✅ Successfully loaded resnet50 in offline mode")
            del model
        except:
            try:
                from torchvision.models import ResNet50_Weights
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                print("✅ Successfully loaded resnet50 in offline mode")
                del model
            except Exception as e:
                print(f"❌ Failed to load resnet50 in offline mode: {e}")

    finally:
        # Re-enable network
        del os.environ['HF_HUB_OFFLINE']


def main():
    """Main download process"""
    print("\n" + "=" * 80)
    print("PRETRAINED MODEL DOWNLOAD SCRIPT")
    print("=" * 80)
    print("\nThis script will download all required pretrained model weights")
    print("to local cache for offline training experiments.")
    print("\nRequired disk space: ~2-3 GB")
    print("Required network: Active internet connection")
    print("=" * 80)

    # Confirm with user
    response = input("\nProceed with download? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("\nDownload cancelled.")
        return

    # Download models
    success = True

    # Download timm models
    if not download_timm_models():
        success = False
        print("\n⚠️  Warning: Some timm models failed to download")

    # Download torchvision models
    if not download_torchvision_models():
        success = False
        print("\n⚠️  Warning: Some torchvision models failed to download")

    # Verify cache
    verify_cache()

    # Test offline loading
    test_offline_loading()

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY")
        print("=" * 80)
        print("\nYou can now run experiments offline.")
        print("The downloaded models are cached in:")
        print(f"  - {Path.home() / '.cache' / 'huggingface'}")
        print(f"  - {Path.home() / '.cache' / 'torch'}")
        print("\nTo run experiments offline, ensure these cache directories")
        print("are accessible from the training environment.")
    else:
        print("⚠️  DOWNLOAD COMPLETED WITH WARNINGS")
        print("=" * 80)
        print("\nSome models may not have downloaded correctly.")
        print("Please check the error messages above and retry if needed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
