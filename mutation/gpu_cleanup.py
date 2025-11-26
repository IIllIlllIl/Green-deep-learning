#!/usr/bin/env python3
"""
GPU Memory Cleanup Utility

This script performs aggressive GPU memory cleanup to prevent memory
accumulation across multiple training experiments.

It combines multiple cleanup strategies:
1. Python garbage collection
2. PyTorch CUDA cache clearing
3. CUDA context reset (if possible)
4. Memory statistics reset
"""

import gc
import sys
import subprocess


def cleanup_gpu_memory(verbose=True):
    """Perform aggressive GPU memory cleanup

    Args:
        verbose: Print cleanup status messages

    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        # Try to import torch
        try:
            import torch
        except ImportError:
            if verbose:
                print("[GPU Cleanup] PyTorch not available, skipping CUDA cleanup")
            return True

        if not torch.cuda.is_available():
            if verbose:
                print("[GPU Cleanup] CUDA not available, skipping GPU cleanup")
            return True

        if verbose:
            # Show memory before cleanup
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU Cleanup] Memory before: allocated={allocated_before:.2f}GB, reserved={reserved_before:.2f}GB")

        # Step 1: Python garbage collection
        if verbose:
            print("[GPU Cleanup] Running Python garbage collection...")
        collected = gc.collect()
        if verbose:
            print(f"[GPU Cleanup]   Collected {collected} objects")

        # Step 2: Clear PyTorch CUDA cache
        if verbose:
            print("[GPU Cleanup] Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()

        # Step 3: Synchronize all CUDA operations
        if verbose:
            print("[GPU Cleanup] Synchronizing CUDA operations...")
        torch.cuda.synchronize()

        # Step 4: Reset peak memory statistics
        if verbose:
            print("[GPU Cleanup] Resetting peak memory statistics...")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Step 5: Run garbage collection again after CUDA cleanup
        gc.collect()

        if verbose:
            # Show memory after cleanup
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            print(f"[GPU Cleanup] Memory after: allocated={allocated_after:.2f}GB, reserved={reserved_after:.2f}GB")

            freed_allocated = allocated_before - allocated_after
            freed_reserved = reserved_before - reserved_after
            if freed_allocated > 0.01 or freed_reserved > 0.01:
                print(f"[GPU Cleanup] ✓ Freed: allocated={freed_allocated:.2f}GB, reserved={freed_reserved:.2f}GB")
            else:
                print(f"[GPU Cleanup] ✓ No significant memory to free")

        return True

    except Exception as e:
        if verbose:
            print(f"[GPU Cleanup] ⚠️  Error during cleanup: {e}")
        return False


def nvidia_smi_reset():
    """Attempt to reset GPU compute mode using nvidia-smi

    This is a more aggressive approach that resets the GPU compute processes.
    Requires root/sudo privileges.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return False

        # This would require sudo, so we don't actually run it
        # Just document it as an option
        print("[GPU Cleanup] Note: For more aggressive cleanup, run:")
        print("              sudo nvidia-smi --gpu-reset")
        print("              (requires stopping all GPU processes)")

        return True

    except Exception:
        return False


def main():
    """Main entry point"""
    verbose = '--quiet' not in sys.argv

    if verbose:
        print("=" * 80)
        print("GPU Memory Cleanup Utility")
        print("=" * 80)

    success = cleanup_gpu_memory(verbose=verbose)

    if verbose:
        if success:
            print("[GPU Cleanup] ✓ Cleanup completed successfully")
        else:
            print("[GPU Cleanup] ⚠️  Cleanup completed with warnings")
        print("=" * 80)

    # Show nvidia-smi info if available
    if '--show-gpu-info' in sys.argv:
        try:
            subprocess.run(['nvidia-smi'], check=False)
        except FileNotFoundError:
            print("[GPU Cleanup] nvidia-smi not found")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
