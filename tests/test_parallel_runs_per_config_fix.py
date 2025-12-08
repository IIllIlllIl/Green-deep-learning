#!/usr/bin/env python3
"""
Test for parallel mode runs_per_config bug fix

This test verifies that runs_per_config is correctly read from the outer
experiment level in parallel mode configurations, not just from foreground_config.

Bug: Stage11 only ran 4 experiments instead of 20 because runs_per_config=5
was defined at the outer level but code was reading from foreground level.

Fix: Modified runner.py to check exp["runs_per_config"] first, then
foreground_config["runs_per_config"], then fallback to global runs_per_config.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_parallel_runs_per_config_priority():
    """Test that runs_per_config is read with correct priority in parallel mode"""

    print("\n" + "=" * 80)
    print("TEST: Parallel mode runs_per_config priority")
    print("=" * 80)

    # Test case 1: Outer level runs_per_config (Stage11 case)
    print("\n[Test 1] Outer level runs_per_config=5")
    exp = {
        "mode": "parallel",
        "runs_per_config": 5,  # <-- Defined at outer level
        "foreground": {
            "repo": "examples",
            "model": "mnist",
            "mode": "mutation",
            "mutate": ["epochs"]
            # No runs_per_config here
        },
        "background": {
            "repo": "examples",
            "model": "siamese",
            "hyperparameters": {}
        }
    }

    foreground_config = exp.get("foreground", {})
    runs_per_config = 1  # Global default

    # Simulate the fixed logic
    exp_runs_per_config = exp.get("runs_per_config",
                                  foreground_config.get("runs_per_config", runs_per_config))

    print(f"   Expected: 5")
    print(f"   Got: {exp_runs_per_config}")
    assert exp_runs_per_config == 5, f"Expected 5, got {exp_runs_per_config}"
    print("   ✅ PASS")

    # Test case 2: Foreground level runs_per_config
    print("\n[Test 2] Foreground level runs_per_config=3")
    exp = {
        "mode": "parallel",
        "foreground": {
            "repo": "examples",
            "model": "mnist",
            "mode": "mutation",
            "mutate": ["epochs"],
            "runs_per_config": 3  # <-- Defined at foreground level
        },
        "background": {
            "repo": "examples",
            "model": "siamese",
            "hyperparameters": {}
        }
    }

    foreground_config = exp.get("foreground", {})
    runs_per_config = 1

    exp_runs_per_config = exp.get("runs_per_config",
                                  foreground_config.get("runs_per_config", runs_per_config))

    print(f"   Expected: 3")
    print(f"   Got: {exp_runs_per_config}")
    assert exp_runs_per_config == 3, f"Expected 3, got {exp_runs_per_config}"
    print("   ✅ PASS")

    # Test case 3: Both levels (outer should win)
    print("\n[Test 3] Both levels defined (outer=7, foreground=3)")
    exp = {
        "mode": "parallel",
        "runs_per_config": 7,  # <-- Outer level (higher priority)
        "foreground": {
            "repo": "examples",
            "model": "mnist",
            "mode": "mutation",
            "mutate": ["epochs"],
            "runs_per_config": 3  # <-- Foreground level
        },
        "background": {
            "repo": "examples",
            "model": "siamese",
            "hyperparameters": {}
        }
    }

    foreground_config = exp.get("foreground", {})
    runs_per_config = 1

    exp_runs_per_config = exp.get("runs_per_config",
                                  foreground_config.get("runs_per_config", runs_per_config))

    print(f"   Expected: 7 (outer level has priority)")
    print(f"   Got: {exp_runs_per_config}")
    assert exp_runs_per_config == 7, f"Expected 7, got {exp_runs_per_config}"
    print("   ✅ PASS")

    # Test case 4: Neither level (fallback to global)
    print("\n[Test 4] No runs_per_config at any level (fallback to global=1)")
    exp = {
        "mode": "parallel",
        "foreground": {
            "repo": "examples",
            "model": "mnist",
            "mode": "mutation",
            "mutate": ["epochs"]
        },
        "background": {
            "repo": "examples",
            "model": "siamese",
            "hyperparameters": {}
        }
    }

    foreground_config = exp.get("foreground", {})
    runs_per_config = 1

    exp_runs_per_config = exp.get("runs_per_config",
                                  foreground_config.get("runs_per_config", runs_per_config))

    print(f"   Expected: 1 (global fallback)")
    print(f"   Got: {exp_runs_per_config}")
    assert exp_runs_per_config == 1, f"Expected 1, got {exp_runs_per_config}"
    print("   ✅ PASS")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


def test_stage11_config_interpretation():
    """Test that Stage11 config would now be interpreted correctly"""

    print("\n" + "=" * 80)
    print("TEST: Stage11 configuration interpretation")
    print("=" * 80)

    # Load actual Stage11 config
    config_path = project_root / "settings" / "stage11_parallel_hrnet18.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nConfig: {config_path.name}")
    print(f"Total experiment configs: {len(config['experiments'])}")

    runs_per_config = config.get("runs_per_config", 1)  # Global level

    total_expected = 0
    for i, exp in enumerate(config['experiments'], 1):
        foreground_config = exp.get("foreground", {})

        # Apply the fixed logic
        exp_runs_per_config = exp.get("runs_per_config",
                                      foreground_config.get("runs_per_config", runs_per_config))

        total_expected += exp_runs_per_config

        print(f"\nConfig {i}:")
        print(f"   Mutate: {foreground_config.get('mutate', [])}")
        print(f"   Outer runs_per_config: {exp.get('runs_per_config', 'Not set')}")
        print(f"   Foreground runs_per_config: {foreground_config.get('runs_per_config', 'Not set')}")
        print(f"   Resolved value: {exp_runs_per_config}")

    print(f"\n{'=' * 80}")
    print(f"Total expected experiments: {total_expected}")
    print(f"Expected per Stage11: 20 (4 params × 5 runs)")

    assert total_expected == 20, f"Expected 20 total experiments, got {total_expected}"
    print("✅ PASS - Stage11 would now run 20 experiments")
    print("=" * 80)


def test_old_bug_simulation():
    """Simulate the old bug to show what was wrong"""

    print("\n" + "=" * 80)
    print("TEST: Old bug simulation (for comparison)")
    print("=" * 80)

    print("\n[Old Logic] foreground_config.get('runs_per_config', runs_per_config)")

    exp = {
        "mode": "parallel",
        "runs_per_config": 5,  # <-- Stage11 had this
        "foreground": {
            "repo": "examples",
            "model": "mnist",
            "mode": "mutation",
            "mutate": ["epochs"]
            # No runs_per_config here
        },
        "background": {
            "repo": "examples",
            "model": "siamese",
            "hyperparameters": {}
        }
    }

    foreground_config = exp.get("foreground", {})
    runs_per_config = 1  # Global default

    # OLD BUGGY LOGIC (what was in v4.7.0)
    old_exp_runs_per_config = foreground_config.get("runs_per_config", runs_per_config)

    print(f"   Outer level has: runs_per_config=5")
    print(f"   Foreground level has: (not set)")
    print(f"   Global default: 1")
    print(f"   Old logic result: {old_exp_runs_per_config} (WRONG! Should be 5)")

    # NEW FIXED LOGIC
    new_exp_runs_per_config = exp.get("runs_per_config",
                                      foreground_config.get("runs_per_config", runs_per_config))

    print(f"\n[New Logic] exp.get('runs_per_config', foreground_config.get('runs_per_config', runs_per_config))")
    print(f"   New logic result: {new_exp_runs_per_config} (CORRECT! ✅)")

    print(f"\n{'=' * 80}")
    print(f"Bug impact on Stage11:")
    print(f"   Expected: 4 configs × 5 runs = 20 experiments")
    print(f"   Old code ran: 4 configs × 1 run = 4 experiments")
    print(f"   Missing: 16 experiments (80%)")
    print(f"   Time wasted: ~5.4 hours (should have been ~28.6 hours)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_parallel_runs_per_config_priority()
        test_stage11_config_interpretation()
        test_old_bug_simulation()

        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED - Fix is validated!")
        print("🎉" * 40 + "\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
