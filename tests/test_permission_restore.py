#!/usr/bin/env python3
"""
Test for automatic permission restoration when running with sudo

This test verifies that the restore_permissions() method correctly:
1. Detects when running as root (with sudo)
2. Identifies the original user from SUDO_USER
3. Restores file ownership to the original user
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mutation.session import ExperimentSession


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_ownership(path):
    """Check the ownership of a file or directory

    Returns:
        tuple: (uid, gid, username)
    """
    import pwd
    stat_info = os.stat(path)
    uid = stat_info.st_uid
    gid = stat_info.st_gid
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        username = f"uid:{uid}"
    return uid, gid, username


def test_restore_permissions_as_root():
    """Test permission restoration when running as root"""
    print_section("Test 1: restore_permissions() running AS ROOT (with sudo)")

    # Check if running as root
    if os.geteuid() != 0:
        print("❌ ERROR: This test must be run with sudo!")
        print("   Usage: sudo python3 tests/test_permission_restore.py")
        return False

    # Check if SUDO_USER is set
    sudo_user = os.environ.get('SUDO_USER')
    if not sudo_user:
        print("❌ ERROR: SUDO_USER environment variable not set!")
        print("   This test must be run via sudo (not direct root login)")
        return False

    print(f"✅ Running as root (euid={os.geteuid()})")
    print(f"✅ Original user: {sudo_user}")

    # Create a temporary results directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_restore_"))
    print(f"\n📁 Created test directory: {temp_dir}")

    try:
        # Create a session (this will create session_dir with root ownership)
        session = ExperimentSession(temp_dir)
        print(f"📁 Created session directory: {session.session_dir}")

        # Create some test files and subdirectories (simulating experiment results)
        test_exp_dir = session.session_dir / "test_experiment_001"
        test_exp_dir.mkdir(exist_ok=True)

        energy_dir = test_exp_dir / "energy"
        energy_dir.mkdir(exist_ok=True)

        # Create test files
        (test_exp_dir / "experiment.json").write_text('{"test": "data"}')
        (test_exp_dir / "training.log").write_text("Training log content")
        (energy_dir / "cpu_energy.txt").write_text("CPU energy data")
        (energy_dir / "gpu_power.csv").write_text("GPU power data")
        (session.session_dir / "summary.csv").write_text("Summary CSV")

        print(f"\n📝 Created test files:")
        print(f"   - {test_exp_dir}/experiment.json")
        print(f"   - {test_exp_dir}/training.log")
        print(f"   - {energy_dir}/cpu_energy.txt")
        print(f"   - {energy_dir}/gpu_power.csv")
        print(f"   - {session.session_dir}/summary.csv")

        # Check ownership BEFORE restoration
        print("\n🔍 File ownership BEFORE restoration:")
        for path in [session.session_dir, test_exp_dir, energy_dir]:
            uid, gid, username = check_ownership(path)
            print(f"   {path.name}: {username} (uid={uid}, gid={gid})")

        # Verify all files are owned by root
        root_uid, _, _ = check_ownership(session.session_dir)
        if root_uid != 0:
            print(f"\n❌ ERROR: Session directory should be owned by root, got uid={root_uid}")
            return False
        print("   ✅ All files owned by root (as expected)")

        # Call restore_permissions()
        print("\n🔧 Calling session.restore_permissions()...")
        session.restore_permissions()

        # Check ownership AFTER restoration
        print("\n🔍 File ownership AFTER restoration:")
        import pwd
        expected_uid = pwd.getpwnam(sudo_user).pw_uid
        expected_gid = pwd.getpwnam(sudo_user).pw_gid

        all_correct = True
        for path in [session.session_dir, test_exp_dir, energy_dir,
                     test_exp_dir / "experiment.json",
                     test_exp_dir / "training.log",
                     energy_dir / "cpu_energy.txt",
                     session.session_dir / "summary.csv"]:
            uid, gid, username = check_ownership(path)
            status = "✅" if uid == expected_uid else "❌"
            print(f"   {status} {path.name}: {username} (uid={uid}, gid={gid})")
            if uid != expected_uid:
                all_correct = False

        if all_correct:
            print(f"\n✅ SUCCESS: All files restored to user '{sudo_user}'")
            return True
        else:
            print(f"\n❌ FAILURE: Some files not restored correctly")
            return False

    finally:
        # Cleanup: restore ownership before deleting (to avoid permission issues)
        if temp_dir.exists():
            subprocess.run(['chown', '-R', f'{os.getuid()}:{os.getgid()}', str(temp_dir)],
                         capture_output=True)
            shutil.rmtree(temp_dir)
            print(f"\n🗑️  Cleaned up test directory")


def test_restore_permissions_as_normal_user():
    """Test that restore_permissions() does nothing when not running as root"""
    print_section("Test 2: restore_permissions() running as NORMAL USER")

    if os.geteuid() == 0:
        print("⏭️  Skipping: This test should run as normal user (without sudo)")
        return True

    print(f"✅ Running as normal user (euid={os.geteuid()})")

    # Create a temporary results directory
    temp_dir = Path(tempfile.mkdtemp(prefix="test_noroot_"))
    print(f"\n📁 Created test directory: {temp_dir}")

    try:
        # Create a session
        session = ExperimentSession(temp_dir)
        print(f"📁 Created session directory: {session.session_dir}")

        # Create a test file
        test_file = session.session_dir / "test.txt"
        test_file.write_text("test content")

        # Get ownership before
        uid_before, gid_before, username_before = check_ownership(test_file)
        print(f"\n🔍 File ownership before: {username_before} (uid={uid_before})")

        # Call restore_permissions() - should do nothing
        print("\n🔧 Calling session.restore_permissions()...")
        session.restore_permissions()

        # Get ownership after
        uid_after, gid_after, username_after = check_ownership(test_file)
        print(f"🔍 File ownership after: {username_after} (uid={uid_after})")

        # Verify ownership unchanged
        if uid_before == uid_after and gid_before == gid_after:
            print(f"\n✅ SUCCESS: Ownership unchanged (as expected)")
            return True
        else:
            print(f"\n❌ FAILURE: Ownership changed unexpectedly")
            return False

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🗑️  Cleaned up test directory")


def main():
    """Run all tests"""
    print("\n" + "█" * 80)
    print("█  PERMISSION RESTORATION TEST SUITE")
    print("█" * 80)

    # Determine which tests to run based on current user
    is_root = os.geteuid() == 0

    if is_root:
        print("\n🔐 Detected: Running as root (with sudo)")
        print("   Will test automatic permission restoration")
        results = []
        results.append(("Root restoration", test_restore_permissions_as_root()))

        # Summary
        print("\n" + "=" * 80)
        print("  TEST SUMMARY")
        print("=" * 80)
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name}")

        all_passed = all(result[1] for result in results)
        if all_passed:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED")
            return 1
    else:
        print("\n👤 Detected: Running as normal user (without sudo)")
        print("   Will test that restore_permissions() does nothing")
        results = []
        results.append(("Normal user no-op", test_restore_permissions_as_normal_user()))

        # Summary
        print("\n" + "=" * 80)
        print("  TEST SUMMARY")
        print("=" * 80)
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status}: {test_name}")

        print("\n" + "=" * 80)
        print("  NEXT STEP")
        print("=" * 80)
        print("To test automatic permission restoration, run:")
        print(f"  sudo python3 {__file__}")
        print("=" * 80)

        all_passed = all(result[1] for result in results)
        if all_passed:
            print("\n✅ Normal user test passed!")
            print("   (Still need to run with sudo for complete testing)")
            return 0
        else:
            print("\n❌ TEST FAILED")
            return 1


if __name__ == "__main__":
    sys.exit(main())
