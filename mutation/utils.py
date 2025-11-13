"""
Utility Functions

This module provides utility functions for mutation.py, including
CPU governor control and logger setup.

Constants:
    ALLOWED_GOVERNORS: Set of valid CPU governor modes
    GOVERNOR_TIMEOUT_SECONDS: Timeout for governor commands
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional


# Constants
ALLOWED_GOVERNORS = {"performance", "powersave", "ondemand", "conservative", "schedutil"}
GOVERNOR_TIMEOUT_SECONDS = 10


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and configure a logger instance

    Args:
        name: Logger name (default: module name)
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def set_governor(
    mode: str,
    project_root: Path,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Set CPU governor mode using governor.sh

    Args:
        mode: Governor mode (must be one of: performance, powersave, ondemand, conservative, schedutil)
        project_root: Path to project root directory (kept for backward compatibility, not used)
        logger: Logger instance for debug messages

    Returns:
        True if successful, False otherwise

    Security:
        - Governor modes are strictly whitelisted to prevent shell injection
        - Requires sudo permissions (checks before execution)

    Note:
        governor.sh is now located in mutation/ package directory
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Validate mode (SECURITY: prevent shell injection)
    if mode not in ALLOWED_GOVERNORS:
        logger.error(f"Invalid governor mode '{mode}'")
        print(f"❌ Error: Invalid governor mode '{mode}'")
        print(f"   Allowed modes: {', '.join(sorted(ALLOWED_GOVERNORS))}")
        return False

    governor_script = Path(__file__).parent / "governor.sh"

    if not governor_script.exists():
        logger.warning(f"Governor script not found: {governor_script}")
        print(f"⚠️  WARNING: Governor script not found: {governor_script}")
        return False

    # Check sudo permissions first
    try:
        sudo_check = subprocess.run(
            ["sudo", "-n", "true"],  # Non-interactive sudo check
            capture_output=True,
            timeout=2
        )
        if sudo_check.returncode != 0:
            logger.warning("sudo permissions required for governor control")
            print(f"⚠️  WARNING: sudo permissions required for governor control")
            print(f"   Run: sudo -v  (or configure passwordless sudo for governor.sh)")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("sudo permission check timed out")
        print(f"⚠️  WARNING: sudo permission check timed out")
        return False
    except Exception as e:
        logger.warning(f"Cannot check sudo permissions: {e}")
        print(f"⚠️  WARNING: Cannot check sudo permissions: {e}")
        return False

    try:
        logger.info(f"Setting CPU governor to: {mode}")
        print(f"🔧 Setting CPU governor to: {mode}")

        result = subprocess.run(
            ["sudo", str(governor_script), mode],  # Now safe - mode is validated
            capture_output=True,
            text=True,
            timeout=GOVERNOR_TIMEOUT_SECONDS
        )

        if result.returncode == 0:
            logger.info(f"CPU governor set to: {mode}")
            print(f"✓ CPU governor set to: {mode}")
            return True
        else:
            logger.warning(f"Failed to set governor: {result.stderr}")
            print(f"⚠️  WARNING: Failed to set governor: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Governor command timed out")
        print(f"⚠️  WARNING: Governor command timed out")
        return False
    except subprocess.SubprocessError as e:
        logger.warning(f"Subprocess error setting governor: {e}")
        print(f"⚠️  WARNING: Subprocess error setting governor: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting governor: {e}", exc_info=True)
        print(f"⚠️  WARNING: Error setting governor: {e}")
        return False
