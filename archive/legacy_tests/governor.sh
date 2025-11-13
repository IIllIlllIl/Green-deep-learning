#!/bin/bash
################################################################################
# Mock Governor Script for Testing
#
# This script simulates the governor.sh behavior without requiring root access
################################################################################

if [ "x$1" = "x" ]; then
    echo "usage: $0 on[demand]|pe[rformance]|po[wersave]|us[erspace]?"
    exit 1
fi

case $1 in
    on*|de*)    governor="ondemand";;
    po*|pw*)    governor="powersave";;
    pe*)        governor="performance";;
    co*)        governor="conservative";;
    us*)        governor="userspace";;
    *)          echo "$1: unrecognized governor"; exit 1;;
esac

echo "[Mock Governor] Setting CPU governor to: $governor"
echo "[Mock Governor] This is a mock script - no actual changes made"

exit 0
