#!/usr/bin/env bash
# install_launchd.sh — bootstrap the three macminibot launchd agents.
#
# What this script does:
#   1. Validates the three plist files exist and parse as XML (plutil).
#   2. PRINTS — but does not run — the exact commands needed to install.
#   3. Lists them with `launchctl list | grep macminibot` after install.
#
# By design this script DOES NOT run launchctl bootstrap itself. Scheduling
# the paper bot is a consequential action; the human runs the commands.

set -euo pipefail

PROJECT_ROOT="/Users/weeoto/macminibot"
PLIST_DIR="${PROJECT_ROOT}/scripts/launchd"

PLISTS=(
    "com.macminibot.paper-runner.plist"
    "com.macminibot.heartbeat.plist"
    "com.macminibot.dashboard.plist"
)

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}macminibot launchd installer (dry-run / instructions only)${NC}"
echo

# ── 1. Validate every plist parses ──────────────────────────────────────────
echo -e "${BOLD}Validating plists...${NC}"
for plist in "${PLISTS[@]}"; do
    path="${PLIST_DIR}/${plist}"
    if [[ ! -f "$path" ]]; then
        echo -e "${RED}MISSING:${NC} $path"
        exit 1
    fi
    if ! plutil -lint "$path" >/dev/null; then
        echo -e "${RED}INVALID XML:${NC} $path"
        plutil -lint "$path"
        exit 1
    fi
    echo -e "  ${GREEN}OK${NC}  $plist"
done
echo

# ── 2. Print the install commands ───────────────────────────────────────────
echo -e "${BOLD}To enable scheduling, run the following commands as your user:${NC}"
echo
for plist in "${PLISTS[@]}"; do
    echo -e "  ${YELLOW}launchctl bootstrap gui/\$UID ${PLIST_DIR}/${plist}${NC}"
done
echo
echo -e "${BOLD}Then verify they loaded:${NC}"
echo -e "  ${YELLOW}launchctl list | grep macminibot${NC}"
echo
echo -e "${BOLD}And print next-fire times:${NC}"
for plist in "${PLISTS[@]}"; do
    label="${plist%.plist}"
    echo -e "  ${YELLOW}launchctl print gui/\$UID/${label} | grep -E 'next fire|state'${NC}"
done
echo

# ── 3. If already loaded, show current state ────────────────────────────────
echo -e "${BOLD}Current state of macminibot launchd jobs:${NC}"
if launchctl list 2>/dev/null | grep macminibot; then
    echo
    echo -e "${BOLD}Next fire times (where available):${NC}"
    for plist in "${PLISTS[@]}"; do
        label="${plist%.plist}"
        if launchctl print "gui/${UID}/${label}" >/dev/null 2>&1; then
            echo -e "  ${BOLD}${label}:${NC}"
            launchctl print "gui/${UID}/${label}" | grep -E 'next fire|state =' | sed 's/^/    /'
        fi
    done
else
    echo "  (none loaded yet)"
fi
echo
echo -e "${GREEN}Done.${NC} No system state was modified by this script."
