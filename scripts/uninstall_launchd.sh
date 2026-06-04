#!/usr/bin/env bash
# uninstall_launchd.sh — cleanly tear down the three macminibot launchd agents.
#
# Idempotent: bootout of a non-loaded label is treated as success.
# Useful for upgrades — bootout, edit the plist, re-bootstrap.

set -uo pipefail

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

echo -e "${BOLD}macminibot launchd uninstaller${NC}"
echo

for plist in "${PLISTS[@]}"; do
    label="${plist%.plist}"
    path="${PLIST_DIR}/${plist}"
    domain_target="gui/${UID}/${label}"

    echo -e "  ${YELLOW}launchctl bootout ${domain_target}${NC}"
    if launchctl bootout "gui/${UID}" "${path}" 2>/dev/null; then
        echo -e "    ${GREEN}OK${NC} booted out ${label}"
    elif ! launchctl print "${domain_target}" >/dev/null 2>&1; then
        echo -e "    ${GREEN}OK${NC} ${label} was not loaded"
    else
        echo -e "    ${RED}FAILED${NC} to bootout ${label} (still loaded)"
    fi
done

echo
echo -e "${BOLD}Remaining macminibot launchd jobs:${NC}"
if launchctl list 2>/dev/null | grep macminibot; then
    echo -e "${RED}WARNING:${NC} some jobs are still loaded."
    exit 1
else
    echo "  (none)"
fi
echo -e "${GREEN}Clean.${NC}"
