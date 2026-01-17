#!/bin/bash
# AI Autonom - Kali Tool Runner
# Executes commands and logs output for visibility

LOG_FILE="/logs/output.log"
FINDINGS_FILE="/findings/findings.txt"

# Log command
log_command() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CMD: $*" >> "$LOG_FILE"
}

# Log output
log_output() {
    while IFS= read -r line; do
        echo "$line"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OUT: $line" >> "$LOG_FILE"
    done
}

# Save finding
save_finding() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$FINDINGS_FILE"
}

# Execute and log
run_logged() {
    log_command "$@"
    "$@" 2>&1 | log_output
    return ${PIPESTATUS[0]}
}

# Main
if [ "$1" == "--help" ]; then
    echo "Usage: run.sh <command> [args...]"
    echo "Executes command with full logging to /logs/output.log"
    exit 0
fi

run_logged "$@"
