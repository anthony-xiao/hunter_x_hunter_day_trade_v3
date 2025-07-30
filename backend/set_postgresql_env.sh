#!/bin/bash
# PostgreSQL UTC Environment Variables
# Add these lines to your shell profile (~/.bashrc, ~/.zshrc, etc.)

export PGTZ=UTC
export TZ=UTC

echo "PostgreSQL timezone environment variables set to UTC"
echo "PGTZ=$PGTZ"
echo "TZ=$TZ"
