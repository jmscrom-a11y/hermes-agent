#!/bin/bash
# Hermes Agent Restart Script
# 재시작 시 기존 프로세스 종료 후 새 인스턴스 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[restart] Stopping existing Hermes Agent..."
pkill -f "python.*bot.py" 2>/dev/null || true
pkill -f "python.*telegram" 2>/dev/null || true

sleep 1

echo "[restart] Starting Hermes Agent..."
exec python -m app.telegram.bot
