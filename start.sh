#!/usr/bin/env bash
# ==============================================================================
# VisionX Studio — Launch Script
# Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ANSI color codes
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[36m"
C_GREEN="\033[32m"
C_YELLOW="\033[33m"
C_RED="\033[31m"

log_info()    { echo -e "${C_CYAN}[INFO]${C_RESET} $*"; }
log_success() { echo -e "${C_GREEN}[SUCCESS]${C_RESET} $*"; }
log_warn()    { echo -e "${C_YELLOW}[WARN]${C_RESET} $*"; }
log_error()   { echo -e "${C_RED}[ERROR]${C_RESET} $*" >&2; }

echo -e "${C_BOLD}${C_CYAN}"
cat << "EOF"
  _    _ _     _             __  __ 
 | |  | (_)   (_)           \ \/ / 
 | |  | |_ ___ _  ___  _ __  \  /  
 | |  | | / __| |/ _ \| '_ \ /  \  
  \ \/ /| \__ \ | (_) | | | / /\ \ 
   \__/ |_|___/_|\___/|_| |/_/  \_\
EOF
echo -e "${C_RESET}"
echo -e "${C_BOLD}VisionX // Parallel Video Comparison & Heatmap Studio${C_RESET}"
echo -e "--------------------------------------------------------"

# 1. Check Python Virtual Environment
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    log_warn "Virtual environment not found at ${VENV_DIR}."
    log_info "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
    log_info "Installing dependencies..."
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install flask ultralytics opencv-python-headless torch torchvision scipy matplotlib
    log_success "Virtual environment initialized."
fi

# 2. Verify Key Directories
mkdir -p "${SCRIPT_DIR}/inputs/video" "${SCRIPT_DIR}/outputs/video" "${SCRIPT_DIR}/outputs/heatmaps"

# 3. Port configuration
PORT="${PORT:-5000}"

# Check if port is already in use
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    PID=$(lsof -Pi :${PORT} -sTCP:LISTEN -t | head -n 1)
    log_warn "Port ${PORT} is already in use by process PID ${PID}."
    read -r -p "Terminate process ${PID} and restart server? [y/N] " response
    if [[ "${response}" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        kill -9 "${PID}" 2>/dev/null || true
        sleep 1
        log_info "Process ${PID} terminated."
    else
        log_error "Please set a different port using: PORT=5001 ./start.sh"
        exit 1
    fi
fi

# 4. Check CUDA Acceleration
CUDA_STATUS=$("${PYTHON_BIN}" -c "import torch; print('CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU only')" 2>/dev/null || echo "CPU")
log_info "Compute Device: ${C_BOLD}${CUDA_STATUS}${C_RESET}"

# 5. Start Server
log_info "Starting VisionX Web Studio on ${C_BOLD}http://localhost:${PORT}${C_RESET}..."
echo -e "--------------------------------------------------------"
echo -e "Press ${C_BOLD}Ctrl+C${C_RESET} to stop the server."
echo ""

# Attempt to open browser if running in a graphical session
if [[ -n "${DISPLAY:-}" ]]; then
    (sleep 1.5 && (xdg-open "http://localhost:${PORT}" 2>/dev/null || sensible-browser "http://localhost:${PORT}" 2>/dev/null || true)) &
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py"
