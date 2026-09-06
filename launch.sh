#!/usr/bin/env bash
# ==============================================================================
# Inmarscan: Rural India Diabetic Retinopathy Tele-Screening System
# Master Hackathon Launch & Orchestration Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    echo ""
    echo "=========================================================================="
    echo "  Inmarscan: Rural India Diabetic Retinopathy Tele-Screening"
    echo "=========================================================================="
    echo "Usage: ./launch.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  --gui         Launch Tkinter Desktop GUI (Official Edge Clinic UI)"
    echo "  --all         Start FastAPI backend in background + launch Tkinter Desktop GUI"
    echo "  --backend     Start FastAPI backend server on http://127.0.0.1:8000"
    echo "  --eval        Run external validation on IDRiD Indian cohort & patient pairing"
    echo "  --tables      Regenerate Black & White PPTX presentation tables & high-res PNGs"
    echo "  --test        Run full automated test suite (pytest)"
    echo "  --frontend    (Optional) Start legacy SvelteKit web interface on :5173"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./launch.sh --gui       # Launch official edge clinic desktop app"
    echo "  ./launch.sh --all       # Launch backend + Tkinter GUI together"
    echo "  ./launch.sh --eval      # Re-compute clinical accuracy benchmarks"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python 3 is required but not found in PATH."
        exit 1
    fi
}

check_node() {
    if ! command -v npm &> /dev/null; then
        echo "[ERROR] npm is required for the optional web frontend. Please install Node.js."
        exit 1
    fi
}

MODE="${1:---help}"

case "$MODE" in
    --backend)
        check_python
        echo "[INFO] Starting FastAPI Backend on http://127.0.0.1:8000 ..."
        python3 -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
        ;;

    --gui)
        check_python
        echo "[INFO] Launching Official Desktop Tkinter Edge GUI ..."
        python3 gui/gui.py
        ;;

    --all)
        check_python
        echo "=========================================================================="
        echo "  Launching Inmarscan System (FastAPI Backend + Tkinter GUI)"
        echo "=========================================================================="
        echo "  Backend API:  http://127.0.0.1:8000 (Swagger docs: /docs)"
        echo "  Desktop UI:   Tkinter Edge Clinic Screening GUI"
        echo "=========================================================================="
        echo "[INFO] Starting backend in background..."
        python3 -m uvicorn backend.server:app --host 127.0.0.1 --port 8000 &
        BACKEND_PID=$!

        cleanup() {
            echo ""
            echo "[INFO] Shutting down background backend server..."
            kill "$BACKEND_PID" 2>/dev/null || true
            exit 0
        }
        trap cleanup SIGINT SIGTERM EXIT

        sleep 1.5
        echo "[INFO] Launching Desktop Tkinter GUI..."
        python3 gui/gui.py
        ;;

    --frontend)
        check_node
        echo "[INFO] Starting optional SvelteKit Web UI on http://127.0.0.1:5173 ..."
        cd frontend
        npm run dev -- --host 127.0.0.1 --port 5173
        ;;

    --eval)
        check_python
        echo "[INFO] Running patient-level pairing & external IDRiD evaluation ..."
        python3 pair_eyes_patient_level.py
        ;;

    --tables)
        check_python
        echo "[INFO] Generating publication-grade Black & White presentation tables ..."
        python3 generate_pptx_tables_and_simulink.py
        ;;

    --test)
        check_python
        echo "[INFO] Executing automated unit & integration test suite ..."
        pytest tests/ -v
        ;;

    --help|-h|*)
        show_help
        ;;
esac
