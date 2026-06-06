#!/bin/bash
# start-services.sh — launch model_haven services via tmux
set -euo pipefail

REPO_ROOT="$(cd "$(git rev-parse --show-toplevel)" && echo "$PWD")"
SERVICES_DIR="$REPO_ROOT/services"
SESSION_NAME="model-haven"

# Discover services: any directory under services/ that has main.py
discover_services() {
    find "$SERVICES_DIR" -path '*/.venv' -prune -o -name main.py -print | \
        xargs -I{} dirname {} | sort -u
}

# Walk up to 2 levels looking for a .venv directory
find_venv() {
    local dir="$1"
    for _ in 0 1 2; do
        [[ -d "$dir/.venv" ]] && echo "$dir/.venv" && return 0
        dir="$(dirname "$dir")"
    done
    return 1
}

# Build associative arrays: name → dir
declare -A SVC_DIR
declare -A SVC_PORT
while IFS= read -r svc_path; do
    name=$(basename "$svc_path")
    SVC_DIR["$name"]="$svc_path"
    SVC_PORT["$name"]="$(awk '/^DEFAULT_PORT[[:space:]]*=/ {print $3; exit}' "$svc_path/main.py")"
done < <(discover_services)

if [[ ${#SVC_DIR[@]} -eq 0 ]]; then
    echo "No services found in $SERVICES_DIR"
    echo "Each service directory needs a main.py file."
    exit 1
fi

# Get sorted service names
mapfile -t ALL_SERVICES < <(printf '%s\n' "${!SVC_DIR[@]}" | sort)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Start model_haven services in a tmux session."
    echo ""
    echo "Available services (auto-discovered):"
    local svc
    for svc in "${ALL_SERVICES[@]}"; do
        echo "  --$svc:${SVC_PORT[$svc]}"
    done
    echo ""
    echo "Options:"
    echo "  --all                    Start all services on their default ports"
    echo "  --{service_name}:{port}  Start a specific service on the given port"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Parse arguments
REQUESTED=()
declare -A REQUESTED_PORT

add_requested() {
    local name="$1"
    local port="$2"

    if [[ ! -v "SVC_DIR[$name]" ]]; then
        echo "Unknown service: $name"
        echo "Available services: ${ALL_SERVICES[*]}"
        exit 1
    fi

    if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        echo "Invalid port for $name: $port"
        exit 1
    fi

    REQUESTED+=("$name")
    REQUESTED_PORT["$name"]="$port"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            for name in "${ALL_SERVICES[@]}"; do
                add_requested "$name" "${SVC_PORT[$name]}"
            done
            shift
            ;;
        -h|--help)
            usage
            ;;
        --*)
            spec="${1#--}"
            if [[ "$spec" != *:* ]]; then
                echo "Service options must include a port: --{service_name}:{port}"
                echo "Example: $0 --sam3:8014 --trellis:8010"
                exit 1
            fi
            name="${spec%%:*}"
            port="${spec#*:}"
            add_requested "$name" "$port"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    usage
fi

# Check tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "  Attach: tmux attach -t $SESSION_NAME"
    echo "  Kill:   tmux kill-session -t $SESSION_NAME"
    exit 1
fi

echo "Starting model_haven services"
echo ""

first=true
for name in "${REQUESTED[@]}"; do
    svc_dir="${SVC_DIR[$name]}"
    port="${REQUESTED_PORT[$name]}"
    if ! venv_path=$(find_venv "$svc_dir"); then
        echo "WARNING: $name — no .venv found (checked up to 2 parent dirs) — run setup.bash first"
        continue
    fi

    echo "  starting $name on port $port ..."

    if $first; then
        tmux new-session -d -s "$SESSION_NAME" -n "$name" -c "$svc_dir" \
            "uv run -- python main.py --port $port"
        first=false
    else
        tmux new-window -t "$SESSION_NAME" -n "$name" -c "$svc_dir" \
            "uv run -- python main.py --port $port"
    fi
done

if $first; then
    echo "No services started"
    exit 1
fi

echo ""
echo "Services running in tmux session '$SESSION_NAME'"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Windows: tmux list-windows -t $SESSION_NAME"
echo "  Kill:    tmux kill-session -t $SESSION_NAME"
