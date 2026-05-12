#!/bin/bash
#
# Performance profiling script for EigenInference
#
# This script:
# 1. Starts Go CPU profiling on the coordinator
# 2. Runs load-test.py to generate traffic
# 3. Generates a flamegraph from the profile
# 4. Optionally profiles the provider (requires SSH access)
# 5. Generates an HTML report with all results
#
# Usage: ./profile.sh [--provider-user USER] [--provider-host HOST]
#

set -e

# Configuration
COORDINATOR_HOST="${COORDINATOR_HOST:-localhost}"
COORDINATOR_PPROF_PORT="${COORDINATOR_PPROF_PORT:-6060}"
COORDINATOR_API_PORT="${COORDINATOR_API_PORT:-8080}"
PROFILING_DURATION="${PROFILING_DURATION:-30}"
PROVIDER_USER="${PROVIDER_USER:-coordinator}"
PROVIDER_HOST="${PROVIDER_HOST:-192.168.1.125}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/tmp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider-user)
            PROVIDER_USER="$2"
            shift 2
            ;;
        --provider-host)
            PROVIDER_HOST="$2"
            shift 2
            ;;
        --duration)
            PROFILING_DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed. Required for pprof."
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed."
        exit 1
    fi
    
    # Check if load-test.py exists
    if [[ ! -f "$SCRIPT_DIR/load-test.py" ]]; then
        log_error "load-test.py not found at $SCRIPT_DIR/load-test.py"
        exit 1
    fi
    
    # Check for aiohttp
    if ! python3 -c "import aiohttp" 2>/dev/null; then
        log_warn "aiohttp not installed. Installing..."
        pip3 install aiohttp
    fi
    
    log_success "Prerequisites OK"
}

# Collect Go CPU profile from coordinator
collect_coordinator_profile() {
    log_info "Starting Go CPU profiling on coordinator (duration: ${PROFILING_DURATION}s)..."
    
    local profile_url="http://${COORDINATOR_HOST}:${COORDINATOR_PPROF_PORT}/debug/pprof/profile?seconds=${PROFILING_DURATION}"
    local profile_file="${OUTPUT_DIR}/coordinator_${TIMESTAMP}.pprof"
    
    # Start profiling in background - this will block for PROFILING_DURATION seconds
    log_info "Starting profile collection from $profile_url"
    log_info "Load test will run during profiling window..."
    
    # We'll run the profile collection and load test in parallel
    (
        curl -s -o "$profile_file" "$profile_url" && \
        log_success "Coordinator profile saved to $profile_file"
    ) &
    PROFILE_PID=$!
    
    # Give pprof a moment to start
    sleep 2
    
    # Run load test while profiling is active
    log_info "Running load test..."
    local load_test_output="${OUTPUT_DIR}/load-test_${TIMESTAMP}.json"
    
    python3 "$SCRIPT_DIR/load-test.py" \
        --url "http://${COORDINATOR_HOST}:${COORDINATOR_API_PORT}" \
        --num-requests 50 \
        --concurrency 10 \
        --output "$load_test_output"
    
    log_success "Load test complete, results: $load_test_output"
    
    # Wait for profile collection to finish
    log_info "Waiting for profile collection to complete..."
    wait $PROFILE_PID
    
    COORDINATOR_PROFILE="$profile_file"
    LOAD_TEST_RESULTS="$load_test_output"
}

# Generate flamegraph from Go profile
generate_flamegraph() {
    log_info "Generating flamegraph from profile..."
    
    local flamegraph_svg="${OUTPUT_DIR}/coordinator-flamegraph_${TIMESTAMP}.svg"
    
    if [[ ! -f "$COORDINATOR_PROFILE" ]]; then
        log_error "Profile file not found: $COORDINATOR_PROFILE"
        return 1
    fi
    
    go tool pprof -svg "$COORDINATOR_PROFILE" > "$flamegraph_svg" 2>/dev/null
    
    # Fix SVG for embedding (remove XML declaration if present)
    sed -i '' 's/<?xml[^>]*?>//' "$flamegraph_svg" 2>/dev/null || \
    sed -i 's/<?xml[^>]*?>//' "$flamegraph_svg"
    
    FLAMEGRAPH_SVG="$flamegraph_svg"
    log_success "Flamegraph saved to $flamegraph_svg"
}

# Collect provider profile via SSH
collect_provider_profile() {
    log_info "Attempting to collect provider profile..."
    
    local provider_profile="${OUTPUT_DIR}/provider_${TIMESTAMP}.txt"
    
    # Try SSH connection first
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${PROVIDER_USER}@${PROVIDER_HOST}" "echo connected" &>/dev/null; then
        log_warn "Cannot SSH to ${PROVIDER_USER}@${PROVIDER_HOST}. Skipping provider profiling."
        log_warn "Set up SSH keys or check connectivity for provider profiling."
        PROVIDER_PROFILE=""
        return 0
    fi
    
    log_info "Connected to provider at ${PROVIDER_HOST}"
    
    # Try samply first (Linux), fall back to sample (macOS)
    if ssh "${PROVIDER_USER}@${PROVIDER_HOST}" "which samply" &>/dev/null; then
        log_info "Using samply for provider profiling..."
        ssh "${PROVIDER_USER}@${PROVIDER_HOST}" \
            "sudo samply record -o /tmp/provider_profile.json -d ${PROFILING_DURATION} -- eigeninference-provider" \
            > "$provider_profile" 2>&1 || true
        
        # Copy profile back
        scp "${PROVIDER_USER}@${PROVIDER_HOST}:/tmp/provider_profile.json" \
            "${OUTPUT_DIR}/provider-profile_${TIMESTAMP}.json" 2>/dev/null || true
        PROVIDER_PROFILE="${OUTPUT_DIR}/provider-profile_${TIMESTAMP}.json"
        
    elif ssh "${PROVIDER_USER}@${PROVIDER_HOST}" "which sample" &>/dev/null; then
        log_info "Using macOS 'sample' command for provider profiling..."
        
        # Find provider process
        local provider_pid=$(ssh "${PROVIDER_USER}@${PROVIDER_HOST}" \
            "pgrep -f eigeninference-provider | head -1")
        
        if [[ -n "$provider_pid" ]]; then
            log_info "Found provider PID: $provider_pid"
            ssh "${PROVIDER_USER}@${PROVIDER_HOST}" \
                "sample $provider_pid ${PROFILING_DURATION} -f /tmp/provider_sample.txt" \
                > /dev/null 2>&1 || true
            
            # Copy sample output
            scp "${PROVIDER_USER}@${PROVIDER_HOST}:/tmp/provider_sample.txt" \
                "$provider_profile" 2>/dev/null || true
            PROVIDER_PROFILE="$provider_profile"
        else
            log_warn "Provider process not found on remote host"
            PROVIDER_PROFILE=""
        fi
    else
        log_warn "No suitable profiler (samply/sample) found on provider"
        PROVIDER_PROFILE=""
    fi
    
    if [[ -n "$PROVIDER_PROFILE" && -f "$PROVIDER_PROFILE" ]]; then
        log_success "Provider profile saved to $PROVIDER_PROFILE"
    fi
}

# Extract per-stage timing from coordinator logs
extract_coordinator_logs() {
    log_info "Extracting coordinator logs..."
    
    local logs_file="${OUTPUT_DIR}/coordinator-logs_${TIMESTAMP}.txt"
    
    # Try to get logs from the coordinator process
    # This assumes coordinator is running via systemd or as a local process
    if command -v journalctl &>/dev/null; then
        journalctl -u coordinator --since "5 minutes ago" > "$logs_file" 2>/dev/null || true
    fi
    
    # Also try to get logs from a local file if present
    if [[ -f "/var/log/eigeninference/coordinator.log" ]]; then
        tail -1000 /var/log/eigeninference/coordinator.log >> "$logs_file" 2>/dev/null || true
    fi
    
    # Try Docker logs if running in container
    if command -v docker &>/dev/null; then
        local container_id=$(docker ps -q --filter "name=coordinator" 2>/dev/null | head -1)
        if [[ -n "$container_id" ]]; then
            docker logs --tail 1000 "$container_id" >> "$logs_file" 2>/dev/null || true
        fi
    fi
    
    if [[ -f "$logs_file" && -s "$logs_file" ]]; then
        COORDINATOR_LOGS="$logs_file"
        log_success "Coordinator logs saved to $logs_file"
    else
        log_warn "Could not collect coordinator logs"
        COORDINATOR_LOGS=""
    fi
}

# Parse load test JSON for stats
parse_load_test_stats() {
    if [[ ! -f "$LOAD_TEST_RESULTS" ]]; then
        log_warn "No load test results to parse"
        return
    fi
    
    # Extract stats using python for reliable JSON parsing
    python3 << EOF
import json

with open("$LOAD_TEST_RESULTS") as f:
    data = json.load(f)

stats = data.get("summary", {})
print("LOAD_TEST_STATS<<EOF")
for key, value in stats.items():
    print(f"{key}={value}")
print("EOF")
EOF
}

# Generate HTML report
generate_html_report() {
    log_info "Generating HTML performance report..."
    
    local report_file="${OUTPUT_DIR}/eigeninference-perf-report_${TIMESTAMP}.html"
    
    # Read flamegraph SVG
    local flamegraph_content=""
    if [[ -f "$FLAMEGRAPH_SVG" ]]; then
        flamegraph_content=$(cat "$FLAMEGRAPH_SVG" | sed 's/</\\u003c/g' | sed 's/>/\\u003e/g' | sed 's/&/\\u0026/g')
    fi
    
    # Read load test stats
    local load_test_stats_json="{}"
    if [[ -f "$LOAD_TEST_RESULTS" ]]; then
        # Extract just the summary
        load_test_stats_json=$(python3 -c "
import json
with open('$LOAD_TEST_RESULTS') as f:
    data = json.load(f)
print(json.dumps(data.get('summary', {})))
")
    fi
    
    # Read coordinator logs
    local coordinator_logs=""
    if [[ -f "$COORDINATOR_LOGS" ]]; then
        coordinator_logs=$(tail -200 "$COORDINATOR_LOGS" | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g')
    fi
    
    # Read provider profile
    local provider_content=""
    if [[ -n "$PROVIDER_PROFILE" && -f "$PROVIDER_PROFILE" ]]; then
        provider_content=$(head -100 "$PROVIDER_PROFILE" | sed 's/</\&lt;/g' | sed 's/>/\&gt;/g')
    fi
    
    # Generate HTML with embedded data
    cat > "$report_file" << HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EigenInference Performance Report - $TIMESTAMP</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f3460;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4ecca3;
            --warning: #ffc107;
            --error: #ff6b6b;
            --border: #2d3748;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5rem;
            color: var(--accent);
            margin-bottom: 10px;
        }
        
        .timestamp {
            color: var(--text-secondary);
            font-size: 1rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        
        .card h2 {
            color: var(--accent);
            font-size: 1.3rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        
        .stat-item {
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 8px;
        }
        
        .stat-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .stat-value.success { color: var(--success); }
        .stat-value.warning { color: var(--warning); }
        .stat-value.error { color: var(--error); }
        
        .flamegraph-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
            margin-bottom: 30px;
            overflow: hidden;
        }
        
        .flamegraph-section h2 {
            color: var(--accent);
            margin-bottom: 16px;
        }
        
        #flamegraph-container {
            background: white;
            border-radius: 8px;
            overflow: auto;
            max-height: 600px;
        }
        
        #flamegraph-container svg {
            width: 100%;
            height: auto;
            min-width: 1000px;
        }
        
        .logs-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        .logs-section h2 {
            color: var(--accent);
            margin-bottom: 16px;
        }
        
        .log-viewer {
            background: #0d1117;
            border-radius: 8px;
            padding: 16px;
            font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
            overflow: auto;
            max-height: 400px;
            white-space: pre-wrap;
            color: #c9d1d9;
        }
        
        .latency-bar {
            height: 24px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .latency-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success), var(--warning), var(--error));
            border-radius: 4px;
        }
        
        .percentile-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 12px;
        }
        
        .percentile-item {
            text-align: center;
            padding: 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
        }
        
        .percentile-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }
        
        .percentile-value {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 4px;
        }
        
        footer {
            text-align: center;
            padding: 30px 0;
            border-top: 1px solid var(--border);
            margin-top: 30px;
            color: var(--text-secondary);
        }
        
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
            
            .stat-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔥 EigenInference Performance Report</h1>
            <p class="timestamp">Generated: $(date -r "$TIMESTAMP" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')</p>
        </header>
        
        <div class="grid">
            <div class="card">
                <h2>📊 Request Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Requests</div>
                        <div class="stat-value" id="total-requests">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Successful</div>
                        <div class="stat-value success" id="successful">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Failed</div>
                        <div class="stat-value error" id="failed">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Throughput</div>
                        <div class="stat-value" id="throughput">- req/s</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>⏱️ Latency (Total)</h2>
                <div class="percentile-grid">
                    <div class="percentile-item">
                        <div class="percentile-label">P50</div>
                        <div class="percentile-value" id="latency-p50">-</div>
                    </div>
                    <div class="percentile-item">
                        <div class="percentile-label">P95</div>
                        <div class="percentile-value" id="latency-p95">-</div>
                    </div>
                    <div class="percentile-item">
                        <div class="percentile-label">P99</div>
                        <div class="percentile-value" id="latency-p99">-</div>
                    </div>
                </div>
                <div class="latency-bar">
                    <div class="latency-fill" id="latency-bar-fill" style="width: 60%"></div>
                </div>
            </div>
            
            <div class="card">
                <h2>🚀 Time to First Byte</h2>
                <div class="percentile-grid">
                    <div class="percentile-item">
                        <div class="percentile-label">P50</div>
                        <div class="percentile-value" id="ttfb-p50">-</div>
                    </div>
                    <div class="percentile-item">
                        <div class="percentile-label">P95</div>
                        <div class="percentile-value" id="ttfb-p95">-</div>
                    </div>
                    <div class="percentile-item">
                        <div class="percentile-label">P99</div>
                        <div class="percentile-value" id="ttfb-p99">-</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 Token Performance</h2>
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Total Tokens</div>
                        <div class="stat-value" id="total-tokens">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Avg Tokens/sec</div>
                        <div class="stat-value success" id="avg-tok-sec">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Avg Latency</div>
                        <div class="stat-value" id="avg-latency">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Total Time</div>
                        <div class="stat-value" id="total-time">-</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="flamegraph-section">
            <h2>🔥 Coordinator Flamegraph</h2>
            <div id="flamegraph-container">
                <p style="padding: 20px; text-align: center; color: #666;">
                    Flamegraph will load here...
                </p>
            </div>
        </div>
        
        <div class="logs-section">
            <h2>📋 Coordinator Logs (Last 200 lines)</h2>
            <div class="log-viewer" id="coordinator-logs">
                Logs not available
            </div>
        </div>
        
        $(if [[ -n "$provider_content" ]]; then
        echo '<div class="logs-section">'
        echo '    <h2>🖥️ Provider Profile</h2>'
        echo '    <div class="log-viewer" id="provider-profile">'
        echo "$provider_content"
        echo '    </div>'
        echo '</div>'
        fi)
        
        <footer>
            <p>EigenInference Performance Analysis | Report generated automatically</p>
        </footer>
    </div>
    
    <script>
        // Load test statistics
        const loadTestStats = $load_test_stats_json;
        
        // Update UI with stats
        function updateStats() {
            if (loadTestStats.total_requests !== undefined) {
                document.getElementById('total-requests').textContent = loadTestStats.total_requests;
            }
            if (loadTestStats.successful_requests !== undefined) {
                document.getElementById('successful').textContent = loadTestStats.successful_requests;
            }
            if (loadTestStats.failed_requests !== undefined) {
                document.getElementById('failed').textContent = loadTestStats.failed_requests;
            }
            if (loadTestStats.throughput_rps !== undefined) {
                document.getElementById('throughput').textContent = loadTestStats.throughput_rps + ' req/s';
            }
            if (loadTestStats.latency_p50_ms !== undefined) {
                document.getElementById('latency-p50').textContent = loadTestStats.latency_p50_ms + ' ms';
            }
            if (loadTestStats.latency_p95_ms !== undefined) {
                document.getElementById('latency-p95').textContent = loadTestStats.latency_p95_ms + ' ms';
            }
            if (loadTestStats.latency_p99_ms !== undefined) {
                document.getElementById('latency-p99').textContent = loadTestStats.latency_p99_ms + ' ms';
            }
            if (loadTestStats.ttfb_p50_ms !== undefined) {
                document.getElementById('ttfb-p50').textContent = loadTestStats.ttfb_p50_ms + ' ms';
            }
            if (loadTestStats.ttfb_p95_ms !== undefined) {
                document.getElementById('ttfb-p95').textContent = loadTestStats.ttfb_p95_ms + ' ms';
            }
            if (loadTestStats.ttfb_p99_ms !== undefined) {
                document.getElementById('ttfb-p99').textContent = loadTestStats.ttfb_p99_ms + ' ms';
            }
            if (loadTestStats.total_tokens !== undefined) {
                document.getElementById('total-tokens').textContent = loadTestStats.total_tokens.toLocaleString();
            }
            if (loadTestStats.avg_tok_per_sec !== undefined) {
                document.getElementById('avg-tok-sec').textContent = loadTestStats.avg_tok_per_sec;
            }
            if (loadTestStats.avg_total_ms !== undefined) {
                document.getElementById('avg-latency').textContent = loadTestStats.avg_total_ms + ' ms';
            }
            if (loadTestStats.total_time_sec !== undefined) {
                document.getElementById('total-time').textContent = loadTestStats.total_time_sec + ' s';
            }
        }
        
        // Load flamegraph
        function loadFlamegraph() {
            const container = document.getElementById('flamegraph-container');
            // Flamegraph SVG is embedded inline via PHP/template
            const svgContent = $(if [[ -f "$FLAMEGRAPH_SVG" ]]; then cat "$FLAMEGRAPH_SVG"; else echo '""'; fi);
            if (svgContent) {
                container.innerHTML = svgContent;
            }
        }
        
        // Initialize
        updateStats();
        // loadFlamegraph(); // Loaded inline
    </script>
</body>
</html>
HTMLEOF

    # Embed flamegraph SVG directly
    if [[ -f "$FLAMEGRAPH_SVG" ]]; then
        # Use python to properly embed the SVG
        python3 << PYEOF
import re

with open("$report_file", "r") as f:
    html = f.read()

with open("$FLAMEGRAPH_SVG", "r") as f:
    svg = f.read()

# Replace placeholder with actual SVG
html = html.replace(
    '<p style="padding: 20px; text-align: center; color: #666;">\n                    Flamegraph will load here...\n                </p>',
    svg
)

with open("$report_file", "w") as f:
    f.write(html)

print("Flamegraph embedded successfully")
PYEOF
    fi
    
    # Embed coordinator logs
    if [[ -f "$COORDINATOR_LOGS" ]]; then
        python3 << PYEOF
with open("$report_file", "r") as f:
    html = f.read()

with open("$COORDINATOR_LOGS", "r") as f:
    logs = f.read()

# Escape for HTML
import html as html_module
logs = html_module.escape(logs)

# Replace placeholder
html = html.replace(
    'Logs not available',
    logs[-50000:] if len(logs) > 50000 else logs  # Limit size
)

with open("$report_file", "w") as f:
    f.write(html)
PYEOF
    fi
    
    HTML_REPORT="$report_file"
    log_success "HTML report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting EigenInference performance profiling..."
    log_info "Timestamp: $TIMESTAMP"
    
    check_prerequisites
    
    # Step 1: Collect coordinator profile + run load test
    collect_coordinator_profile
    
    # Step 2: Generate flamegraph
    generate_flamegraph
    
    # Step 3: Collect provider profile (optional)
    collect_provider_profile
    
    # Step 4: Extract coordinator logs
    extract_coordinator_logs
    
    # Step 5: Generate HTML report
    generate_html_report
    
    # Summary
    echo ""
    log_success "=========================================="
    log_success "Profiling complete!"
    log_success "=========================================="
    echo ""
    echo "Generated files:"
    [[ -n "$COORDINATOR_PROFILE" && -f "$COORDINATOR_PROFILE" ]] && echo "  📊 Coordinator profile: $COORDINATOR_PROFILE"
    [[ -n "$FLAMEGRAPH_SVG" && -f "$FLAMEGRAPH_SVG" ]] && echo "  🔥 Flamegraph: $FLAMEGRAPH_SVG"
    [[ -n "$LOAD_TEST_RESULTS" && -f "$LOAD_TEST_RESULTS" ]] && echo "  📈 Load test results: $LOAD_TEST_RESULTS"
    [[ -n "$PROVIDER_PROFILE" && -f "$PROVIDER_PROFILE" ]] && echo "  🖥️ Provider profile: $PROVIDER_PROFILE"
    [[ -n "$HTML_REPORT" && -f "$HTML_REPORT" ]] && echo "  📄 HTML report: $HTML_REPORT"
    echo ""
    
    # Open report if possible
    if [[ -f "$HTML_REPORT" ]]; then
        if command -v open &>/dev/null; then
            log_info "Opening report in browser..."
            open "$HTML_REPORT"
        elif command -v xdg-open &>/dev/null; then
            log_info "Opening report in browser..."
            xdg-open "$HTML_REPORT"
        fi
    fi
}

main "$@"
