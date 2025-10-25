#!/usr/bin/env python3
"""
Argus Streamlit telemetry app + Prometheus exporter.
Merged changes:
 - Landing page with logo + "Argus Agent" title to the right.
 - Landing container cleared on Start (no faded leftover).
 - Company profile selection (Mid-tier / Large-size) influences generators.
 - Generator function signatures restored to their original no-arg form so existing Grafana/Prometheus usage works.
 - Large-size company makes telemetry busier (burstier, larger magnitudes).
 - Region metadata cards added in main body and sidebar.
 - Prometheus metric names left unchanged.
"""

import logging
import math
import random
import socket
import time
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import threading
import json
import io
import csv
import copy
import os

import requests
import streamlit as st  # Streamlit UI

# Must be the first Streamlit page config call in the process.
try:
    st.set_page_config(
        page_title="Argus Agent",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------
# Timezone helper
# ---------------------------
MANILA_TZ = ZoneInfo("Asia/Manila")


def now_manila():
    """Return current datetime in Asia/Manila timezone."""
    return datetime.now(MANILA_TZ)


# ---------------------------
# Basic config
# ---------------------------
METRICS_PORT = 8000
INTERVAL = 15  # seconds between emitted snapshots
SLEEP_STEP = 1
MAX_FEED = 20

# Constants (kept for UI/context only)
CPU_WATTS = 65.0
GPU_WATTS = 200.0
GRID_CF = 0.691  # kgCO2e per kWh


# --- Expansion / cardinality configuration (tunable) ---
# You can configure these via environment variables, e.g. ARGUS_CONTAINERS_MIN, ARGUS_CONTAINERS_MAX, ARGUS_HOST_RANGE
def _env_int(name, default):
    try:
        v = os.environ.get(name)
        return int(v) if v is not None else default
    except Exception:
        return default

EXPAND_CONFIG = {
    "containers_min": _env_int("ARGUS_CONTAINERS_MIN", 6),
    "containers_max": _env_int("ARGUS_CONTAINERS_MAX", 60),
    "vms_min": _env_int("ARGUS_VMS_MIN", 4),
    "vms_max": _env_int("ARGUS_VMS_MAX", 20),
    "apps_min": _env_int("ARGUS_APPS_MIN", 6),
    "apps_max": _env_int("ARGUS_APPS_MAX", 60),
    "net_min": _env_int("ARGUS_NET_MIN", 4),
    "net_max": _env_int("ARGUS_NET_MAX", 24),
    "host_range": _env_int("ARGUS_HOST_RANGE", 200),
    "container_random_id": os.environ.get("ARGUS_CONTAINER_RANDOM_ID", "uuid").lower(),  # "uuid" or "numeric"
}

# Optional deterministic seeding for reproducible labelsets:
SEED_ENV = os.environ.get("ARGUS_SEED")
if SEED_ENV is not None:
    try:
        RANDOM_SEED = int(SEED_ENV)
        random.seed(RANDOM_SEED)
    except Exception:
        RANDOM_SEED = None
else:
    RANDOM_SEED = None

# Helper to create unique container id (uses UUID by default for high cardinality)
def _make_container_id(base_id=None):
    mode = EXPAND_CONFIG.get("container_random_id", "uuid")
    if mode == "numeric":
        return f"container_{random.randint(1, 1000000)}"
    # default uuid mode
    try:
        return f"container_{uuid.uuid4().hex[:12]}"
    except Exception:
        return f"container_{int(time.time()*1000) % 1000000}"


# External/config keys (demo - not used for actual live API in this sandbox)
zip_code = 1000
country_code = "PH"
lat = "12.8797"
lon = "121.7740"
api_key = "eff1f48d5f126f1208c0b00a26791796"
precipitation_key = "37573286ba4c48ba88b02651250110"
location = "Manila"
end_date = now_manila()

# Region metadata (keeps a small local copy to display)
REGION_META = {
    "Manila": {
        "wue": 0.002,
        "water_avail": 0.8,
        "no2": 0.00012,
    }
}

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("argus_streamlit")

# ---------------------------
# Module-level last-seen state for deltas & small caches
# ---------------------------
_LAST_SEEN = {}

# ---------------------------
# Metric objects (module-level) — DO NOT CHANGE metric names
# ---------------------------
try:
    # Core telemetry metrics
    g_container_cpu = Gauge(
        "argus_container_cpu_util_pct",
        "Container CPU utilization percent",
        ["instance", "host_id", "container_id", "container_image"],
    )
    g_container_memory_rss = Gauge(
        "argus_container_memory_rss_bytes",
        "Container memory RSS bytes",
        ["instance", "host_id", "container_id"],
    )
    g_container_network_rx = Gauge(
        "argus_container_network_rx_bytes",
        "Container network rx bytes",
        ["instance", "host_id", "container_id"],
    )
    g_container_uptime = Gauge(
        "argus_container_uptime_seconds",
        "Container uptime seconds",
        ["instance", "host_id", "container_id"],
    )

    g_vm_cpu = Gauge("argus_vm_cpu_pct", "VM CPU percent", ["instance", "host_id"])
    g_vm_memory_rss = Gauge(
        "argus_vm_memory_rss_bytes", "VM memory rss bytes", ["instance", "host_id"]
    )

    g_app_req_rate = Gauge(
        "argus_app_request_rate_rps",
        "Application request rate (rps)",
        ["instance", "host_id", "container_id"],
    )
    g_app_error_rate = Gauge(
        "argus_app_error_rate_pct",
        "Application error rate percent",
        ["instance", "host_id", "container_id"],
    )
    g_app_cpu = Gauge(
        "argus_app_cpu_util_pct", "App CPU util percent", ["instance", "host_id", "container_id"]
    )

    c_app_requests_total = Counter(
        "argus_app_requests_total",
        "Total number of requests handled by the app",
        ["instance", "host_id", "container_id"],
    )

    h_app_latency_seconds = Histogram(
        "argus_app_latency_seconds",
        "Request latency (seconds)",
        ["instance", "host_id", "container_id"],
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    g_orch_pod_count = Gauge("argus_orchestrator_pod_count", "Number of pods", ["instance"])
    g_orch_api_latency_ms = Gauge(
        "argus_orchestrator_api_latency_ms", "Cluster API latency ms", ["instance"]
    )

    g_net_throughput = Gauge(
        "argus_network_interface_throughput_bps",
        "Network interface throughput bps",
        ["instance", "host_id"],
    )
    g_net_rtt = Gauge("argus_network_rtt_ms", "Network RTT ms", ["instance", "host_id"])
    g_net_packet_loss = Gauge(
        "argus_network_packet_loss_pct", "Network packet loss percent", ["instance", "host_id"]
    )

    # GPU and sensor metrics
    g_container_sensor_temp_c = Gauge(
        "argus_container_sensor_temp_c", "Container sensor temperature C", ["instance", "host_id", "container_id"]
    )
    g_container_gpu_util_pct = Gauge(
        "argus_container_gpu_util_pct", "Container GPU util percent", ["instance", "host_id", "container_id"]
    )

    # zone-labeled CPU metrics
    g_container_cpu_zone = Gauge(
        "argus_container_cpu_util_pct_zone",
        "Container CPU percent with zone",
        ["instance", "host_id", "container_id", "zone"],
    )
    g_vm_cpu_zone = Gauge("argus_vm_cpu_pct_zone", "VM CPU percent with zone", ["instance", "host_id", "zone"])

    # event counters
    c_cpu_throttle_total = Counter(
        "argus_cpu_throttle_total", "CPU throttle events total", ["instance", "host_id", "container_id"]
    )
    c_pod_restarts_total = Counter("argus_pod_restarts_total", "Pod restart total", ["instance"])
    c_failover_events_total = Counter("argus_failover_events_total", "Failover events total", ["instance"])
    c_latency_spike_total = Counter(
        "argus_latency_spike_total", "Latency spike events total", ["instance", "host_id", "container_id"]
    )

    # region-level configuration exposed so Prometheus can reference them per-region
    g_region_wue = Gauge("region_wue", "Water use efficiency", ["region"])
    g_region_water_avail = Gauge("region_water_avail", "Regional water availability (normalized)", ["region"])
    g_region_no2_emission_factor = Gauge(
        "region_no2_emission_factor", "NO2 emission factor (kg per kWh)", ["region"]
    )

except Exception as e:
    logger.warning("Metric registration warning: %s", e)

# ---------------------------
# Start Prometheus server (guarded)
# ---------------------------
_METRICS_SERVER_STARTED = False
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    res = s.connect_ex(("127.0.0.1", METRICS_PORT))
    s.close()
    if res == 0:
        _METRICS_SERVER_STARTED = True
        logger.info("Metrics port %d already in use — assuming metrics server is running", METRICS_PORT)
    else:
        start_http_server(METRICS_PORT, addr="0.0.0.0")
        _METRICS_SERVER_STARTED = True
        logger.info("Started Prometheus metrics server on port %d", METRICS_PORT)
except Exception as e:
    logger.exception("Failed to start Prometheus metrics server: %s", e)

# ---------------------------
# Helpers & external fetch functions
# ---------------------------
def human_bytes(n: int) -> str:
    step_unit = 1024.0
    if n < step_unit:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= step_unit
        if n < step_unit:
            return f"{n:,.2f} {unit}"
    return f"{n:,.2f} PB"


def human_seconds(s: int) -> str:
    if s < 60:
        return f"{s} s"
    m, s2 = divmod(s, 60)
    if m < 60:
        return f"{m}m {s2}s"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m"


def fetch_weather_data():
    """Simulated weather payload"""
    return {
        "city": {"name": "Manila"},
        "list": [{
            "main": {
                "temp": random.uniform(25, 35),
            },
            "weather": [{
                "description": random.choice(["clear sky", "few clouds", "scattered clouds", "broken clouds", "shower rain", "rain", "thunderstorm", "mist"])
            }]
        }]
    }


def fetch_precipitation():
    return random.uniform(0, 200)


def fetch_air_quality():
    return {
        "list": [{
            "main": {"aqi": random.randint(1, 5)},
            "components": {
                "co": random.uniform(0, 10),
                "no": random.uniform(0, 10),
                "no2": random.uniform(0, 10),
                "o3": random.uniform(0, 10),
                "pm2_5": random.uniform(0, 10),
                "pm10": random.uniform(0, 10),
            }
        }]
    }

# ---------------------------
# Random metric generators (no-arg signatures like the original app)
# ---------------------------
def _time_phase(period_seconds=300.0):
    """Return a slow phase -1..1 for 'diurnal' style variation."""
    return math.sin(time.time() / max(1.0, period_seconds) * 2 * math.pi)


def _is_large_company():
    # safe access to session_state (default to Mid-tier)
    try:
        return st.session_state.get("company_type", "Mid-tier Company") == "Large-size Company"
    except Exception:
        return False


def generate_container_metrics():
    """
    No-arg signature. Stronger volatility for demo.
    Mid-tier: medium amplitude swings.
    Large-size: much larger amplitude, more frequent spikes/dips.
    """
    company_large = _is_large_company()
    # base slow phase for a gentle baseline wave
    phase = _time_phase(period_seconds=600.0)

    # amplitude controllers
    amp_base = 40.0 if company_large else 18.0
    noise_sigma = 10.0 if company_large else 4.5
    spike_prob = 0.22 if company_large else 0.10
    dip_prob = 0.12 if company_large else 0.06

    # core CPU baseline + phase swing
    base_cpu = (50.0 if company_large else 30.0) + amp_base * phase
    cpu_noise = random.gauss(0.0, noise_sigma)

    # random spike or dip event
    event = None
    event_delta = 0.0
    r = random.random()
    if r < spike_prob:
        # spike up
        event = "spike"
        event_delta = random.uniform(30.0, 120.0) if company_large else random.uniform(15.0, 60.0)
    elif r < spike_prob + dip_prob:
        # dip down
        event = "dip"
        event_delta = -random.uniform(20.0, 90.0) if company_large else -random.uniform(8.0, 35.0)

    cpu_val = max(0.0, min(100.0, base_cpu + cpu_noise + event_delta))

    # GPU follows CPU but scaled and independently spiky sometimes
    gpu_base_amp = 22.0 if company_large else 10.0
    gpu_noise = random.gauss(0.0, noise_sigma * 0.9)
    gpu_phase = _time_phase(period_seconds=420.0)
    gpu_val = max(0.0, min(100.0, (15.0 if company_large else 6.0) + gpu_base_amp * gpu_phase + gpu_noise + (event_delta * 0.6)))

    # temperature tracks CPU but with jitter
    sensor_temp = 24.0 + (cpu_val / (4.8 if company_large else 6.5)) + random.gauss(0, 1.6)

    # network: large companies show much bigger throughput swings
    base_net = int((300000 if company_large else 90000) * (0.35 + 0.7 * abs(phase)))
    net_noise = int(random.gauss(0, base_net * (0.35 if company_large else 0.18)))
    # occasional huge burst on large company
    if company_large and random.random() < 0.18:
        net_noise += int(random.uniform(base_net * 1.5, base_net * 6.0))
    network_rx = max(0, base_net + net_noise + int(event_delta * 2000))

    # memory and other dims scale with company size but with volatility
    mem_rss = random.randint(0, (256 if company_large else 32) * 1024**2)
    mem_limit = (512 * 1024**2) if company_large else (64 * 1024**2)

    return {
        "timestamp": now_manila().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "container_image": f"image_{random.choice(['nginx', 'mysql', 'redis', 'flask', 'django'])}",
        "cpu_util_pct": round(cpu_val, 2),
        "gpu_util_pct": round(gpu_val, 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": int(mem_rss),
        "memory_limit_bytes": int(mem_limit),
        "disk_read_bytes": random.randint(0, 1024**4) if company_large else random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**4) if company_large else random.randint(0, 1024**3),
        "io_ops": random.randint(0, 800) if company_large else random.randint(0, 200),
        "network_rx_bytes": abs(int(network_rx)),
        "network_tx_bytes": random.randint(0, 1024**5) if company_large else random.randint(0, 1024**3),
        "process_count": random.randint(1, 500) if company_large else random.randint(1, 80),
        "restart_count": random.randint(0, 5) + (1 if event == "spike" and random.random() < 0.08 else 0),
        "uptime_seconds": random.randint(0, 86400),
        "sensor_temp_c": round(sensor_temp, 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        "burst": bool(event is not None),
    }

def generate_vm_metrics():
    company_large = _is_large_company()
    phase = _time_phase(period_seconds=900.0)

    # stronger amplitude for large
    amp = 38.0 if company_large else 18.0
    base = (50.0 if company_large else 28.0) + amp * phase
    sigma = 8.0 if company_large else 4.0

    # random dramatic swings
    swing = random.gauss(0.0, sigma)
    if random.random() < (0.18 if company_large else 0.08):
        # big up or down swing
        swing += random.choice([-1, 1]) * random.uniform(20.0, 90.0 if company_large else 30.0)

    vm_cpu = max(0.0, min(100.0, base + swing))

    host_power = max(40.0, (600.0 if company_large else 280.0) + vm_cpu * (35.0 if company_large else 22.0) + random.gauss(0, 120.0))
    return {
        "timestamp": now_manila().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "vm_cpu_pct": round(vm_cpu, 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "vm_cpu_steal_pct": round(random.uniform(0.0, 8.0 if not (random.random() < 0.04) else 30.0), 2),
        "memory_rss_bytes": random.randint(0, (512 if company_large else 64) * 1024**2),
        "memory_limit_bytes": random.randint((512 if company_large else 64) * 1024**2, (2048 if company_large else 128) * 1024**2),
        "disk_iops": random.randint(0, 1000 if company_large else 200),
        "disk_read_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "disk_write_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "network_rx_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "network_tx_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "host_power_estimate_w": round(host_power, 2),
        "hypervisor_overhead_pct": round(random.uniform(0.0, 100.0), 2),
        "uptime_seconds": random.randint(0, 86400),
        "burst": random.random() < (0.12 if company_large else 0.06),
    }
def generate_app_metrics():
    company_large = _is_large_company()
    phase = _time_phase(period_seconds=400.0)

    # base RPS and amplitude
    base_rps = (220.0 if company_large else 60.0) + (120.0 if company_large else 40.0) * phase
    noise = random.gauss(0, 30.0 if company_large else 10.0)

    # very spiky behavior probability and magnitude
    spike_p = 0.28 if company_large else 0.12
    dip_p = 0.10 if company_large else 0.05

    r = random.random()
    event_delta = 0.0
    if r < spike_p:
        # big traffic spike
        event_delta = random.uniform(3.0, 12.0) if company_large else random.uniform(2.0, 6.0)
    elif r < spike_p + dip_p:
        # sudden drop
        event_delta = -random.uniform(0.5, 0.95)

    # compute request rate
    request_rate = max(0.1, (base_rps + noise) * max(0.1, 1.0 + event_delta))

    # latency reacts to spikes: p95 increases dramatically on spikes
    if event_delta > 1.0:
        p95 = random.uniform(1000.0, 6000.0) if company_large else random.uniform(600.0, 3000.0)
    else:
        p95 = abs(random.gauss(120.0 if company_large else 80.0, 120.0))

    # p99 slightly larger
    p99 = p95 + random.uniform(50.0, 1400.0)
    p50 = max(1.0, p95 * random.uniform(0.12, 0.4))

    # error rate increases with severe spikes
    base_err = random.gauss(1.0, 1.6)
    err_increase = (random.uniform(3.0, 25.0) if event_delta > 1.0 else random.uniform(0.0, 3.5))
    error_rate = min(100.0, max(0.0, base_err + err_increase))

    cpu_pct = max(0.0, min(100.0, 8.0 + request_rate / (1.7 if company_large else 3.5) + random.gauss(0, 8.0)))

    return {
        "timestamp": now_manila().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "request_rate_rps": round(request_rate, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p99_ms": round(p99, 2),
        "error_rate_pct": round(error_rate, 2),
        "db_connection_count": random.randint(0, 2000 if company_large else 300),
        "cache_hit_ratio": round(random.uniform(30.0, 99.9), 2),
        "queue_length": random.randint(0, 2000 if company_large else 250),
        "cpu_util_pct": round(cpu_pct, 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": random.randint(0, (1024 if company_large else 64) * 1024**2),
        "disk_read_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "disk_write_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "network_rx_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "network_tx_bytes": random.randint(0, 1024**6 if company_large else 1024**4),
        "process_count": random.randint(1, 1000 if company_large else 80),
        "restart_count": random.randint(0, 4),
        "sensor_temp_c": round(24.0 + random.gauss(0, 2.0) + (8.0 if event_delta > 1.0 and company_large else 0.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        "burst": (event_delta > 0.0),
    }


def generate_orchestrator_metrics():
    company_large = _is_large_company()
    base_phase = _time_phase(period_seconds=800.0)
    base_nodes = random.randint(8, 60) if company_large else random.randint(1, 18)
    # big jumps more often for large
    jump_prob = 0.18 if company_large else 0.06
    jump = 0
    if random.random() < jump_prob:
        jump = random.randint(20, 400) if company_large else random.randint(5, 40)

    pod_count = max(1, (random.randint(50, 1000) if company_large else random.randint(1, 200)) + int(base_phase * (200 if company_large else 60)) + jump)
    pod_failed = random.randint(0, 200 if company_large else 20)
    pod_pending = random.randint(0, 200 if company_large else 20)
    api_latency = round(random.uniform(10.0, 120.0 if not jump else (1200.0 if company_large else 600.0)), 2)

    return {
        "timestamp": now_manila().isoformat(),
        "node_count": base_nodes,
        "pod_count": pod_count,
        "pod_status_pending": pod_pending,
        "pod_status_running": max(0, pod_count - pod_failed - pod_pending),
        "pod_status_failed": pod_failed,
        "scheduler_evictions": random.randint(0, 500 if company_large else 40),
        "cluster_api_latency_ms": api_latency,
        "cluster_autoscaler_actions": random.randint(0, 600 if company_large else 30),
        "aggregated_cpu_util_pct": round(random.uniform(0.0, 100.0), 2),
        "aggregated_memory_rss_bytes": random.randint(0, (2048 if company_large else 128) * 1024**2),
        "aggregated_network_bytes": random.randint(0, (64 if company_large else 4) * 1024**4),
        "restart_count": random.randint(0, 60 if company_large else 6),
        "uptime_seconds": random.randint(0, 86400),
        "burst": (jump > 0),
    }


def generate_network_metrics():
    company_large = _is_large_company()
    base_phase = _time_phase(period_seconds=500.0)
    # base throughput
    if company_large:
        base_throughput = int(700000 * (0.2 + 0.9 * abs(base_phase)))
    else:
        base_throughput = int(150000 * (0.2 + 0.75 * abs(base_phase)))

    # volatility
    sigma = int(base_throughput * (0.45 if company_large else 0.18))
    throughput = max(0, int(random.gauss(base_throughput, sigma)))

    # occasional huge up/down swings
    if random.random() < (0.22 if company_large else 0.08):
        direction = random.choice([-1, 1])
        throughput = max(0, throughput + int(direction * random.uniform(base_throughput * 0.8, base_throughput * 4.0)))

    packet_loss = round(random.uniform(0.0, 2.0 if not company_large else 6.0), 2)
    if random.random() < (0.12 if company_large else 0.05):
        # bursty packet loss events
        packet_loss = round(random.uniform(5.0, 50.0) if company_large else random.uniform(2.0, 12.0), 2)

    rtt = round(random.uniform(1.0, 120.0 if not company_large else 1000.0), 2)
    jitter = round(random.uniform(0.0, 80.0 if not company_large else 600.0), 2)

    return {
        "timestamp": now_manila().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "interface_throughput_bps": throughput,
        "network_rx_bytes": max(0, throughput * random.randint(1, 8)),
        "network_tx_bytes": max(0, throughput * random.randint(1, 8)),
        "packet_loss_pct": packet_loss,
        "rtt_ms": rtt,
        "jitter_ms": jitter,
        "active_flows": random.randint(0, 20000 if company_large else 2000),
        "bgp_changes": random.randint(0, 40 if company_large else 6),
        "psu_efficiency_pct": round(random.uniform(60.0, 98.0), 2),
        "sensor_temp_c": round(random.uniform(18.0, 85.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        "burst": random.random() < (0.2 if company_large else 0.07),
    }

# ---------------------------
# Helper functions (continued)
# ---------------------------
def host_to_zone(host_id: str) -> str:
    try:
        n = int(host_id.split("_")[1])
        return "urban" if n <= 5 else "rural"
    except Exception:
        return "unknown"


def _key(instance, name):
    return f"{instance}::{name}"
# ---------------------------
# Update Prometheus metrics from generated dictionaries (NO indices computed here)
# ---------------------------
def update_prometheus_from_metrics(
    instance_id,
    collection_interval_s,
    container_metrics,
    vm_metrics,
    app_metrics,
    orchestrator_metrics,
    network_metrics,
):
    """
    Updated exporter: if the incoming section is a single dict (the UI snapshot),
    expand it into multiple simulated items for Prometheus so Grafana shows many
    containers/hosts. If the incoming section is already a list, use it directly.
    """
    inst = str(instance_id)
    global _LAST_SEEN

    # helpers: expand a single dict into a list of similar dicts
    def _expand_containers(base):

            out = []
            # decide how many containers to generate
            cnt = random.randint(EXPAND_CONFIG["containers_min"], EXPAND_CONFIG["containers_max"])
            for _ in range(cnt):
                c = copy.deepcopy(base)
                # host range expanded for higher cardinality
                c["host_id"] = f"host_{random.randint(1, EXPAND_CONFIG['host_range'])}"
                # container id: prefer uuid to avoid collisions
                c["container_id"] = _make_container_id(c.get("container_id"))
                c["container_image"] = c.get("container_image", "image_nginx")
                # perturb numeric values slightly but bounded
                try:
                    c["cpu_util_pct"] = round(
                        max(0.0, min(100.0, float(c.get("cpu_util_pct", 0.0)) + random.gauss(0, 6.0))),
                        2,
                    )
                except Exception:
                    c["cpu_util_pct"] = float(c.get("cpu_util_pct", 0.0))
                try:
                    c["memory_rss_bytes"] = int(max(0, int(c.get("memory_rss_bytes", 0)) * random.uniform(0.7, 1.6)))
                except Exception:
                    c["memory_rss_bytes"] = int(c.get("memory_rss_bytes", 0))
                # ensure uptime increases slightly
                try:
                    c["uptime_seconds"] = int(max(0, int(c.get("uptime_seconds", 0)) + random.randint(0, 300)))
                except Exception:
                    c["uptime_seconds"] = int(c.get("uptime_seconds", 0))
                out.append(c)
            return out

    def _expand_vms(base):

            out = []
            cnt = random.randint(EXPAND_CONFIG["vms_min"], EXPAND_CONFIG["vms_max"])
            for _ in range(cnt):
                v = copy.deepcopy(base)
                v["host_id"] = f"host_{random.randint(1, EXPAND_CONFIG['host_range'])}"
                try:
                    v["vm_cpu_pct"] = round(max(0.0, min(100.0, float(v.get("vm_cpu_pct", 0.0)) + random.gauss(0, 5.0))), 2)
                except Exception:
                    v["vm_cpu_pct"] = float(v.get("vm_cpu_pct", 0.0))
                try:
                    v["vm_memory_rss"] = int(max(0, int(v.get("vm_memory_rss", 0)) * random.uniform(0.75, 1.5)))
                except Exception:
                    v["vm_memory_rss"] = int(v.get("vm_memory_rss", 0))
                out.append(v)
            return out

    def _expand_apps(base):

            out = []
            cnt = random.randint(EXPAND_CONFIG["apps_min"], EXPAND_CONFIG["apps_max"])
            for _ in range(cnt):
                a = copy.deepcopy(base)
                a["host_id"] = f"host_{random.randint(1, EXPAND_CONFIG['host_range'])}"
                a["container_id"] = _make_container_id(a.get("container_id"))
                try:
                    a["request_rate_rps"] = max(0.1, float(a.get("request_rate_rps", 0.0)) * random.uniform(0.6, 1.8))
                except Exception:
                    a["request_rate_rps"] = float(a.get("request_rate_rps", 0.0))
                # recompute cpu slightly w.r.t requests
                try:
                    a["cpu_util_pct"] = round(max(0.0, min(100.0, 8.0 * float(a["request_rate_rps"]) / 3.0 + random.gauss(0, 6.0))), 2)
                except Exception:
                    a["cpu_util_pct"] = float(a.get("cpu_util_pct", 0.0))
                try:
                    a["error_rate_pct"] = max(0.0, min(100.0, float(a.get("error_rate_pct", 0.0)) * random.uniform(0.8, 1.3)))
                except Exception:
                    a["error_rate_pct"] = float(a.get("error_rate_pct", 0.0))
                out.append(a)
            return out

    def _expand_networks(base):

            out = []
            cnt = random.randint(EXPAND_CONFIG["net_min"], EXPAND_CONFIG["net_max"])
            for i in range(cnt):
                n = copy.deepcopy(base)
                n["host_id"] = f"host_{random.randint(1, EXPAND_CONFIG['host_range'])}"
                # ensure interface name uniqueness per host
                try:
                    base_if = str(n.get("interface_name", f"eth{i}"))
                    n["interface_name"] = f"{base_if}_{random.randint(1,9999)}"
                    n["interface_throughput_bps"] = int(max(0, int(n.get("interface_throughput_bps", 0)) * random.uniform(0.5, 1.6)))
                except Exception:
                    n["interface_throughput_bps"] = int(n.get("interface_throughput_bps", 0))
                out.append(n)
            return out
    # ---------------------------
    # container - can be a single dict or a list
    # ---------------------------
    try:
        containers = []
        if container_metrics:
            if isinstance(container_metrics, list):
                containers = container_metrics
            elif isinstance(container_metrics, dict):
                containers = _expand_containers(container_metrics)
            # iterate and set metrics for each container item
            for cm in containers:
                host = str(cm.get("host_id", "unknown"))
                cid = str(cm.get("container_id", "unknown"))
                image = str(cm.get("container_image", "unknown"))
                try:
                    cpu_val = float(cm.get("cpu_util_pct", 0.0))
                except Exception:
                    cpu_val = 0.0

                g_container_cpu.labels(
                    instance=inst, host_id=host, container_id=cid, container_image=image
                ).set(cpu_val)

                g_container_cpu_zone.labels(
                    instance=inst, host_id=host, container_id=cid, zone=host_to_zone(host)
                ).set(cpu_val)

                g_container_memory_rss.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(cm.get("memory_rss_bytes", 0))
                )
                g_container_network_rx.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(cm.get("network_rx_bytes", 0))
                )
                g_container_uptime.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(cm.get("uptime_seconds", 0))
                )

                g_container_gpu_util_pct.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(cm.get("gpu_util_pct", 0.0))
                )
                g_container_sensor_temp_c.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(cm.get("sensor_temp_c", 0.0))
                )

                # throttle simulation -> counter (occasional)
                throttle_key = _key(inst, f"throttle::{host}::{cid}")
                prev_throttle = _LAST_SEEN.get(throttle_key, 0)
                throttle_events = 0

                burst_flag = bool(cm.get("burst", False))
                prob = 0.02
                if cpu_val > 85.0:
                    prob = 0.25
                if burst_flag:
                    prob = max(prob, 0.35)

                if random.random() < prob:
                    throttle_events = random.randint(1, 1 if not burst_flag else 4)

                if throttle_events > 0:
                    c_cpu_throttle_total.labels(instance=inst, host_id=host, container_id=cid).inc(throttle_events)
                    _LAST_SEEN[throttle_key] = prev_throttle + throttle_events

                # pod restart delta -> counter (track simulated restarts per-container if present)
                try:
                    rest_key = _key(inst, f"restarts::{host}::{cid}")
                    prev = _LAST_SEEN.get(rest_key, 0)
                    cur = int(cm.get("restart_count", 0))
                    if cur > prev:
                        delta = cur - prev
                        if cm.get("burst", False) and random.random() < 0.15:
                            delta += random.randint(0, 3)
                        c_pod_restarts_total.labels(instance=inst).inc(delta)
                        _LAST_SEEN[rest_key] = cur
                except Exception:
                    pass

    except Exception:
        logger.exception("Failed to set container metrics")

    # ---------------------------
    # VM - accept single dict or list; expand if single
    # ---------------------------
    try:
        vms = []
        if vm_metrics:
            if isinstance(vm_metrics, list):
                vms = vm_metrics
            elif isinstance(vm_metrics, dict):
                vms = _expand_vms(vm_metrics)
            for vm in vms:
                host = str(vm.get("host_id", "unknown"))
                try:
                    vm_cpu_val = float(vm.get("vm_cpu_pct", 0.0))
                except Exception:
                    vm_cpu_val = 0.0
                g_vm_cpu.labels(instance=inst, host_id=host).set(vm_cpu_val)
                g_vm_cpu_zone.labels(instance=inst, host_id=host, zone=host_to_zone(host)).set(vm_cpu_val)
                g_vm_memory_rss.labels(instance=inst, host_id=host).set(float(vm.get("memory_rss_bytes", 0)))
                _LAST_SEEN.setdefault("zone_latest_cpu", {})[host] = vm_cpu_val
                _LAST_SEEN[f"host_power_w::{host}"] = float(vm.get("host_power_estimate_w", 0.0))
    except Exception:
        logger.exception("Failed to set VM metrics")

    # ---------------------------
    # APP - accept single dict or list; expand if single
    # ---------------------------
    try:
        apps = []
        if app_metrics:
            if isinstance(app_metrics, list):
                apps = app_metrics
            elif isinstance(app_metrics, dict):
                apps = _expand_apps(app_metrics)
            for am in apps:
                host = str(am.get("host_id", "unknown"))
                cid = str(am.get("container_id", "unknown"))

                g_app_req_rate.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(am.get("request_rate_rps", 0.0))
                )
                g_app_error_rate.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(am.get("error_rate_pct", 0.0))
                )
                g_app_cpu.labels(instance=inst, host_id=host, container_id=cid).set(
                    float(am.get("cpu_util_pct", 0.0))
                )

                try:
                    rps = float(am.get("request_rate_rps", 0.0))
                except Exception:
                    rps = 0.0
                approx_delta = max(0.0, rps * float(collection_interval_s))
                if am.get("burst", False):
                    approx_delta *= random.uniform(2.0, 6.0)
                if approx_delta > 0:
                    try:
                        c_app_requests_total.labels(instance=inst, host_id=host, container_id=cid).inc(approx_delta)
                    except Exception:
                        logger.exception("Failed to increment app requests counter")

                try:
                    p50_ms = am.get("latency_p50_ms")
                    p95_ms = am.get("latency_p95_ms")
                    p99_ms = am.get("latency_p99_ms")
                    if p50_ms is not None:
                        h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(
                            float(p50_ms) / 1000.0
                        )
                    if p95_ms is not None:
                        h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(
                            float(p95_ms) / 1000.0
                        )
                    if p99_ms is not None:
                        h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(
                            float(p99_ms) / 1000.0
                        )
                except Exception:
                    logger.exception("Failed to observe latency histogram")

                try:
                    p95_val = float(am.get("latency_p95_ms", 0.0))
                    spike_key = _key(inst, f"latency_spike::{host}::{cid}")
                    prev_spikes = _LAST_SEEN.get(spike_key, 0)
                    spike_inc = 0
                    if p95_val > 500.0:
                        spike_inc = 1 if not am.get("burst", False) else random.randint(1, 6)
                    if spike_inc > 0:
                        c_latency_spike_total.labels(instance=inst, host_id=host, container_id=cid).inc(spike_inc)
                        _LAST_SEEN[spike_key] = prev_spikes + spike_inc
                except Exception:
                    logger.exception("Latency spike logic failed")

                try:
                    rest_key_app = _key(inst, f"restarts_app::{host}::{cid}")
                    prev_a = _LAST_SEEN.get(rest_key_app, 0)
                    cura = int(am.get("restart_count", 0))
                    if cura > prev_a:
                        delta = cura - prev_a
                        c_pod_restarts_total.labels(instance=inst).inc(delta)
                        _LAST_SEEN[rest_key_app] = cura
                except Exception:
                    pass

    except Exception:
        logger.exception("Failed to set app metrics")

    # ---------------------------
    # Orchestrator (unchanged) & network
    # ---------------------------
    try:
        if orchestrator_metrics:
            g_orch_pod_count.labels(instance=inst).set(int(orchestrator_metrics.get("pod_count", 0)))
            g_orch_api_latency_ms.labels(instance=inst).set(float(orchestrator_metrics.get("cluster_api_latency_ms", 0.0)))

            orch_restart_key = _key(inst, "orch_restarts")
            prev_orch = _LAST_SEEN.get(orch_restart_key, 0)
            cur_orch = int(orchestrator_metrics.get("restart_count", 0))
            if cur_orch > prev_orch:
                c_pod_restarts_total.labels(instance=inst).inc(cur_orch - prev_orch)
                _LAST_SEEN[orch_restart_key] = cur_orch

            failed = int(orchestrator_metrics.get("pod_status_failed", 0))
            fail_key = _key(inst, "failed_pods")
            prev_failed = _LAST_SEEN.get(fail_key, 0)
            if failed > prev_failed and failed > 2:
                delta_fail = max(0, failed - prev_failed)
                if orchestrator_metrics.get("burst", False):
                    delta_fail += random.randint(0, 6)
                c_failover_events_total.labels(instance=inst).inc(delta_fail)
                _LAST_SEEN[fail_key] = failed
    except Exception:
        logger.exception("Failed to set orchestrator metrics")

    try:
        networks = []
        if network_metrics:
            if isinstance(network_metrics, list):
                networks = network_metrics
            elif isinstance(network_metrics, dict):
                networks = _expand_networks(network_metrics)
            for nm in networks:
                host = str(nm.get("host_id", "unknown"))
                g_net_throughput.labels(instance=inst, host_id=host).set(
                    float(nm.get("interface_throughput_bps", 0))
                )
                g_net_rtt.labels(instance=inst, host_id=host).set(float(nm.get("rtt_ms", 0.0)))
                g_net_packet_loss.labels(instance=inst, host_id=host).set(
                    float(nm.get("packet_loss_pct", 0.0))
                )
                try:
                    g_container_sensor_temp_c.labels(instance=inst, host_id=host, container_id="network_sensor").set(
                        float(nm.get("sensor_temp_c", 0.0))
                    )
                except Exception:
                    pass
    except Exception:
        logger.exception("Failed to set network metrics")


# ---------------------------
# Seed helper (added) - call this right after last_metrics is initialized
# ---------------------------
def seed_prometheus():
    """Seed Prometheus so /metrics has values immediately."""
    try:
        # ensure session_state has last_metrics and an instance id
        if "instance_id" not in st.session_state or "last_metrics" not in st.session_state:
            return
        update_prometheus_from_metrics(
            st.session_state.instance_id,
            INTERVAL,
            st.session_state.last_metrics.get("container"),
            st.session_state.last_metrics.get("vm"),
            st.session_state.last_metrics.get("app"),
            st.session_state.last_metrics.get("orchestrator"),
            st.session_state.last_metrics.get("network"),
        )
        logger.info("Seeded Prometheus metrics for instance %s", st.session_state.instance_id)
    except Exception:
        logger.exception("Failed to seed Prometheus metrics")

# ---------------------------
# Background fetcher for external APIs
# ---------------------------
def _start_background_fetch(interval_s=120):
    if st.session_state.get("_bg_fetcher_started"):
        return
    st.session_state["_bg_fetcher_started"] = True

    st.session_state.setdefault("weather_data", None)
    st.session_state.setdefault("precipitation_total", None)
    st.session_state.setdefault("air_quality_data", None)
    st.session_state.setdefault("external_last_update", None)

    def _bg_loop():
        while True:
            try:
                st.session_state["weather_data"] = fetch_weather_data()
            except Exception as e:
                logger.exception("Background weather fetch failed: %s", e)
            try:
                st.session_state["precipitation_total"] = fetch_precipitation()
            except Exception as e:
                logger.exception("Background precipitation fetch failed: %s", e)
            try:
                st.session_state["air_quality_data"] = fetch_air_quality()
            except Exception as e:
                logger.exception("Background air quality fetch failed: %s", e)
            try:
                st.session_state["external_last_update"] = now_manila().isoformat()
            except Exception:
                st.session_state["external_last_update"] = datetime.utcnow().isoformat() + "Z"
            time.sleep(interval_s)

    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()

# ---------------------------
# Utilities for download
# ---------------------------
def _flatten_snapshot(snapshot: dict) -> dict:
    out = {
        "emit_seq": snapshot.get("emit_seq"),
        "timestamp": snapshot.get("timestamp"),
        "instance_id": snapshot.get("instance_id"),
    }
    metrics = snapshot.get("metrics", {})
    for section in ["container", "vm", "app", "orchestrator", "network"]:
        sec = metrics.get(section, {})
        if isinstance(sec, dict):
            for k, v in sec.items():
                key = f"{section}_{k}"
                if isinstance(v, (dict, list)):
                    out[key] = json.dumps(v)
                else:
                    out[key] = v
        else:
            out[section] = json.dumps(sec)
    return out


def _generate_csv_bytes(history: list) -> bytes:
    if not history:
        return b""
    rows = [_flatten_snapshot(s) for s in history]
    fieldnames = set()
    for r in rows:
        fieldnames.update(r.keys())
    ordered = ["emit_seq", "timestamp", "instance_id"] + sorted(k for k in fieldnames if k not in {"emit_seq", "timestamp", "instance_id"})
    sio = io.StringIO()
    writer = csv.DictWriter(sio, fieldnames=ordered, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return sio.getvalue().encode("utf-8")


# ---------------------------
# Logo loader for landing page
# ---------------------------
def _try_load_logo():
    candidate_paths = [
        "Agent/Assets/Argus Logo.png",
        "Agent/Assets/Argus Logo.JPG",
        "Agent/Assets/Argus Logo.jpeg",
        "Agent/Assets/Argus_Logo.png",
        "./Agent/Assets/Argus Logo.png",
        "./Assets/Argus Logo.png",
        "Assets/Argus Logo.png",
        "/mnt/data/Agent/Assets/Argus Logo.png",
        "/mnt/data/Argus Logo.png",
        "Argus Logo.png",
    ]
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return None
# ---------------------------
# Streamlit UI + main loop (landing screen + main UI)
# ---------------------------
def main():
    # initialize session state (ensure company_type exists before generators are ever called)
    if "instance_id" not in st.session_state:
        st.session_state.instance_id = str(uuid.uuid4())[:8]
    if "company_type" not in st.session_state:
        st.session_state.company_type = "Mid-tier Company"
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []
    if "emit_seq" not in st.session_state:
        st.session_state.emit_seq = 0
    if "last_emit" not in st.session_state:
        st.session_state.last_emit = time.monotonic()
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {
            "container": generate_container_metrics(),
            "vm": generate_vm_metrics(),
            "app": generate_app_metrics(),
            "orchestrator": generate_orchestrator_metrics(),
            "network": generate_network_metrics(),
        }
        # seed Prometheus so the /metrics endpoint has values immediately at startup
        try:
            seed_prometheus()
        except Exception:
            # seed_prometheus already logs; swallow to avoid UI crash
            pass
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "started" not in st.session_state:
        st.session_state.started = False

    # export region-level gauges for Prometheus reference
    try:
        g_region_wue.labels(region="Manila").set(REGION_META["Manila"]["wue"])
        g_region_water_avail.labels(region="Manila").set(REGION_META["Manila"]["water_avail"])
        g_region_no2_emission_factor.labels(region="Manila").set(REGION_META["Manila"]["no2"])
    except Exception:
        pass

    _start_background_fetch(interval_s=120)

    # sidebar style adjustments
    st.markdown(
        """
    <style>
    section[data-testid="stSidebar"] > div[style] {
    min-width: 220px;
    max-width: 220px;
    }
    section[data-testid="stSidebar"] .css-1d391kg, section[data-testid="stSidebar"] .css-1v3fvcr {
    padding: 6px 8px;
    font-size: 14px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # If user hasn't started, show the selection landing screen (single container)
    if not st.session_state.started:
        landing = st.container()   # single container for entire landing UI
        with landing:
            # header + logo (centered) — show name next to logo
            logo_path = _try_load_logo()
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                # two columns: left for logo, right for title & subtitle
                logo_col, title_col = st.columns([1, 4])
                with logo_col:
                    if logo_path:
                        try:
                            st.image(logo_path, use_column_width=False, width=96)
                        except Exception:
                            try:
                                st.image(logo_path)
                            except Exception:
                                st.markdown("<div style='font-weight:800;font-size:20px;'>Argus Agent</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='font-weight:800;font-size:20px;'>Argus Agent</div>", unsafe_allow_html=True)

                with title_col:
                    st.markdown(
                        "<div style='display:flex;flex-direction:column;justify-content:center;'>"
                        "<div style='font-size:26px;font-weight:800;color:#e2e8f0;margin-bottom:6px;'>Argus Agent</div>"
                        "<div style='color:#94a3b8;margin-top:0;'>Choose a company profile to begin the telemetry simulation.</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            st.write("")  # spacing
            sel_col1, sel_col2, sel_col3 = st.columns([1, 2, 1])
            with sel_col2:
                st.markdown("<div style='padding:18px;border-radius:12px;background:linear-gradient(135deg,#0b2a3b,#083047);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h2 style='margin-top:0;'>Start a Simulation</h2>", unsafe_allow_html=True)
                st.markdown("<div style='color:#94a3b8;margin-bottom:10px;'>Select the company profile you want to simulate. Large-size companies will produce higher and busier telemetry.</div>", unsafe_allow_html=True)

                st.session_state.setdefault("company_choice_pending", "Mid-tier Company")
                st.session_state.company_choice_pending = st.radio(
                    "Company profile:",
                    ["Mid-tier Company", "Large-size Company"],
                    index=0,
                    horizontal=False,
                )

                st.write("")  # spacing
                start_btn = st.button("Start simulating!")
                st.markdown("</div>", unsafe_allow_html=True)

        # When Start is clicked, clear landing UI and rerun (or stop)
        if start_btn:
            try:
                landing.empty()
            except Exception:
                pass

            st.session_state.company_type = st.session_state.company_choice_pending
            st.session_state.started = True
            st.session_state.emit_seq = 0
            st.session_state.metrics_history = []
            st.session_state.last_emit = time.monotonic()
            st.session_state.last_metrics = {
                "container": generate_container_metrics(),
                "vm": generate_vm_metrics(),
                "app": generate_app_metrics(),
                "orchestrator": generate_orchestrator_metrics(),
                "network": generate_network_metrics(),
            }

            # seed Prometheus immediately after Start clicked
            try:
                seed_prometheus()
            except Exception:
                pass

            try:
                st.experimental_rerun()
            except Exception:
                st.stop()

        # avoid rendering the rest until Start clicked
        return

    # Main UI after start
    st.markdown(
        """
    <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
    <div>
    <div style='font-size:22px; font-weight:800; color:#e2e8f0;'>Argus Environmental Telemetry</div>
    <div style='color:#94a3b8; margin-top:4px;'>Real-time simulated telemetry • Weather • Air Quality • System Metrics</div>
    </div>
    <div><div style='display:inline-block;padding:6px 10px;border-radius:12px;background:#10b981;color:white;font-weight:700;'>ACTIVE</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("# Argus Agent")
        st.markdown(f"**Instance ID:** {st.session_state.instance_id}")
        st.markdown("**Location:** Manila")

        countdown_sidebar = st.empty()

        # Download controls
        st.markdown("---")
        st.markdown("### Download generated telemetry")
        st.markdown("Download the full history of emitted telemetry snapshots for this instance.")
        dl_format = st.selectbox("Format", ["JSON", "CSV"], index=0)
        if len(st.session_state.metrics_history) == 0:
            st.markdown("_No telemetry emitted yet — wait for the first update._")
        else:
            if dl_format == "JSON":
                payload = json.dumps(st.session_state.metrics_history, indent=2).encode("utf-8")
                st.download_button(
                    "Download JSON",
                    data=payload,
                    file_name=f"argus_metrics_{st.session_state.instance_id}.json",
                    mime="application/json",
                )
            else:
                csv_bytes = _generate_csv_bytes(st.session_state.metrics_history)
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name=f"argus_metrics_{st.session_state.instance_id}.csv",
                    mime="text/csv",
                )

    main_area = st.container()
    left, right = main_area.columns([1.4, 1])

    container_placeholder = left.empty()
    vm_placeholder = left.empty()
    app_placeholder = left.empty()
    orchestrator_placeholder = left.empty()
    network_placeholder = left.empty()
    # region metadata in main body (small cards)
    region_meta_placeholder = left.empty()
    region_placeholder = left.empty()

    feed_box = right.empty()
    try:
        while True:
            now_mon = time.monotonic()
            elapsed = now_mon - st.session_state.last_emit
            remaining = max(0.0, INTERVAL - elapsed)
            remaining_ceil = math.ceil(remaining)

            try:
                countdown_sidebar.markdown(
                    f"<div style='background:#047857;padding:8px;border-radius:8px;color:white;text-align:center;font-weight:700;'>Next update in <strong>{remaining_ceil}s</strong></div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                pass

            if elapsed >= INTERVAL:
                st.session_state.last_metrics = {
                    "container": generate_container_metrics(),
                    "vm": generate_vm_metrics(),
                    "app": generate_app_metrics(),
                    "orchestrator": generate_orchestrator_metrics(),
                    "network": generate_network_metrics(),
                }
                st.session_state.last_emit = now_mon
                st.session_state.emit_seq += 1

                entry = f"{now_manila().strftime('%H:%M:%S')} | Update #{st.session_state.emit_seq} | Generated telemetry snapshot."
                st.session_state.agent_log.insert(0, entry)
                st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

                try:
                    snapshot = {
                        "emit_seq": st.session_state.emit_seq,
                        "timestamp": now_manila().isoformat(),
                        "instance_id": st.session_state.instance_id,
                        "metrics": copy.deepcopy(st.session_state.last_metrics),
                    }
                    st.session_state.metrics_history.insert(0, snapshot)
                    MAX_HISTORY = 5000
                    if len(st.session_state.metrics_history) > MAX_HISTORY:
                        st.session_state.metrics_history = st.session_state.metrics_history[:MAX_HISTORY]
                except Exception:
                    logger.exception("Failed to record snapshot into history for download")

                try:
                    update_prometheus_from_metrics(
                        st.session_state.instance_id,
                        INTERVAL,
                        st.session_state.last_metrics["container"],
                        st.session_state.last_metrics["vm"],
                        st.session_state.last_metrics["app"],
                        st.session_state.last_metrics["orchestrator"],
                        st.session_state.last_metrics["network"],
                    )
                    st.session_state.agent_log.insert(0, f"Prometheus metrics updated (emit #{st.session_state.emit_seq})")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
                except Exception as e:
                    logger.exception("Failed to update Prometheus metrics: %s", e)
                    st.session_state.agent_log.insert(0, f"Failed to update Prometheus metrics: {e}")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

            container_metrics = st.session_state.last_metrics["container"]
            vm_metrics = st.session_state.last_metrics["vm"]
            app_metrics = st.session_state.last_metrics["app"]
            orchestrator_metrics = st.session_state.last_metrics["orchestrator"]
            network_metrics = st.session_state.last_metrics["network"]

            # Container card
            with container_placeholder.container():
                st.markdown("<div style='margin-bottom:8px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Container Infrastructure</h3>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:#94a3b8;font-size:13px'>{container_metrics['container_image']} • {container_metrics['container_id']} • Host: {container_metrics['host_id']}</div>",
                    unsafe_allow_html=True,
                )
                k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
                k1.metric("CPU Usage", f"{container_metrics['cpu_util_pct']:.1f}%")
                k2.metric("Memory", f"{human_bytes(container_metrics['memory_rss_bytes'])}")
                k3.metric("Network RX", f"{human_bytes(container_metrics['network_rx_bytes'])}")
                k4.metric("Uptime", f"{human_seconds(container_metrics['uptime_seconds'])}")
                with st.expander("View raw JSON - container", expanded=False):
                    st.json(container_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # VM card
            with vm_placeholder.container():
                st.markdown("<div style='margin-top:6px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Virtual Machine</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>Host: {vm_metrics['host_id']}</div>", unsafe_allow_html=True)
                v1, v2, v3, v4 = st.columns([1, 1, 1, 1])
                v1.metric("CPU %", f"{vm_metrics['vm_cpu_pct']}%")
                v2.metric("Memory", f"{human_bytes(vm_metrics['memory_rss_bytes'])}")
                v3.metric("Disk IOPS", f"{vm_metrics['disk_iops']}")
                v4.metric("Power Est.", f"{vm_metrics['host_power_estimate_w']} W")
                with st.expander("View raw JSON - vm", expanded=False):
                    st.json(vm_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # App card
            with app_placeholder.container():
                st.markdown("<div style='margin-top:6px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Application</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>{app_metrics['container_id']} on {app_metrics['host_id']}</div>", unsafe_allow_html=True)
                a1, a2, a3 = st.columns([1, 1, 1])
                a1.metric("Req/s", f"{app_metrics['request_rate_rps']}")
                a2.metric("P95 (ms)", f"{app_metrics['latency_p95_ms']}")
                a3.metric("Errors %", f"{app_metrics['error_rate_pct']}%")
                with st.expander("View raw JSON - app", expanded=False):
                    st.json(app_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Orchestrator card
            with orchestrator_placeholder.container():
                st.markdown("<div style='margin-top:6px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Orchestrator</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>Nodes: {orchestrator_metrics['node_count']}</div>", unsafe_allow_html=True)
                o1, o2 = st.columns([1, 1])
                o1.metric("Pods", f"{orchestrator_metrics['pod_count']}")
                o2.metric("API Latency ms", f"{orchestrator_metrics['cluster_api_latency_ms']}")
                with st.expander("View raw JSON - orchestrator", expanded=False):
                    st.json(orchestrator_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Network card
            with network_placeholder.container():
                st.markdown("<div style='margin-top:6px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>Network</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>Host: {network_metrics['host_id']}</div>", unsafe_allow_html=True)
                n1, n2, n3 = st.columns([1, 1, 1])
                n1.metric("Throughput bps", f"{network_metrics['interface_throughput_bps']}")
                n2.metric("RTT ms", f"{network_metrics['rtt_ms']}")
                n3.metric("Packet Loss %", f"{network_metrics['packet_loss_pct']}%")
                with st.expander("View raw JSON - network", expanded=False):
                    st.json(network_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Region metadata cards in main body (small cards)
            try:
                rm = REGION_META.get("Manila", {})
                wue = rm.get("wue", "—")
                water = rm.get("water_avail", "—")
                no2 = rm.get("no2", "—")

                region_meta_placeholder.markdown(
                    f"""
                    <div style="display:flex;gap:12px;margin-top:8px;">
                      <div style="flex:1;padding:12px;border-radius:12px;background:linear-gradient(135deg,#082936,#033649);color:#e2e8f0;">
                        <div style="font-size:13px;font-weight:700;">WUE</div>
                        <div style="font-size:20px;margin-top:8px;font-weight:800;">{wue}</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:8px;">Prometheus metric: <code>region_wue</code></div>
                      </div>
                      <div style="flex:1;padding:12px;border-radius:12px;background:linear-gradient(135deg,#082936,#033649);color:#e2e8f0;">
                        <div style="font-size:13px;font-weight:700;">Water availability</div>
                        <div style="font-size:20px;margin-top:8px;font-weight:800;">{water}</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:8px;">Prometheus metric: <code>region_water_avail</code></div>
                      </div>
                      <div style="flex:1;padding:12px;border-radius:12px;background:linear-gradient(135deg,#082936,#033649);color:#e2e8f0;">
                        <div style="font-size:13px;font-weight:700;">NO₂ factor</div>
                        <div style="font-size:20px;margin-top:8px;font-weight:800;">{no2}</div>
                        <div style="font-size:12px;color:#94a3b8;margin-top:8px;">Prometheus metric: <code>region_no2_emission_factor</code></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception:
                try:
                    region_meta_placeholder.empty()
                except Exception:
                    pass

            # External data card (weather / precipitation / air quality)
            ext_weather = st.session_state.get("weather_data")
            ext_precip = st.session_state.get("precipitation_total")
            ext_aq = st.session_state.get("air_quality_data")
            ext_last = st.session_state.get("external_last_update")

            region_placeholder.markdown("<div style='margin-top:8px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#083047,#0b3f56);color:#e2e8f0;'>", unsafe_allow_html=True)
            region_placeholder.markdown("<h3 style='margin:0 0 6px 0;'>External Data</h3>", unsafe_allow_html=True)

            weather_summary = "Not available"
            try:
                if ext_weather:
                    city = ext_weather.get("city", {}).get("name") or ext_weather.get("city")
                    first = None
                    if isinstance(ext_weather.get("list"), list) and len(ext_weather["list"]) > 0:
                        first = ext_weather["list"][0]
                    temp = None
                    desc = None
                    if first and isinstance(first, dict):
                        main = first.get("main", {})
                        weather_arr = first.get("weather", [])
                        temp = main.get("temp") if isinstance(main, dict) else None
                        if isinstance(weather_arr, list) and weather_arr:
                            desc = weather_arr[0].get("description")
                    pieces = []
                    if city:
                        pieces.append(str(city))
                    if temp is not None:
                        pieces.append(f"{temp}°C")
                    if desc:
                        pieces.append(desc.capitalize())
                    weather_summary = " • ".join(pieces) if pieces else "Available"
            except Exception:
                weather_summary = "Available (parse failed)"

            region_placeholder.markdown(f"<div style='font-weight:700'>Weather:</div> {weather_summary}", unsafe_allow_html=True)

            prec_text = f"{ext_precip} mm (30d total)" if ext_precip is not None else "Not available"
            region_placeholder.markdown(f"<div style='margin-top:6px;font-weight:700'>Precipitation:</div> {prec_text}", unsafe_allow_html=True)

            aq_text = "Not available"
            try:
                if ext_aq:
                    if isinstance(ext_aq.get("list"), list) and len(ext_aq["list"]) > 0:
                        a0 = ext_aq["list"][0]
                        aqi = a0.get("main", {}).get("aqi")
                        if aqi is not None:
                            aq_text = f"AQI: {aqi}"
                        else:
                            comps = a0.get("components")
                            if comps:
                                comps_short = ", ".join(f"{k}:{round(v,3)}" for k, v in list(comps.items())[:3])
                                aq_text = f"Components: {comps_short}"
                            else:
                                aq_text = "Available"
            except Exception:
                aq_text = "Available (parse failed)"

            region_placeholder.markdown(f"<div style='margin-top:6px;font-weight:700'>Air quality:</div> {aq_text}", unsafe_allow_html=True)

            if ext_last:
                region_placeholder.markdown(f"<div style='margin-top:8px;color:#94a3b8;font-size:12px'>External data last updated: {ext_last}</div>", unsafe_allow_html=True)
            else:
                region_placeholder.markdown(f"<div style='margin-top:8px;color:#94a3b8;font-size:12px'>External data not yet fetched</div>", unsafe_allow_html=True)

           
            # Right column: live feed
            feed_box.markdown("<div style='font-weight:800;margin-bottom:6px;'>Live Feed</div>", unsafe_allow_html=True)
            feed_box.markdown(
                f"<div style='padding:8px;'><div style='background:linear-gradient(135deg,#021026,#081520);padding:12px;border-radius:8px;color:#e2e8f0;min-height:200px;max-height:680px;overflow-y:auto;'><pre style='white-space:pre-wrap;margin:0;font-size:13px'>{'<br>'.join(st.session_state.agent_log)}</pre></div></div>",
                unsafe_allow_html=True,
            )

            time.sleep(SLEEP_STEP)

    except Exception as e:
        err_entry = f"[{now_manila().isoformat()}] ERROR | {str(e)}"
        st.session_state.agent_log.insert(0, err_entry)
        st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
        feed_box.markdown("<pre style='white-space:pre-wrap'>" + "\n".join(st.session_state.agent_log) + "</pre>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
