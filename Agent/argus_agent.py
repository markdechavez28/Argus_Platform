#!/usr/bin/env python3
"""
Revised Argus Streamlit telemetry app + Prometheus exporter (merged & cleaned).
- Exports raw metrics and event counters needed by Prometheus to compute indices
- Does NOT compute environmental indices locally anymore (Prometheus should compute them)
- UI updated to show region metadata and instruct that indices are computed in Prometheus
- Generators produce stronger, correlated, and bursty randomness so Grafana dashboards look lively.
"""

import logging
import math
import random
import socket
import time
import uuid
from datetime import datetime, timedelta
import threading

import requests
import streamlit as st  # <-- import streamlit here and set page config immediately

# Must be the first Streamlit command executed in the process.
try:
    st.set_page_config(
        page_title="Argus Environmental Telemetry (Revised)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    # If some environment already set it, continue.
    pass

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------
# Basic config
# ---------------------------
METRICS_PORT = 8000
INTERVAL = 15  # seconds
SLEEP_STEP = 1
MAX_FEED = 20

# Constants (kept for UI/context only)
CPU_WATTS = 65.0
GPU_WATTS = 200.0
GRID_CF = 0.691  # kgCO2e per kWh

# External/config keys (demo) — move to env vars for production
zip_code = 1000
country_code = "PH"
lat = "12.8797"
lon = "121.7740"
api_key = "eff1f48d5f126f1208c0b00a26791796"
precipitation_key = "37573286ba4c48ba88b02651250110"
location = "Manila"
end_date = datetime.now()

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
logger = logging.getLogger("argus_streamlit_revised")

# ---------------------------
# Module-level last-seen state for deltas & small caches
# ---------------------------
_LAST_SEEN = {}

# ---------------------------
# Metric objects (module-level)
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
        "h_app_latency_seconds",
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

    # NEW metrics required for Prometheus-side environmental indices
    # Ensure GPU metric exists so Prometheus can discover it
    g_container_sensor_temp_c = Gauge(
        "argus_container_sensor_temp_c", "Container sensor temperature C", ["instance", "host_id", "container_id"]
    )
    g_container_gpu_util_pct = Gauge(
        "argus_container_gpu_util_pct", "Container GPU util percent", ["instance", "host_id", "container_id"]
    )

    # zone-labeled CPU for urban/rural grouping
    g_container_cpu_zone = Gauge(
        "argus_container_cpu_util_pct_zone",
        "Container CPU percent with zone",
        ["instance", "host_id", "container_id", "zone"],
    )
    g_vm_cpu_zone = Gauge("argus_vm_cpu_pct_zone", "VM CPU percent with zone", ["instance", "host_id", "zone"])

    # event counters (Prometheus counters; use increase() in PromQL)
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
    logger.warning("Metric registration warning (revised): %s", e)

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
        start_http_server(METRICS_PORT)
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
    """
    Lightweight OpenWeatherMap forecast fetch (metric units).
    Returns parsed JSON or None.
    """
    weather_url = f"https://api.openweathermap.org/data/2.5/forecast?zip={zip_code},{country_code}&appid={api_key}&units=metric"
    try:
        response = requests.get(weather_url, timeout=8)
    except Exception as e:
        logger.exception("Weather API request failed: %s", e)
        return None
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Weather API Error: %s", response.status_code)
        return None


def fetch_precipitation():
    """
    Heavy: loops 30 days and fetches historic precipitation from weatherapi.com.
    DON'T call this frequently — background fetcher refreshes it slowly by default.
    Returns total precipitation (mm) for the last 30 days or None on repeated failures.
    """
    total_precipitation = 0.0
    did_any = False
    for day in range(30):
        date = end_date - timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        precipitation_url = f"http://api.weatherapi.com/v1/history.json?key={precipitation_key}&q={location}&dt={date_str}"
        try:
            response = requests.get(precipitation_url, timeout=8)
        except Exception as e:
            logger.exception("Precipitation API request failed for %s: %s", date_str, e)
            continue
        if response.status_code == 200:
            data = response.json()
            if "forecast" in data and "forecastday" in data["forecast"] and len(data["forecast"]["forecastday"]) > 0:
                total_precipitation += data["forecast"]["forecastday"][0]["day"].get("totalprec_mm", 0)
                did_any = True
            else:
                logger.error("Error fetching precipitation data for %s: %s", date_str, response.status_code)
        else:
            logger.error("Precipitation API HTTP error for %s: %s", date_str, response.status_code)
    if not did_any:
        return None
    return total_precipitation


def fetch_air_quality():
    """
    OpenWeatherMap air pollution endpoint.
    """
    air_quality_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(air_quality_url, timeout=8)
    except Exception as e:
        logger.exception("Air Quality API request failed: %s", e)
        return None
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Air Quality API Error: %s", response.status_code)
        return None


# ---------------------------
# Random metric generators (burstier / more visible randomness)
# ---------------------------
def _time_phase(period_seconds=300.0):
    """Return a slow phase -1..1 for 'diurnal' style variation."""
    return math.sin(time.time() / max(1.0, period_seconds) * 2 * math.pi)


def generate_container_metrics():
    now = time.time()
    phase = _time_phase(period_seconds=600.0)  # slow ~10m wave

    base_cpu = 30.0 + 18.0 * phase + random.gauss(0, 4.0)

    # occasional burst (spike) with small probability
    burst = random.random() < 0.08  # 8% chance
    if burst:
        burst_boost = random.uniform(20.0, 55.0)
    else:
        burst_boost = 0.0

    cpu_val = max(0.0, min(100.0, base_cpu + burst_boost + random.gauss(0, 3.0)))

    # GPU slightly correlated but independent spiking occasionally
    base_gpu = 8.0 + 12.0 * _time_phase(period_seconds=400.0) + random.gauss(0, 3.0)
    gpu_val = max(0.0, min(100.0, base_gpu + (burst_boost * 0.6) + random.gauss(0, 5.0)))

    # sensor temp (correlates weakly with cpu, plus noise)
    sensor_temp = 24.0 + (cpu_val / 6.0) + random.gauss(0, 1.5)
    if burst and random.random() < 0.6:
        sensor_temp += random.uniform(1.5, 6.0)

    # network values: baseline plus bursty spikes when cpu bursts
    base_net = max(1000, int(100000 * (0.2 + 0.6 * abs(phase))))
    network_rx = base_net + (int(random.gauss(0, base_net * 0.2)))
    if burst and random.random() < 0.6:
        network_rx += int(random.uniform(base_net * 1.0, base_net * 4.0))

    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "container_image": f"image_{random.choice(['nginx', 'mysql', 'redis', 'flask'])}",
        "cpu_util_pct": round(cpu_val, 2),
        "gpu_util_pct": round(gpu_val, 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "memory_limit_bytes": random.randint(8 * 1024**2, 16 * 1024**2),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "io_ops": random.randint(0, 100),
        "network_rx_bytes": abs(int(network_rx)),
        "network_tx_bytes": random.randint(0, 1024**3),
        "process_count": random.randint(1, 50),
        # restart_count more likely to increase slightly during bursts
        "restart_count": random.randint(0, 2) + (1 if burst and random.random() < 0.05 else 0),
        "uptime_seconds": random.randint(0, 86400),
        "sensor_temp_c": round(sensor_temp, 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        # internal debug flag (not exported) to influence update logic
        "burst": burst,
    }


def generate_vm_metrics():
    now = time.time()
    phase = _time_phase(period_seconds=900.0)
    base_vm_cpu = 25.0 + 20.0 * phase + random.gauss(0, 5.0)
    burst = random.random() < 0.06
    vm_cpu = max(
        0.0,
        min(100.0, base_vm_cpu + (random.uniform(10.0, 45.0) if burst else 0.0) + random.gauss(0, 4.0)),
    )
    host_power = max(50.0, 300.0 + vm_cpu * 20.0 + random.gauss(0, 75.0))
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "vm_cpu_pct": round(vm_cpu, 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "vm_cpu_steal_pct": round(random.uniform(0.0, 5.0 if not burst else 20.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "memory_limit_bytes": random.randint(8 * 1024**2, 16 * 1024**2),
        "disk_iops": random.randint(0, 200 if burst else 100),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "host_power_estimate_w": round(host_power, 2),
        "hypervisor_overhead_pct": round(random.uniform(0.0, 100.0), 2),
        "uptime_seconds": random.randint(0, 86400),
        "burst": burst,
    }


def generate_app_metrics():
    # app metrics intentionally heavy-tail and bursty
    now = time.time()
    phase = _time_phase(period_seconds=400.0)
    base_rps = max(1.0, 60.0 + 40.0 * phase + random.gauss(0, 20.0))
    burst = random.random() < 0.07
    request_rate = base_rps * (random.uniform(1.0, 1.6) if not burst else random.uniform(2.5, 8.0))

    # latency: mostly small, but heavy tail on bursts
    if burst and random.random() < 0.85:
        p95 = random.uniform(600.0, 3000.0)
        p99 = p95 + random.uniform(50.0, 800.0)
    else:
        p95 = abs(random.gauss(120.0, 80.0))
        p99 = p95 + abs(random.gauss(10.0, 50.0))

    # error rate grows during bursts
    error_rate = min(100.0, max(0.0, random.gauss(1.5, 2.5) + (random.uniform(2.0, 12.0) if burst else 0.0)))

    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "request_rate_rps": round(request_rate, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p50_ms": round(max(1.0, p95 * random.uniform(0.2, 0.6)), 2),
        "latency_p99_ms": round(p99, 2),
        "error_rate_pct": round(error_rate, 2),
        "db_connection_count": random.randint(0, 200),
        "cache_hit_ratio": round(random.uniform(50.0, 99.0) if not burst else random.uniform(20.0, 90.0), 2),
        "queue_length": random.randint(0, 300 if burst else 100),
        "cpu_util_pct": round(
            max(0.0, min(100.0, 10.0 + request_rate / 5.0 + random.gauss(0, 8.0) + (30.0 if burst else 0.0))),
            2,
        ),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "process_count": random.randint(1, 50),
        # restart_count occasional small increases
        "restart_count": random.randint(0, 1) + (1 if burst and random.random() < 0.03 else 0),
        "sensor_temp_c": round(24.0 + (random.gauss(0, 2.0) + (2.0 if burst else 0.0)), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        "burst": burst,
    }


def generate_orchestrator_metrics():
    burst = random.random() < 0.05
    return {
        "timestamp": datetime.now().isoformat(),
        "node_count": random.randint(1, 20),
        "pod_count": random.randint(1, 200) + (20 if burst else 0),
        "pod_status_pending": random.randint(0, 20),
        "pod_status_running": random.randint(1, 200),
        "pod_status_failed": random.randint(0, 20) + (random.randint(0, 10) if burst else 0),
        "scheduler_evictions": random.randint(0, 50),
        "cluster_api_latency_ms": round(random.uniform(0.0, 300.0 if not burst else 1200.0), 2),
        "cluster_autoscaler_actions": random.randint(0, 50),
        "aggregated_cpu_util_pct": round(random.uniform(0.0, 100.0), 2),
        "aggregated_memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "aggregated_network_bytes": random.randint(0, 1024**3),
        "restart_count": random.randint(0, 3) + (1 if burst and random.random() < 0.05 else 0),
        "uptime_seconds": random.randint(0, 86400),
        "burst": burst,
    }


def generate_network_metrics():
    burst = random.random() < 0.06
    base_phase = _time_phase(period_seconds=500.0)
    interface_throughput = int(
        max(0, (100000 * (0.2 + 0.7 * abs(base_phase))) + random.gauss(0, 15000) + (150000 if burst else 0))
    )
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "interface_throughput_bps": interface_throughput,
        "network_rx_bytes": random.randint(0, 1024**3) + (interface_throughput * random.randint(0, 5)),
        "network_tx_bytes": random.randint(0, 1024**3),
        "packet_loss_pct": round(random.uniform(0.0, 2.0 if not burst else 15.0), 2),
        "rtt_ms": round(random.uniform(0.0, 100.0 if not burst else 800.0), 2),
        "jitter_ms": round(random.uniform(0.0, 50.0 if not burst else 300.0), 2),
        "active_flows": random.randint(0, 500 if burst else 120),
        "bgp_changes": random.randint(0, 10),
        "psu_efficiency_pct": round(random.uniform(70.0, 98.0), 2),
        "sensor_temp_c": round(random.uniform(18.0, 85.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2),
        "burst": burst,
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
    inst = str(instance_id)
    global _LAST_SEEN

    # container
    try:
        if container_metrics:
            host = str(container_metrics.get("host_id", "unknown"))
            cid = str(container_metrics.get("container_id", "unknown"))
            image = str(container_metrics.get("container_image", "unknown"))
            cpu_val = float(container_metrics.get("cpu_util_pct", 0.0))

            g_container_cpu.labels(
                instance=inst, host_id=host, container_id=cid, container_image=image
            ).set(cpu_val)

            # export zone-labeled cpu for Prometheus-side aggregation
            g_container_cpu_zone.labels(
                instance=inst, host_id=host, container_id=cid, zone=host_to_zone(host)
            ).set(cpu_val)

            g_container_memory_rss.labels(instance=inst, host_id=host, container_id=cid).set(
                float(container_metrics.get("memory_rss_bytes", 0))
            )
            g_container_network_rx.labels(instance=inst, host_id=host, container_id=cid).set(
                float(container_metrics.get("network_rx_bytes", 0))
            )
            g_container_uptime.labels(instance=inst, host_id=host, container_id=cid).set(
                float(container_metrics.get("uptime_seconds", 0))
            )

            # gpu + sensor temp - ensure metric always set (default 0)
            g_container_gpu_util_pct.labels(instance=inst, host_id=host, container_id=cid).set(
                float(container_metrics.get("gpu_util_pct", 0.0))
            )
            g_container_sensor_temp_c.labels(instance=inst, host_id=host, container_id=cid).set(
                float(container_metrics.get("sensor_temp_c", 0.0))
            )

            # throttle simulation -> counter (Prometheus counter)
            throttle_key = _key(inst, f"throttle::{host}::{cid}")
            prev_throttle = _LAST_SEEN.get(throttle_key, 0)
            throttle_events = 0

            # stronger throttle probability if CPU high or during burst
            burst_flag = bool(container_metrics.get("burst", False))
            prob = 0.02
            if cpu_val > 85.0:
                prob = 0.25
            if burst_flag:
                prob = max(prob, 0.35)

            # number of throttle events during this interval (could be >1 when burst)
            if random.random() < prob:
                throttle_events = random.randint(1, 1 if not burst_flag else 3)

            if throttle_events > 0:
                c_cpu_throttle_total.labels(instance=inst, host_id=host, container_id=cid).inc(throttle_events)
                _LAST_SEEN[throttle_key] = prev_throttle + throttle_events

            # pod restart delta -> counter
            rest_key = _key(inst, f"restarts::{host}::{cid}")
            prev = _LAST_SEEN.get(rest_key, 0)
            cur = int(container_metrics.get("restart_count", 0))
            if cur > prev:
                delta = cur - prev
                # if burst, sometimes simulate multiple pod restarts
                if container_metrics.get("burst", False) and random.random() < 0.1:
                    delta += random.randint(0, 2)
                c_pod_restarts_total.labels(instance=inst).inc(delta)
                _LAST_SEEN[rest_key] = cur

    except Exception:
        logger.exception("Failed to set container metrics")

    # VM
    try:
        if vm_metrics:
            host = str(vm_metrics.get("host_id", "unknown"))
            vm_cpu_val = float(vm_metrics.get("vm_cpu_pct", 0.0))
            g_vm_cpu.labels(instance=inst, host_id=host).set(vm_cpu_val)
            g_vm_cpu_zone.labels(instance=inst, host_id=host, zone=host_to_zone(host)).set(vm_cpu_val)
            g_vm_memory_rss.labels(instance=inst, host_id=host).set(
                float(vm_metrics.get("memory_rss_bytes", 0))
            )

            # keep latest VM CPU per host for Prometheus grouping if needed
            _LAST_SEEN.setdefault("zone_latest_cpu", {})[host] = vm_cpu_val

            # host power estimate (exposed in _LAST_SEEN for UI convenience only)
            _LAST_SEEN[f"host_power_w::{host}"] = float(vm_metrics.get("host_power_estimate_w", 0.0))
    except Exception:
        logger.exception("Failed to set VM metrics")

    # APP
    try:
        if app_metrics:
            host = str(app_metrics.get("host_id", "unknown"))
            cid = str(app_metrics.get("container_id", "unknown"))

            g_app_req_rate.labels(instance=inst, host_id=host, container_id=cid).set(
                float(app_metrics.get("request_rate_rps", 0.0))
            )
            g_app_error_rate.labels(instance=inst, host_id=host, container_id=cid).set(
                float(app_metrics.get("error_rate_pct", 0.0))
            )
            g_app_cpu.labels(instance=inst, host_id=host, container_id=cid).set(
                float(app_metrics.get("cpu_util_pct", 0.0))
            )

            # approximate counter increment from rps, amplify during bursts
            try:
                rps = float(app_metrics.get("request_rate_rps", 0.0))
            except Exception:
                rps = 0.0
            approx_delta = max(0.0, rps * float(collection_interval_s))
            if app_metrics.get("burst", False):
                approx_delta *= random.uniform(1.5, 5.0)
            if approx_delta > 0:
                try:
                    c_app_requests_total.labels(instance=inst, host_id=host, container_id=cid).inc(approx_delta)
                except Exception:
                    logger.exception("Failed to increment app requests counter")

            # record latency into histogram (Prometheus will compute quantiles)
            try:
                p50_ms = app_metrics.get("latency_p50_ms")
                p95_ms = app_metrics.get("latency_p95_ms")
                p99_ms = app_metrics.get("latency_p99_ms")
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

            # latency spike simulation -> counter (bigger counts in bursts)
            try:
                p95_val = float(app_metrics.get("latency_p95_ms", 0.0))
                spike_key = _key(inst, f"latency_spike::{host}::{cid}")
                prev_spikes = _LAST_SEEN.get(spike_key, 0)
                spike_inc = 0
                if p95_val > 500.0:
                    spike_inc = 1 if not app_metrics.get("burst", False) else random.randint(1, 4)
                if spike_inc > 0:
                    c_latency_spike_total.labels(instance=inst, host_id=host, container_id=cid).inc(spike_inc)
                    _LAST_SEEN[spike_key] = prev_spikes + spike_inc
            except Exception:
                logger.exception("Latency spike logic failed")

            # reflect app restarts
            rest_key_app = _key(inst, f"restarts_app::{host}::{cid}")
            prev_a = _LAST_SEEN.get(rest_key_app, 0)
            cura = int(app_metrics.get("restart_count", 0))
            if cura > prev_a:
                delta = cura - prev_a
                c_pod_restarts_total.labels(instance=inst).inc(delta)
                _LAST_SEEN[rest_key_app] = cura

    except Exception:
        logger.exception("Failed to set app metrics")

    # Orchestrator & network
    try:
        if orchestrator_metrics:
            g_orch_pod_count.labels(instance=inst).set(int(orchestrator_metrics.get("pod_count", 0)))
            g_orch_api_latency_ms.labels(instance=inst).set(float(orchestrator_metrics.get("cluster_api_latency_ms", 0.0)))

            # orchestrator restart_count influences pod restarts counter
            orch_restart_key = _key(inst, "orch_restarts")
            prev_orch = _LAST_SEEN.get(orch_restart_key, 0)
            cur_orch = int(orchestrator_metrics.get("restart_count", 0))
            if cur_orch > prev_orch:
                c_pod_restarts_total.labels(instance=inst).inc(cur_orch - prev_orch)
                _LAST_SEEN[orch_restart_key] = cur_orch

            # failover simulation -> counter (more likely during bursts)
            failed = int(orchestrator_metrics.get("pod_status_failed", 0))
            fail_key = _key(inst, "failed_pods")
            prev_failed = _LAST_SEEN.get(fail_key, 0)
            if failed > prev_failed and failed > 2:
                delta_fail = max(0, failed - prev_failed)
                # amplify during orchestrator burst
                if orchestrator_metrics.get("burst", False):
                    delta_fail += random.randint(0, 3)
                c_failover_events_total.labels(instance=inst).inc(delta_fail)
                _LAST_SEEN[fail_key] = failed
    except Exception:
        logger.exception("Failed to set orchestrator metrics")

    try:
        if network_metrics:
            host = str(network_metrics.get("host_id", "unknown"))
            g_net_throughput.labels(instance=inst, host_id=host).set(
                float(network_metrics.get("interface_throughput_bps", 0))
            )
            g_net_rtt.labels(instance=inst, host_id=host).set(float(network_metrics.get("rtt_ms", 0.0)))
            g_net_packet_loss.labels(instance=inst, host_id=host).set(
                float(network_metrics.get("packet_loss_pct", 0.0))
            )
            # also export sensor temp from network sensor as container sensor with a synthetic container id
            try:
                g_container_sensor_temp_c.labels(instance=inst, host_id=host, container_id="network_sensor").set(
                    float(network_metrics.get("sensor_temp_c", 0.0))
                )
            except Exception:
                # intentionally swallow errors for the auxiliary sensor export
                pass
    except Exception:
        logger.exception("Failed to set network metrics")


# ---------------------------
# Background fetcher for external APIs
# ---------------------------
def _start_background_fetch(interval_s=120):
    """
    Start a daemon thread that refreshes external API data into st.session_state.
    interval_s: how often to refresh (seconds). Default 120s.
    """
    if st.session_state.get("_bg_fetcher_started"):
        return
    st.session_state["_bg_fetcher_started"] = True

    # seed values (in case fetch fails immediately)
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
                # precipitation is heavy (30 days loop) — fetch less frequently in production
                st.session_state["precipitation_total"] = fetch_precipitation()
            except Exception as e:
                logger.exception("Background precipitation fetch failed: %s", e)
            try:
                st.session_state["air_quality_data"] = fetch_air_quality()
            except Exception as e:
                logger.exception("Background air quality fetch failed: %s", e)
            st.session_state["external_last_update"] = datetime.utcnow().isoformat() + "Z"
            time.sleep(interval_s)

    t = threading.Thread(target=_bg_loop, daemon=True)
    t.start()


# ---------------------------
# Streamlit UI + main loop (no local indices)
# ---------------------------
def main():
    # NOTE: st.set_page_config() already run.
    # Session state
    if "instance_id" not in st.session_state:
        st.session_state.instance_id = str(uuid.uuid4())[:8]
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

    # populate region-level gauges for Prometheus reference
    try:
        g_region_wue.labels(region="Manila").set(REGION_META["Manila"]["wue"])
        g_region_water_avail.labels(region="Manila").set(REGION_META["Manila"]["water_avail"])
        g_region_no2_emission_factor.labels(region="Manila").set(REGION_META["Manila"]["no2"])
    except Exception:
        pass

    # Start background fetcher (does nothing if already started)
    _start_background_fetch(interval_s=120)

    # style: narrow the sidebar using CSS (safe after set_page_config)
    st.markdown(
        """
    <style>
    /* Narrow the sidebar */
    section[data-testid="stSidebar"] > div[style] {
    min-width: 220px;
    max-width: 220px;
    }
    /* Improve compactness inside sidebar */
    section[data-testid="stSidebar"] .css-1d391kg, section[data-testid="stSidebar"] .css-1v3fvcr {
    padding: 6px 8px;
    font-size: 14px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # page header
    st.markdown(
        """
    <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
    <div>
    <div style='font-size:22px; font-weight:800; color:#e2e8f0;'>🛰️ Argus Environmental Telemetry — Revised</div>
    <div style='color:#94a3b8; margin-top:4px;'>Real-time simulated telemetry • Weather • Air Quality • System Metrics</div>
    </div>
    <div><div style='display:inline-block;padding:6px 10px;border-radius:12px;background:#10b981;color:white;font-weight:700;'>ACTIVE</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar content: Instance ID, Location, and countdown placeholder (next update)
    with st.sidebar:
        st.markdown("# 🛡️ Argus Agent (Revised)")
        st.markdown(f"**Instance ID:** {st.session_state.instance_id}")
        st.markdown("**Location:** Manila")
        st.markdown("---")
        st.markdown("Indices are computed in Prometheus (paste PromQL to Grafana).")
        # countdown placeholder in sidebar (updates every loop)
        countdown_sidebar = st.empty()
        st.markdown("---")
        show_raw = st.checkbox("Show global raw JSON", value=False)

    main_area = st.container()
    # left column (main cards) + right column (feed)
    left, right = main_area.columns([1.4, 1])

    container_placeholder = left.empty()
    vm_placeholder = left.empty()
    app_placeholder = left.empty()
    orchestrator_placeholder = left.empty()
    network_placeholder = left.empty()
    # region metadata in main body (left) — user asked to see it in the main body
    region_placeholder = left.empty()

    feed_box = right.empty()

    try:
        while True:
            now_mon = time.monotonic()
            elapsed = now_mon - st.session_state.last_emit
            remaining = max(0.0, INTERVAL - elapsed)
            remaining_ceil = math.ceil(remaining)

            # update the sidebar countdown every loop
            try:
                countdown_sidebar.markdown(
                    f"<div style='background:#047857;padding:8px;border-radius:8px;color:white;text-align:center;font-weight:700;'>Next update in <strong>{remaining_ceil}s</strong></div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                # ignore sidebar update errors
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

                entry = f"🕐 {datetime.now().strftime('%H:%M:%S')} | Update #{st.session_state.emit_seq} | Generated telemetry snapshot."
                st.session_state.agent_log.insert(0, entry)
                st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

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
                    st.session_state.agent_log.insert(0, f"📡 Prometheus metrics updated (emit #{st.session_state.emit_seq})")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
                except Exception as e:
                    logger.exception("Failed to update Prometheus metrics: %s", e)
                    st.session_state.agent_log.insert(0, f"❌ Failed to update Prometheus metrics: {e}")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

            container_metrics = st.session_state.last_metrics["container"]
            vm_metrics = st.session_state.last_metrics["vm"]
            app_metrics = st.session_state.last_metrics["app"]
            orchestrator_metrics = st.session_state.last_metrics["orchestrator"]
            network_metrics = st.session_state.last_metrics["network"]

            # Container card
            with container_placeholder.container():
                st.markdown("<div style='margin-bottom:8px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>🐳 Container Infrastructure</h3>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:#94a3b8;font-size:13px'>{container_metrics['container_image']} • {container_metrics['container_id']} • Host: {container_metrics['host_id']}</div>",
                    unsafe_allow_html=True,
                )
                k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
                k1.metric("CPU Usage", f"{container_metrics['cpu_util_pct']:.1f}%")
                k2.metric("Memory", f"{human_bytes(container_metrics['memory_rss_bytes'])}")
                k3.metric("Network RX", f"{human_bytes(container_metrics['network_rx_bytes'])}")
                k4.metric("Uptime", f"{human_seconds(container_metrics['uptime_seconds'])}")
                # JSON expander restored (updates every interval)
                with st.expander("View raw JSON - container", expanded=False):
                    st.json(container_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # VM card
            with vm_placeholder.container():
                st.markdown("<div style='margin-top:6px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#0f172a,#0c4a6e);color:#e2e8f0;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin:0;'>🖥️ Virtual Machine</h3>", unsafe_allow_html=True)
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
                st.markdown("<h3 style='margin:0;'>📦 Application</h3>", unsafe_allow_html=True)
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
                st.markdown("<h3 style='margin:0;'>🗂️ Orchestrator</h3>", unsafe_allow_html=True)
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
                st.markdown("<h3 style='margin:0;'>🌐 Network</h3>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#94a3b8;font-size:13px'>Host: {network_metrics['host_id']}</div>", unsafe_allow_html=True)
                n1, n2, n3 = st.columns([1, 1, 1])
                n1.metric("Throughput bps", f"{network_metrics['interface_throughput_bps']}")
                n2.metric("RTT ms", f"{network_metrics['rtt_ms']}")
                n3.metric("Packet Loss %", f"{network_metrics['packet_loss_pct']}%")
                with st.expander("View raw JSON - network", expanded=False):
                    st.json(network_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # --- External data card (weather / precipitation / air quality) ---
            ext_weather = st.session_state.get("weather_data")
            ext_precip = st.session_state.get("precipitation_total")
            ext_aq = st.session_state.get("air_quality_data")
            ext_last = st.session_state.get("external_last_update")

            region_placeholder.markdown("<div style='margin-top:8px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#083047,#0b3f56);color:#e2e8f0;'>", unsafe_allow_html=True)
            region_placeholder.markdown("<h3 style='margin:0 0 6px 0;'>☁️ External Data</h3>", unsafe_allow_html=True)

            # Weather summary (defensive parsing)
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

            # Precipitation summary
            prec_text = f"{ext_precip} mm (30d total)" if ext_precip is not None else "Not available"
            region_placeholder.markdown(f"<div style='margin-top:6px;font-weight:700'>Precipitation:</div> {prec_text}", unsafe_allow_html=True)

            # Air quality summary (defensive)
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

            # Expanders with raw JSON so you can inspect the data
            with region_placeholder.container():
                with st.expander("Show raw weather JSON", expanded=False):
                    st.json(ext_weather)
                with st.expander("Show raw precipitation data", expanded=False):
                    st.json(ext_precip)
                with st.expander("Show raw air quality JSON", expanded=False):
                    st.json(ext_aq)

            # --- Region metadata card (kept below external data) ---
            region_placeholder.markdown("<div style='margin-top:8px;padding:12px;border-radius:12px;background:linear-gradient(135deg,#07203a,#0a3850);color:#e2e8f0;'>", unsafe_allow_html=True)
            region_placeholder.markdown("<h3 style='margin:0;'>🌍 Region metadata & Prometheus indices</h3>", unsafe_allow_html=True)
            region_placeholder.markdown(f"**Region:** Manila")
            region_placeholder.markdown(f"WUE (region_wue): {REGION_META['Manila']['wue']}")
            region_placeholder.markdown(f"Water availability (region_water_avail): {REGION_META['Manila']['water_avail']}")
            region_placeholder.markdown(f"NO₂ factor (region_no2_emission_factor): {REGION_META['Manila']['no2']}")
            region_placeholder.markdown("<div style='margin-top:6px;color:#94a3b8;'>Indices computed in Prometheus — paste PromQL into Grafana/Prometheus.</div>", unsafe_allow_html=True)
            region_placeholder.markdown("</div>", unsafe_allow_html=True)

            # Right column: only the feed (emit updates)
            feed_box.markdown(
                f"<div style='padding:8px;'><div style='background:linear-gradient(135deg,#021026,#081520);padding:12px;border-radius:8px;color:#e2e8f0;min-height:200px;max-height:680px;overflow-y:auto;'><pre style='white-space:pre-wrap;margin:0;font-size:13px'>{'<br>'.join(st.session_state.agent_log)}</pre></div></div>",
                unsafe_allow_html=True,
            )

            # optionally show global raw JSON in sidebar
            if show_raw:
                st.sidebar.markdown("### 📋 Latest Raw Telemetry")
                st.sidebar.json(
                    {
                        "container": container_metrics,
                        "vm": vm_metrics,
                        "app": app_metrics,
                        "orchestrator": orchestrator_metrics,
                        "network": network_metrics,
                    }
                )

            time.sleep(SLEEP_STEP)

    except Exception as e:
        err_entry = f"❌ [{datetime.now().isoformat()}] ERROR | {str(e)}"
        st.session_state.agent_log.insert(0, err_entry)
        st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
        feed_box.markdown("<pre style='white-space:pre-wrap'>" + "\n".join(st.session_state.agent_log) + "</pre>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
