"""Read-only Linux hardware sample. Executable on a configured remote Linux host through SSH."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import io
import json
import math
import os
import platform
import subprocess
import time


def empty_source(source_id, label, operating_system, training_host):
    return {
        'id': source_id, 'label': label, 'hostname': None, 'os': operating_system,
        'isTrainingHost': training_host, 'status': 'warming', 'sampledAt': None,
        'ageSeconds': None,
        'cpu': {'name': None, 'logicalCores': None, 'utilizationPercent': None,
                'powerWatts': None, 'powerReason': None},
        'memory': {'usedBytes': None, 'totalBytes': None, 'utilizationPercent': None},
        'gpus': [], 'notes': [],
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def cpu_times(text):
    """Linux guest times are already included in user/nice; do not count twice."""
    row = next(line for line in text.splitlines() if line.startswith('cpu '))
    values = [int(value) for value in row.split()[1:]]
    if len(values) < 4:
        raise ValueError('Incomplete aggregate CPU counters.')
    return sum(values[:8]), values[3] + (values[4] if len(values) > 4 else 0)


def cpu_percent(before, after):
    total, idle = after[0] - before[0], after[1] - before[1]
    if total <= 0 or idle < 0:
        return None
    return max(0.0, min(100.0, 100.0 * (total - idle) / total))


def memory_info(text):
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(rest.split()[0]) * 1024  # Linux kB here means KiB.
    total, available = values.get('MemTotal'), values.get('MemAvailable')
    if not total or available is None:
        return {'usedBytes': None, 'totalBytes': total, 'utilizationPercent': None}
    used = max(0, min(total, total - available))
    return {'usedBytes': used, 'totalBytes': total, 'utilizationPercent': 100.0 * used / total}


def number(value, lower=0, upper=None):
    try:
        value = float(value.strip())
    except (ValueError, TypeError, AttributeError):
        return None
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        return None
    return value


def gpu_rows(text):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 8:
            continue
        try:
            index = int(row[0].strip())
        except ValueError:
            continue
        used, total = number(row[3]), number(row[4])
        result.append({
            'index': index, 'name': row[1].strip(),
            'utilizationPercent': number(row[2], upper=100),
            'memoryUsedBytes': None if used is None else int(used * 1024 ** 2),
            'memoryTotalBytes': None if total is None else int(total * 1024 ** 2),
            'powerWatts': number(row[5]), 'powerLimitWatts': number(row[6]),
            'temperatureC': number(row[7], lower=-100, upper=250),
        })
    return result


def package_energy(root=Path('/sys/class/powercap')):
    """Read package zones only: never sum core/DRAM/psys subdomains."""
    found = {}
    try:
        zones = list(Path(root).iterdir())
    except OSError:
        return found
    for zone in zones:
        try:
            name = (zone / 'name').read_text().strip()
            if not name.startswith('package-'):
                continue
            key = str(zone.resolve())
            found[key] = (int((zone / 'energy_uj').read_text()),
                          int((zone / 'max_energy_range_uj').read_text()))
        except (OSError, ValueError):
            continue
    return found


def package_watts(before, after, elapsed):
    if not before or set(before) != set(after) or elapsed <= 0:
        return None
    energy = 0
    for key, (start, limit) in before.items():
        end, end_limit = after[key]
        if limit <= 0 or limit != end_limit or not (0 <= start <= limit and 0 <= end <= limit):
            return None
        energy += end - start if end >= start else limit - start + end
    return energy / 1_000_000 / elapsed


def sample_linux(interval=0.3):
    sample = empty_source('home', 'Linux · remote host', 'Linux', False)
    sample['hostname'] = platform.node()
    sample['cpu']['logicalCores'] = os.cpu_count()
    first = cpu_times(Path('/proc/stat').read_text())
    energy_before = package_energy()
    started = time.monotonic()
    time.sleep(interval)
    second = cpu_times(Path('/proc/stat').read_text())
    energy_after = package_energy()
    elapsed = time.monotonic() - started
    sample['cpu']['utilizationPercent'] = cpu_percent(first, second)
    sample['cpu']['powerWatts'] = package_watts(energy_before, energy_after, elapsed)
    sample['cpu']['powerReason'] = (
        'CPU package energy counters are unavailable or not readable by this SSH user.'
        if sample['cpu']['powerWatts'] is None else None)
    try:
        rows = Path('/proc/cpuinfo').read_text().splitlines()
        sample['cpu']['name'] = next((row.split(':', 1)[1].strip() for row in rows
                                      if row.startswith('model name')), None)
    except OSError:
        pass
    sample['memory'] = memory_info(Path('/proc/meminfo').read_text())
    try:
        command = ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu',
                   '--format=csv,noheader,nounits']
        response = subprocess.run(command, capture_output=True, text=True, timeout=3)
        if response.returncode != 0:
            raise RuntimeError('NVIDIA telemetry command failed.')
        sample['gpus'] = gpu_rows(response.stdout)
        if not sample['gpus']:
            sample['notes'].append('No NVIDIA GPU measurements are available.')
        elif any(any(gpu[key] is None for key in ('utilizationPercent', 'powerWatts', 'temperatureC')) for gpu in sample['gpus']):
            sample['notes'].append('Some GPU sensors returned unavailable readings.')
    except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError):
        sample['notes'].append('NVIDIA GPU telemetry is unavailable; no device usage or power is assumed.')
    sample['notes'].append('Whole-host/device activity includes all workloads; these readings do not isolate the demo process.')
    sample['status'], sample['sampledAt'], sample['ageSeconds'] = 'ok', utc_now(), 0.0
    return sample


if __name__ == '__main__':
    print(json.dumps(sample_linux(), allow_nan=False))
