// Independent, read-only hardware telemetry. Never invokes a training action.
import {sessionFetch,telemetryUrl,hosted} from './presenter-session.js';
const panel = document.getElementById('panel-convergence');
const monitor = document.getElementById('hardware-monitor');
const sourcesNode = document.getElementById('telemetry-sources');
const sourceNodes = {home: document.getElementById('telemetry-home'), local: document.getElementById('telemetry-local')};
const statusNode = document.getElementById('telemetry-status');
const modeNode = document.getElementById('telemetry-mode-note');
const refreshButton = document.getElementById('telemetry-refresh');
const recordedButton = document.getElementById('mode-recorded');
const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const percent = value => number(value) && value <= 100 ? `${value.toFixed(1)}%` : 'Unavailable';
const watts = value => number(value) ? `${value.toFixed(1)} W` : 'Unavailable';
const gib = value => number(value) ? (value / 1073741824).toFixed(1) : null;
const memory = (used, total) => number(used) && number(total) && total > 0
  ? `${gib(used)} / ${gib(total)} GiB` : 'Unavailable';
const make = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const sourceDefaults = [
  {id: 'home', label: 'Remote GPU training host', isTrainingHost: true},
  ...(!hosted?[{id: 'local', label: 'Windows · local gateway', isTrainingHost: false}]:[]),
];
let timer = null;
let request = null;
let lastSnapshot = null;
let receivedAt = 0;
let failures = 0;
const endpoint=telemetryUrl;
if(hosted)sourceNodes.local?.closest('details')?.setAttribute('hidden','');

function active() {
  return !panel.hidden && !document.hidden;
}

function metric(label, value, detail) {
  const item = make('div', 'hardware-metric');
  item.append(make('dt', null, label), make('dd', value === 'Unavailable' ? 'unavailable' : '', value));
  if (detail) item.append(make('small', null, detail));
  return item;
}

function sourceAge(source) {
  if (!number(source?.ageSeconds)) return null;
  return Math.max(0, source.ageSeconds + (performance.now() - receivedAt) / 1000);
}

function sourceCard(source, fallback, connectionFailed, detailsOpen) {
  const age = sourceAge(source);
  const hasSample = source && Number.isFinite(Date.parse(source.sampledAt));
  const supportedStatus = ['ok', 'warming', 'stale', 'unavailable'].includes(source?.status);
  let state = supportedStatus ? source.status : 'warming';
  if (connectionFailed) state = hasSample ? 'stale' : 'unavailable';
  else if (state === 'ok' && (!hasSample || age === null)) state = 'unavailable';
  else if (state === 'ok' && age > 15) state = 'stale';
  const hasValues = hasSample && ['ok', 'stale'].includes(state);
  const card = make('section', `telemetry-source telemetry-${state}`);
  card.dataset.source = fallback.id;
  const heading = make('div', 'telemetry-source-heading');
  const trainingHost = typeof source?.isTrainingHost === 'boolean' ? source.isTrainingHost : fallback.isTrainingHost;
  const roleLabel = fallback.id === 'home'
    ? trainingHost ? 'Remote GPU training host' : 'Remote host · other workloads'
    : trainingHost ? 'Windows · CPU training host' : 'Windows · local gateway';
  heading.append(make('h4', null, roleLabel),
    make('span', `telemetry-badge telemetry-badge-${state}`,
      ({ok: 'Live', warming: 'Waiting', stale: 'Stale sample', unavailable: 'Unavailable'})[state]));
  const time = hasSample ? new Date(source.sampledAt) : null;
  const stamp = make('p', 'telemetry-timestamp', time
    ? `Sampled ${time.toLocaleTimeString()}${age === null ? '' : ` · ${Math.floor(age)} s ago`}`
    : 'No hardware sample received yet.');
  if (time) stamp.title = `Measurement time: ${source.sampledAt}`;
  card.append(heading, stamp);
  const cpu = hasValues ? source.cpu || {} : {};
  const ram = hasValues ? source.memory || {} : {};
  const metrics = make('dl', 'hardware-metrics host-metrics');
  metrics.append(metric('CPU use', percent(cpu.utilizationPercent)),
    metric('RAM used / total', memory(ram.usedBytes, ram.totalBytes), number(ram.utilizationPercent) ? percent(ram.utilizationPercent) : null));
  if (number(cpu.powerWatts)) metrics.append(metric('CPU package power', watts(cpu.powerWatts)));
  card.append(metrics);
  const gpus = hasValues && Array.isArray(source.gpus) ? source.gpus : [];
  for (const gpu of gpus) {
    const device = make('div', 'telemetry-gpu');
    device.append(make('h5', null, 'NVDIA GPU'));
    const gpuMetrics = make('dl', 'hardware-metrics gpu-metrics');
    gpuMetrics.append(metric('GPU use', percent(gpu.utilizationPercent)),
      metric('VRAM used / total', memory(gpu.memoryUsedBytes, gpu.memoryTotalBytes)),
      metric('GPU power draw', watts(gpu.powerWatts), `Limit ${watts(gpu.powerLimitWatts)}`),
      metric('Temperature', number(gpu.temperatureC) ? `${gpu.temperatureC.toFixed(0)} °C` : 'Unavailable'));
    device.append(gpuMetrics);
    card.append(device);
  }
  if (!gpus.length) card.append(make('p', 'telemetry-device-note', fallback.id === 'local' && !trainingHost
    ? 'Local gateway only; model training runs on home.'
    : hasValues ? 'No GPU readings are available in this sample.' : 'GPU measurements are not available yet.'));
  const details = make('details', 'telemetry-details');
  details.dataset.sourceDetails = fallback.id;
  details.open = detailsOpen;
  details.append(make('summary', null, 'Measurement details'));
  if (source?.hostname) details.append(make('p', null, `Host: ${source.hostname} · ${source.os || fallback.id}`));
  if (cpu.name || number(cpu.logicalCores)) details.append(make('p', null,
    [cpu.name, number(cpu.logicalCores) ? `${cpu.logicalCores} logical CPUs` : null].filter(Boolean).join(' · ')));
  for (const note of Array.isArray(source?.notes) ? source.notes : []) {
    if (typeof note !== 'string' || !note) continue;
    let publicNote = note;
    for (const gpu of Array.isArray(source?.gpus) ? source.gpus : []) {
      if (typeof gpu.name !== 'string' || !gpu.name.trim()) continue;
      const aliases = [gpu.name, gpu.name.replace(/^NVIDIA\s+/i, ''), gpu.name.replace(/^NVIDIA\s+GeForce\s+/i, '')];
      for (const alias of aliases) if (alias.trim()) publicNote = publicNote.replaceAll(alias, 'NVDIA GPU');
    }
    publicNote = publicNote.replaceAll('NVIDIA GPU', 'NVDIA GPU');
    details.append(make('p', null, publicNote));
  }
  if (state === 'stale') details.append(make('p', null, 'These are the last measured values. They are not current readings.'));
  card.append(details);
  return card;
}

function render(connectionFailed = false) {
  const opened = new Set([...sourcesNode.querySelectorAll('details[data-source-details][open]')].map(node => node.dataset.sourceDetails));
  const sources = Array.isArray(lastSnapshot?.sources) ? lastSnapshot.sources : [];
  // Keep the native local-host disclosure itself mounted so refresh never closes it.
  for (const fallback of sourceDefaults) sourceNodes[fallback.id].replaceChildren(sourceCard(
    sources.find(source => source.id === fallback.id), fallback, connectionFailed, opened.has(fallback.id)));
  const isRecorded = recordedButton.getAttribute('aria-pressed') === 'true';
  modeNode.textContent = isRecorded
    ? 'Current hardware activity. These are not historical measurements from the recorded run.'
    : 'Whole-host activity. The Linux GPU is shared with Qwen and other workloads; these readings do not isolate this training run.';
}

function schedule(delay) {
  if (timer !== null) clearTimeout(timer);
  timer = active() ? setTimeout(() => { timer = null; refresh(); }, delay) : null;
}

async function refresh() {
  if (request || !active()) return;
  if (timer !== null) { clearTimeout(timer); timer = null; }
  if (!endpoint) {
    statusNode.textContent = 'Open this demo through its local launcher to see hardware measurements.';
    render(true);
    return;
  }
  const current = new AbortController();
  const startedAt = performance.now();
  request = current;
  refreshButton.disabled = true;
  const timeout = setTimeout(() => current.abort('timeout'), 4000);
  try {
    const response = await sessionFetch(endpoint, {cache: 'no-store', credentials: hosted?'same-origin':'omit', signal: current.signal});
    if (!response.ok) throw new Error('Telemetry response unavailable.');
    const snapshot = await response.json();
    if (snapshot?.schemaVersion !== 1 || !Array.isArray(snapshot.sources)) throw new Error('Unsupported telemetry response.');
    lastSnapshot = snapshot;
    receivedAt = performance.now();
    failures = 0;
    render();
    statusNode.textContent = 'Refreshes every second while this panel is visible.';
  } catch (error) {
    if (current.signal.reason !== 'hidden') {
      failures++;
      render(true);
      statusNode.textContent = lastSnapshot
        ? 'Monitor connection interrupted. Last samples are marked stale; training can continue.'
        : hosted?'Hardware monitor unavailable. Use Refresh to check the home connection; training can continue.':'Hardware monitor unavailable. Start the demo launcher to reconnect; training can continue.';
    }
  } finally {
    clearTimeout(timeout);
    if (request === current) request = null;
    refreshButton.disabled = false;
    schedule(failures ? Math.min(30000, 1000 * 2 ** Math.min(failures, 5)) : Math.max(0, 1000 - (performance.now() - startedAt)));
  }
}

function visibilityChanged() {
  if (timer !== null) { clearTimeout(timer); timer = null; }
  if (!active()) { request?.abort('hidden'); return; }
  render(failures > 0);
  refresh();
}

refreshButton.addEventListener('click', refresh);
new MutationObserver(visibilityChanged).observe(panel, {attributes: true, attributeFilter: ['hidden']});
new MutationObserver(() => render(failures > 0)).observe(recordedButton, {attributes: true, attributeFilter: ['aria-pressed']});
document.addEventListener('visibilitychange', visibilityChanged);
window.addEventListener('pageshow', visibilityChanged);
window.addEventListener('pagehide', () => { if (timer !== null) clearTimeout(timer); request?.abort('hidden'); });
render();
visibilityChanged();
