const WEATHER_ICONS = { sunny: "☀️ Sunny", rainy: "🌧️ Rainy", snow: "❄️ Snow", night: "🌙 Night" };
const LINE_LABELS = { none: "No line", short: "Short line", long: "Long line" };

function setCard(id, cls, verdict, detail) {
  const card = document.getElementById(`card-${id}`);
  card.className = `card ${cls}`;
  document.getElementById(`${id}-verdict`).textContent = verdict;
  document.getElementById(`${id}-detail`).textContent = detail;
}

// "since when" — walk history backwards while the value is unchanged
function sinceLabel(history, key) {
  if (!history.length) return "";
  const current = history[history.length - 1][key];
  let start = history[history.length - 1];
  for (let i = history.length - 2; i >= 0; i--) {
    if (history[i][key] !== current) break;
    start = history[i];
  }
  const t = new Date(start.ts);
  return `since ${t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
}

function renderTimeline(id, history, classFor) {
  const ticks = document.querySelector(`#timeline-${id} .tl-ticks`);
  ticks.innerHTML = "";
  for (const row of history) {
    const s = document.createElement("span");
    s.className = classFor(row);
    s.title = new Date(row.ts).toLocaleTimeString();
    ticks.appendChild(s);
  }
}

async function refresh() {
  let data;
  try {
    const res = await fetch("/api/status");
    data = await res.json();
  } catch {
    return; // server briefly unreachable; try again next tick
  }

  document.getElementById("stale-badge").classList.toggle("hidden", !data.stale);

  const player = document.getElementById("player");
  const src = `https://www.youtube.com/embed/${data.video_id}?autoplay=1&mute=1`;
  if (player.src !== src) player.src = src;

  const latest = data.latest;
  if (!latest) return;
  const history = data.history;

  setCard(
    "gondola",
    latest.gondola_moving ? "good" : "idle",
    latest.gondola_moving ? "Running" : "Stopped",
    `motion ${latest.motion_score.toFixed(4)} · ${sinceLabel(history, "gondola_moving")}`
  );
  setCard(
    "line",
    { none: "good", short: "warn", long: "bad" }[latest.line_status],
    `${latest.person_count} ${latest.person_count === 1 ? "person" : "people"}`,
    `${LINE_LABELS[latest.line_status].toLowerCase()} · ${sinceLabel(history, "line_status")}`
  );

  const today = data.today;
  const peakNote = today.peak_ts
    ? `peak ${today.peak} at ${new Date(today.peak_ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`
    : "no one seen in line yet";
  setCard(
    "line-today",
    "idle",
    `${today.total_sightings}`,
    `sightings since midnight · ${peakNote}`
  );
  setCard(
    "weather",
    "idle",
    WEATHER_ICONS[latest.weather] ?? latest.weather,
    `brightness ${latest.brightness} · ${sinceLabel(history, "weather")}`
  );

  renderTimeline("gondola", history, r => (r.gondola_moving ? "t-moving" : "t-still"));
  renderTimeline("line", history, r => `t-${r.line_status}`);
  renderTimeline("weather", history, r => `t-${r.weather}`);

  const updated = new Date(latest.ts).toLocaleTimeString();
  document.getElementById("summary").textContent =
    `${history.length} observations · updated ${updated} · new reading every ${data.interval_seconds}s`;
}

refresh();
setInterval(refresh, 5000);
