const $ = (id) => document.getElementById(id);

const SAMPLE = [
  "Terrible quality, broke on day one",
  "It's okay, does the job but nothing special",
  "Absolutely love it, best purchase this year",
  "Decent for the price but the instructions were useless",
  "Stopped working after a month, support never responded",
  "Great value, battery lasts all week",
  "Packaging was damaged but the product itself is fine",
  "Five stars, my kids use it every day",
];

let toastTimer;
function toast(msg, ms = 5000) {
  const el = $("toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

async function checkModel() {
  const el = $("model-state");
  try {
    const cfg = await (await fetch("/api/config")).json();
    if (!cfg.inference.configured) {
      el.textContent = "model endpoint not configured";
      el.className = "model-state down";
    } else if (cfg.inference.ready) {
      el.textContent = `model ready — ${cfg.model_name}`;
      el.className = "model-state ready";
    } else {
      el.textContent = "model endpoint not responding";
      el.className = "model-state down";
    }
  } catch {
    el.textContent = "backend unreachable";
    el.className = "model-state down";
  }
}

function parseCsv(text) {
  // MVP parser: takes the longest field per line, handles simple quoting.
  return text
    .split(/\r?\n/)
    .map((line) => {
      const fields = line.match(/("([^"]|"")*"|[^,]+)/g) || [];
      const cleaned = fields.map((f) => f.replace(/^"|"$/g, "").replace(/""/g, '"').trim());
      cleaned.sort((a, b) => b.length - a.length);
      return cleaned[0] || "";
    })
    .filter((r) => r.length > 3);
}

function render(data) {
  $("empty").hidden = true;
  $("results").hidden = false;

  const s = data.summary;
  $("avg-stars").textContent = s.average_stars.toFixed(2);
  $("review-count").textContent = s.count;
  $("negative-share").textContent = Math.round(s.negative_share * 100) + "%";

  const max = Math.max(...Object.values(s.distribution), 1);
  $("dist").innerHTML = [5, 4, 3, 2, 1]
    .map((star) => {
      const n = s.distribution[String(star)];
      return `<div class="dist-row" data-stars="${star}">
        <div class="dist-label">${star} star${star > 1 ? "s" : ""}</div>
        <div class="dist-track"><div class="dist-fill" data-w="${(n / max) * 100}"></div></div>
        <div class="dist-count">${n}</div>
      </div>`;
    })
    .join("");
  requestAnimationFrame(() =>
    document.querySelectorAll(".dist-fill").forEach((el) => (el.style.width = el.dataset.w + "%"))
  );

  const themes = (list, elId) => {
    $(elId).innerHTML = list.length
      ? list.map((t) => `<li>${t.term}<em>${t.count}</em></li>`).join("")
      : '<li class="none">Nothing here yet</li>';
  };
  themes(s.negative_themes, "neg-themes");
  themes(s.positive_themes, "pos-themes");

  $("review-rows").innerHTML = data.results
    .map(
      (r) => `<tr>
        <td><span class="star-chip${r.stars <= 2 ? " low" : ""}">${r.stars}★</span></td>
        <td>${(r.confidence * 100).toFixed(0)}%</td>
        <td>${escapeHtml(r.review)}</td>
      </tr>`
    )
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function analyze() {
  const reviews = $("reviews-input")
    .value.split(/\r?\n/)
    .map((r) => r.trim())
    .filter(Boolean);
  if (!reviews.length) return toast("Paste at least one review first.");
  if (reviews.length > 500) return toast("Limit is 500 reviews per run — trim the list.");

  const btn = $("analyze-btn");
  btn.disabled = true;
  btn.textContent = "Scoring…";
  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviews }),
    });
    const body = await resp.json();
    if (!resp.ok) throw new Error(body.detail || `Request failed (${resp.status})`);
    render(body);
  } catch (e) {
    toast(e.message, 8000);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze reviews";
  }
}

$("analyze-btn").addEventListener("click", analyze);
$("sample-btn").addEventListener("click", () => {
  $("reviews-input").value = SAMPLE.join("\n");
});
$("csv-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const rows = parseCsv(await file.text());
  $("reviews-input").value = rows.slice(0, 500).join("\n");
  $("input-hint").textContent = `Loaded ${Math.min(rows.length, 500)} reviews from ${file.name}.`;
});

checkModel();
