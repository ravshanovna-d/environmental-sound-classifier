// Audio demo: decode -> resample to 22.05 kHz (OfflineAudioContext) -> log-mel (mel.js) -> ONNX ResNet18.
const $ = (id) => document.getElementById(id);
let meta, melCfg, session, busy = false;

// --- audio -------------------------------------------------------------------------------
async function decodeToTarget(arrayBuffer) {
  // Decode at 44.1 kHz (the ESC-50 rate), then render one mono channel at the training rate.
  const decoder = new OfflineAudioContext(1, 1, 44100);
  const decoded = await decoder.decodeAudioData(arrayBuffer);
  const len = Math.max(1, Math.ceil(decoded.duration * melCfg.sr));
  const ctx = new OfflineAudioContext(1, len, melCfg.sr);   // downmixes to mono and resamples
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0);
}

// --- spectrogram drawing (magma-like colour map) -----------------------------------------
const MAGMA = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
  [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
function colour(v) {               // v in [0, 1]
  const x = Math.min(1, Math.max(0, v)) * (MAGMA.length - 1), i = Math.min(MAGMA.length - 2, Math.floor(x)), f = x - i;
  return MAGMA[i].map((c, k) => c + (MAGMA[i + 1][k] - c) * f);
}
function drawSpec(mel) {
  const c = $("spec"), nM = melCfg.n_mels, nT = melCfg.n_frames;
  c.width = nT; c.height = nM;
  const ctx = c.getContext("2d"), img = ctx.createImageData(nT, nM);
  for (let m = 0; m < nM; m++) {
    for (let t = 0; t < nT; t++) {
      const v = (mel[m * nT + t] + 2) / 4;                   // normalised range is [-2, 2]
      const [r, g, b] = colour(v), p = ((nM - 1 - m) * nT + t) * 4;   // low frequencies at the bottom
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// --- inference ---------------------------------------------------------------------------
async function classify(arrayBuffer, label) {
  if (!session || busy) return;
  busy = true; setStatus("Analysing…");
  try {
    const raw = await decodeToTarget(arrayBuffer);
    const y = LogMel.fitLength(raw, melCfg);
    const mel = LogMel.logMel(y, melCfg);
    drawSpec(mel);
    const out = await session.run({ mel: new ort.Tensor("float32", mel, [1, 1, melCfg.n_mels, melCfg.n_frames]) });
    showResult(Array.from(out.probs.data), label);
    setStatus("");
  } catch (e) {
    console.error(e);
    setStatus("Could not read this audio file.");
  } finally { busy = false; }
}

function showResult(probs, label) {
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const truth = label ? meta.classes.indexOf(label) : -1;
  const top = order[0][1];
  const verdict = truth < 0 ? "" : top === truth
    ? `<p class="muted" style="color:var(--good)">✓ Correct — this clip is labelled “${meta.class_names[truth]}”.</p>`
    : `<p class="muted" style="color:var(--bad)">✗ This clip is labelled “${meta.class_names[truth]}”.</p>`;
  $("result").innerHTML = `<div class="muted">Top-3 predictions</div><div class="top3">` +
    order.map(([p, i], k) => `<div class="${k === 0 ? "first" : ""}">
      <div class="lbl"><span>${meta.class_names[i]}</span><span>${(p * 100).toFixed(1)}%</span></div>
      <div class="bar"><span style="width:${(p * 100).toFixed(1)}%;background:${k === 0 ? "var(--accent)" : "var(--muted)"}"></span></div>
    </div>`).join("") + `</div>${verdict}
    <p class="muted">The model only knows these 10 classes: ${meta.class_names.join(", ")}. Any other sound is mapped to the closest one.</p>`;
}

function setStatus(s) { $("status").textContent = s; }

// --- inputs ------------------------------------------------------------------------------
function renderExamples() {
  $("examples").innerHTML = meta.examples.map((e, i) => `<button class="btn" type="button" data-i="${i}">${e.name}</button>`).join("");
  $("examples").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    const e = meta.examples[+b.dataset.i];
    $("examples").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    $("player").src = e.file;
    $("player").play().catch(() => {});
    const buf = await (await fetch(e.file)).arrayBuffer();
    classify(buf, e.label);
  }));
}

$("file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  $("examples").querySelectorAll("button").forEach((x) => x.classList.remove("active"));
  $("player").src = URL.createObjectURL(f);
  classify(await f.arrayBuffer(), null);
  ev.target.value = "";
});

$("record").addEventListener("click", async () => {
  if (!navigator.mediaDevices || !window.MediaRecorder) { setStatus("Recording is not supported in this browser."); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch { setStatus("Microphone permission denied."); return; }
  const rec = new MediaRecorder(stream), chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    $("record").disabled = false;
    const blob = new Blob(chunks, { type: rec.mimeType });
    $("player").src = URL.createObjectURL(blob);
    $("examples").querySelectorAll("button").forEach((x) => x.classList.remove("active"));
    classify(await blob.arrayBuffer(), null);
  };
  $("record").disabled = true;
  rec.start();
  let left = 5;
  setStatus(`Recording… ${left}`);
  const timer = setInterval(() => { left -= 1; if (left > 0) setStatus(`Recording… ${left}`); }, 1000);
  setTimeout(() => { clearInterval(timer); rec.stop(); }, 5000);
});

// --- page content from meta.json ---------------------------------------------------------
const pct = (v) => (v * 100).toFixed(1) + "%";
function renderPage() {
  $("lead").textContent = meta.subtitle;
  $("chips").innerHTML = meta.tags.map((t) => `<span class="chip">${t}</span>`).join("");
  $("links").innerHTML = [
    [meta.repo_url, "GitHub repo"], [meta.notebook_url, "Training notebook"], [meta.dataset.url, "Dataset"],
  ].map(([u, t]) => `<a class="btn" href="${u}" target="_blank" rel="noopener">${t}</a>`).join("");

  const m = meta.test_metrics;
  $("metrics").innerHTML = [
    [pct(m["Accuracy"]), "Test accuracy"], [pct(m["Top-3 accuracy"]), "Top-3 accuracy"],
    [m["Macro F1"].toFixed(3), "Macro F1"], [meta.web_model.size_mb + " MB", "Model in browser (int8)"],
    [meta.parity.max_abs_diff_db.toExponential(1) + " dB", "Max JS vs librosa difference"],
  ].map(([v, k]) => `<div class="metric"><b>${v}</b><span>${k}</span></div>`).join("");

  $("compare").innerHTML = `<thead><tr><th>Strategy</th><th>Trainable params</th><th>Learning rate</th><th>Val acc</th><th>Test acc</th></tr></thead><tbody>` +
    meta.comparison.map((r) => `<tr class="${r.strategy === meta.best_strategy ? "best" : ""}"><td>${r.strategy}</td>
      <td>${(r.trainable_params / 1e6).toFixed(r.trainable_params < 1e5 ? 3 : 1)}M</td><td>${r.lr}</td>
      <td>${pct(r.best_val_acc)}</td><td>${pct(r.test_acc)}</td></tr>`).join("") + "</tbody>";

  $("figs").innerHTML = meta.figures.map((f) => `<figure style="margin:0"><img src="${f.src}" alt="${f.caption}" loading="lazy"><figcaption class="muted">${f.caption}</figcaption></figure>`).join("");
  $("steps").innerHTML = meta.steps.map((s) => `<li>${s}</li>`).join("");
  $("footer").innerHTML = `${meta.footer} · Audio: <a href="${meta.dataset.url}" target="_blank" rel="noopener">ESC-50</a> by K. J. Piczak, ` +
    `clips from <a href="https://freesound.org" target="_blank" rel="noopener">freesound.org</a>, ${meta.dataset.license} (non-commercial).`;
}

(async function init() {
  [meta, melCfg] = await Promise.all([fetch("meta.json").then((r) => r.json()), fetch("mel_filters.json").then((r) => r.json())]);
  renderPage();
  renderExamples();
  try {
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    setStatus("Model ready.");
  } catch (e) {
    console.error(e);
    setStatus("Could not load the model.");
  }
})();
