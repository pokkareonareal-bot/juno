"use strict";
// Juno Studio front end. No framework, no build step, no network beyond this
// server. Everything shown comes from the server's status object and its
// event stream; every string from speech goes in through textContent.

const TOKEN = document.querySelector('meta[name="studio-token"]').content;
const $ = (sel) => document.querySelector(sel);

let S = null;                 // latest status from the server
let models = [];
let sessions = [];
let recDeadline = null;       // local clock for the recording countdown
let recSeconds = null;

// -- plumbing -----------------------------------------------------------------

async function api(path, body) {
  const opts = { headers: { "X-Studio-Token": TOKEN } };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  let data;
  try { data = await res.json(); } catch { throw new Error(`${res.status} ${res.statusText}`); }
  if (!data.ok) throw new Error(data.error || "failed");
  return data.result;
}

async function act(fn, okMessage) {
  try {
    const out = await fn();
    if (okMessage) toast(typeof okMessage === "function" ? okMessage(out) : okMessage);
    return out;
  } catch (err) {
    toast(err.message, true);
    return undefined;
  }
}

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    // The page's CSP forbids style attributes; the CSSOM is allowed.
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

let toastTimer = null;
function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (error ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), error ? 6000 : 3500);
}

const pct = (x, d = 0) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)}%`);
const LABELS = {
  assistant_directed: "For Juno",
  human_directed: "To a person",
  background_or_media: "Media / noise",
};
const PROMPT_WHO = {
  assistant_directed: "Talk to Juno",
  human_directed: "Talk to each other",
  background_or_media: "Media only, nobody speaks",
};
const COLORS = {
  assistant_directed: "var(--act)",
  human_directed: "var(--human)",
  background_or_media: "var(--media)",
};

function seconds(n) {
  n = Math.round(n);
  if (n < 60) return `${n} s`;
  const h_ = Math.floor(n / 3600), m = Math.floor((n % 3600) / 60), s = n % 60;
  return [h_ ? `${h_} h` : "", m ? `${m} min` : "", s ? `${s} s` : ""].filter(Boolean).join(" ");
}
function slotText(slots) {
  return Object.entries(slots || {})
    .map(([k, v]) => {
      const val = v && typeof v === "object" ? v.value : v;
      return k === "duration" && typeof val === "number" ? `${k} = ${seconds(val)}` : `${k} = ${val}`;
    })
    .join(", ");
}

// -- events ---------------------------------------------------------------------

function connect() {
  const es = new EventSource("/api/events?t=" + encodeURIComponent(TOKEN));
  es.addEventListener("status", (e) => { S = JSON.parse(e.data); render(); });
  es.addEventListener("level", (e) => setMeter(JSON.parse(e.data).db));
  es.addEventListener("utterance", (e) => addUtterance(JSON.parse(e.data)));
  es.addEventListener("segment", (e) => {
    const d = JSON.parse(e.data);
    if (d.dropped) toast(`Skipped one utterance: ${d.dropped}`);
  });
  es.addEventListener("job", (e) => renderJob(JSON.parse(e.data)));
  es.addEventListener("problem", (e) => toast(JSON.parse(e.data).message, true));
  es.onerror = () => {
    $("#chips").replaceChildren(h("span", { class: "chip" }, h("span", { class: "dot warn" }),
      h("span", { class: "v", text: "reconnecting to the studio…" })));
  };
}

function setMeter(db) {
  // -60 dBFS -> empty, -10 dBFS -> full
  const w = Math.max(0, Math.min(100, ((db + 60) / 50) * 100));
  for (const id of ["#try-meter"]) { const m = $(id); if (m) m.style.width = `${w}%`; }
  const pm = document.querySelector(".prompt .meter > div");
  if (pm) pm.style.width = `${w}%`;
}

// -- the whole page, from S ---------------------------------------------------------

function render() {
  if (!S) return;
  renderChips();
  renderLoading();
  renderTry();
  renderCollect();
  if (!$("#tab-results").hidden) renderResultsControls();
}

function renderChips() {
  const live = S.listening;
  const loadState = (k) => S.loading[k] || "waiting";
  const teacherDot = loadState("teacher") === "ready" ? "ok" : loadState("teacher").startsWith("failed") ? "warn" : "";
  $("#chips").replaceChildren(
    h("span", { class: "chip", title: S.mic },
      h("span", { class: "dot" + (live ? " live" : "") }),
      h("span", { class: "k", text: live ? (S.purpose === "collect" ? "Recording" : "Listening") : "Mic" }),
      h("span", { class: "v", text: S.mic })),
    h("span", { class: "chip", title: S.model ? S.model.encoder : "no model loaded" },
      h("span", { class: "dot" + (S.model ? " ok" : "") }),
      h("span", { class: "k", text: "Model" }),
      h("span", { class: "v", text: S.model ? S.model.name : "none" })),
    h("span", { class: "chip", title: "The cascade: speech-to-text + the intent engine" },
      h("span", { class: "dot " + teacherDot }),
      h("span", { class: "k", text: "Cascade" }),
      h("span", { class: "v", text: S.teacher })),
  );
}

function renderLoading() {
  const box = $("#loading");
  const items = Object.entries(S.loading).filter(([, v]) => v !== "ready");
  if (!items.length && S.ready) { box.hidden = true; return; }
  const loading = items.filter(([, v]) => v === "loading").map(([k]) => k);
  const failed = items.filter(([, v]) => v.startsWith("failed"));
  box.replaceChildren(
    loading.length ? h("span", {}, h("span", { class: "spin" }),
      `Loading ${loading.join(", ")}… the first time can take a minute.`) : null,
    ...failed.map(([k, v]) => h("div", { text: `${k}: ${v}` })),
    !loading.length && !failed.length && !S.ready ? h("span", {}, h("span", { class: "spin" }), "Starting…") : null,
  );
  box.hidden = false;
}

// -- TRY ------------------------------------------------------------------------------

function renderTry() {
  const btn = $("#try-toggle");
  const tryLive = S.listening && S.purpose === "try";
  btn.textContent = tryLive ? "Stop listening" : "Start listening";
  btn.classList.toggle("live", tryLive);
  btn.disabled = !S.ready || (!!S.session && !tryLive);
  $("#try-hint").textContent = S.session
    ? "A collection session is open in the Collect tab; finish or abandon it to try things here."
    : S.simulated
      ? "This studio is playing WAV files instead of a microphone (started with --simulate)."
      : "Talk normally: to Juno, to someone else, or let the TV play. Nothing is saved unless you label utterances and save them.";

  $("#st-awaiting").checked = !!S.state.awaiting_answer;
  $("#st-timer").checked = !!S.state.timer_running;

  const st = S.stats;
  $("#try-stats").replaceChildren(
    stat(st.utterances, "utterances"),
    stat(pct(st.stt_avoided), "handled without STT"),
    stat(st.agreement === null ? "–" : pct(st.agreement), "agree with the cascade"),
    stat(st.reflex_ms_p50 === null ? "–" : `${st.reflex_ms_p50} ms`, "Reflex, median"),
  );
  $("#label-count").textContent = S.labelled;
  $("#save-labels").disabled = !S.labelled;

  const sel = $("#model-select");
  const current = S.model ? S.model.path : "";
  if (sel.dataset.for !== JSON.stringify(models.map((m) => m.path)) || sel.value !== current) {
    sel.replaceChildren(
      h("option", { value: "", text: models.length ? "Choose a model…" : "No models in models/ yet" }),
      ...models.map((m) => h("option", { value: m.path, text: m.name, selected: m.path === current })));
    sel.dataset.for = JSON.stringify(models.map((m) => m.path));
    sel.value = current;
  }
  const m = S.model;
  $("#model-meta").textContent = m
    ? `${m.targets || "?"} labels · ${shortEnc(m.encoder)} · held-out: ${pct(m.holdout.stt_avoided)} handled without STT, ` +
      `${pct(m.holdout.false_activation, 1)} false activations` + (m.distributable ? "" : " · not distributable")
    : "Without a model, only the cascade runs: you'll see transcripts but no Reflex decisions.";
}

const stat = (n, l) => h("div", { class: "stat" }, h("div", { class: "n", text: n }), h("div", { class: "l", text: l }));
const shortEnc = (e) => (e || "").replace("mlx-community/", "").replace(":stats", "");

const AGREE = {
  agree: ["good", "✓ Agrees with the cascade"],
  reflex_would_miss: ["bad", "Reflex dropped something the cascade accepted"],
  reflex_would_act_on_ignored: ["bad", "Reflex acted on something the cascade ignored"],
  different_intent: ["bad", "Different answer from the cascade"],
  escalated: ["meh", "Escalated, so the cascade decided"],
  no_model: ["meh", "No model: the cascade only"],
  no_teacher: ["meh", "Cascade still loading"],
};

function addUtterance(u) {
  $("#feed-empty")?.remove();
  const feed = $("#feed");
  const card = u.dropped
    ? h("article", { class: "card utt" }, h("div", { class: "dropped", text: `Dropped (${u.duration} s): ${u.dropped}` }))
    : utteranceCard(u);
  feed.prepend(card);
  while (feed.children.length > 100) feed.lastChild.remove();
}

function utteranceCard(u) {
  const reflex = u.reflex;
  const cascade = u.cascade;
  const route = reflex ? reflex.route : "none";
  const intent = reflex && reflex.intent ? reflex.intent : null;
  const headText = !reflex ? "no model" : route === "ignore" ? `not for Juno` : intent ? intent.value : "";
  const head = h("div", { class: "utt-head" },
    h("span", { class: `r ${route}`, text: route === "none" ? "—" : route.toUpperCase() }),
    h("span", { class: "intent", text: headText }),
    reflex && route !== "ignore" && intent ? h("span", { class: "p", text: pct(intent.p) }) : null,
    reflex && route === "ignore" ? h("span", { class: "p", text: `${pct(1 - (reflex.addressed.probs || {}).assistant_directed)} sure` }) : null,
    reflex && Object.keys(reflex.slots || {}).length ? h("span", { class: "p", text: slotText(reflex.slots) }) : null,
    h("span", { class: "meta", text: [reflex ? `${reflex.ms.toFixed(0)} ms` : null, u.at, `${u.duration} s`].filter(Boolean).join(" · ") }),
  );

  const body = h("div", { class: "utt-body" });
  if (reflex) {
    const probs = reflex.addressed.probs || {};
    body.append(
      h("div", { class: "who-bar", title: "who it was for, according to Reflex" },
        ...Object.keys(LABELS).map((k) => h("span", { style: `width:${(100 * (probs[k] || 0)).toFixed(1)}%;background:${COLORS[k]}` }))),
      h("div", { class: "who-legend" },
        ...Object.keys(LABELS).map((k) => h("span", {}, h("i", { style: `background:${COLORS[k]}` }), `${LABELS[k]} ${pct(probs[k])}`))),
    );
    if (route !== "ignore" && reflex.intent_top) {
      body.append(h("div", { class: "alts", text: "top intents: " + reflex.intent_top.map(([n, p]) => `${n} ${p.toFixed(2)}`).join(" · ") }));
    }
    if (reflex.reason && route === "escalate") body.append(h("div", { class: "alts", text: `why escalate: ${reflex.reason}` }));
  }

  const [cls, text] = AGREE[u.agreement] || ["meh", u.agreement];
  const cascadeRow = h("div", { class: "cascade" },
    h("span", { class: "k", text: "Cascade heard" }),
    h("span", { class: "heard", text: cascade ? `“${cascade.transcript || "(nothing)"}”` : "…" }),
    cascade ? h("span", { class: "verdict", text: cascade.accepted
      ? `for Juno · ${cascade.intent || "?"}${Object.keys(cascade.slots || {}).length ? " · " + slotText(cascade.slots) : ""} · ${Math.round(cascade.stt_ms)} ms`
      : `not for Juno · ${Math.round(cascade.stt_ms)} ms` }) : null,
    h("span", { class: `agree ${cls}`, text }),
  );

  return h("article", { class: "card utt", "data-id": u.id }, head, reflex ? body : null, cascadeRow, labeler(u));
}

function labeler(u) {
  const box = h("div", { class: "labeler" });
  const guess = (u.cascade && u.cascade.intent) || (u.reflex && u.reflex.intent && u.reflex.intent.value) || "open_request";
  let choice = null;
  const intentSel = h("select", { "aria-label": "intent", hidden: true },
    ...(S ? S.intents : []).map((i) => h("option", { value: i, text: i, selected: i === guess })));
  const save = h("button", { class: "secondary", disabled: true, text: "Save label" });
  const seg = h("div", { class: "seg", role: "group", "aria-label": "who was it for" },
    ...Object.entries(LABELS).map(([k, label]) => h("button", {
      "aria-pressed": "false", text: label, onclick: (e) => {
        choice = k;
        for (const b of seg.children) b.setAttribute("aria-pressed", String(b === e.currentTarget));
        intentSel.hidden = k !== "assistant_directed";
        save.disabled = false;
      },
    })));
  save.addEventListener("click", async () => {
    const res = await act(() => api("/api/try/label", { id: u.id, addressed: choice, intent: choice === "assistant_directed" ? intentSel.value : null }));
    if (res) {
      box.querySelector(".done")?.remove();
      box.append(h("span", { class: "done", text: "Labelled ✓" }));
    }
  });
  box.append(h("span", { class: "q", text: "What was it really?" }), seg, intentSel, save);
  return box;
}

// -- COLLECT ----------------------------------------------------------------------------

function renderCollect() {
  const sess = S.session;
  $("#collect-setup").hidden = !!sess;
  $("#collect-live").hidden = !sess;
  if (!sess) {
    const box = $("#cs-encoders");
    const have = JSON.stringify(S.encoders);
    if (box.dataset.for !== have) {
      box.replaceChildren(...S.encoders.map((e) => h("label", {}, h("input", { type: "checkbox", value: e, checked: true }), shortEnc(e))));
      box.dataset.for = have;
    }
    $("#cs-start").disabled = !S.ready;
    return;
  }

  $("#sess-title").textContent = sess.id;
  $("#sess-meta").textContent = `${sess.room} · ${sess.speakers.join(", ")} · consent: ${sess.consent} · ${sess.mic}`;
  $("#sess-counts").replaceChildren(
    ...Object.keys(LABELS).map((k) => h("span", { class: "pill" }, h("i", { style: `background:${COLORS[k]}` }), `${LABELS[k]} ${sess.counts[k] || 0}`)));
  $("#sess-progress").style.width = `${(100 * sess.current) / Math.max(1, sess.steps.length - 1)}%`;
  $("#sess-finish").disabled = !sess.rows;

  renderPrompt(sess);
  renderFree(sess);
  renderHeard(sess);
}

function renderPrompt(sess) {
  const step = sess.steps[sess.current];
  const rec = S.recording;
  const recordingThis = rec && rec.step === step.index;
  const busyElsewhere = rec && !recordingThis;
  if (recordingThis && rec.remaining !== null) {
    recDeadline = performance.now() + rec.remaining * 1000;
    recSeconds = rec.seconds;
  } else if (!rec) {
    recDeadline = null;
  }
  const done = sess.per_step[step.index] || 0;
  const card = $("#prompt");
  card.replaceChildren(
    h("div", { class: "top-line" },
      h("span", { class: `who ${step.label}`, text: PROMPT_WHO[step.label] }),
      step.speaker ? h("span", { class: "speaker", text: step.speaker }) : null,
      h("span", { class: "step-n", text: `prompt ${step.index + 1} of ${sess.steps.length}` })),
    h("p", { class: "text", text: step.text }),
    h("p", { class: "intent-tag", text: step.label === "assistant_directed"
      ? `labelled as: ${step.intent || "whatever command the cascade hears"}`
      : `labelled as: ${LABELS[step.label].toLowerCase()}` }),
    recordingThis
      ? h("div", {},
          h("div", { class: "rec-progress" }, h("div", { id: "rec-bar" })),
          h("div", { class: "rec-row" },
            h("span", { class: "timer", id: "rec-timer", text: `${Math.ceil(rec.remaining || 0)} s` }),
            h("div", { class: "meter", style: "flex:1;margin:0" }, h("div", {})),
            h("button", { class: "danger", text: "■ Stop", onclick: () => act(() => api("/api/session/stop", {})) })))
      : h("div", { class: "rec-row", style: "margin-top:14px" },
          h("button", { class: "primary", disabled: busyElsewhere || !S.ready, text: `● Record ${Math.round(step.seconds)} s`,
            onclick: () => act(() => api("/api/session/record", { step: step.index })) }),
          done ? h("span", { class: "hint", style: "margin:0", text: `✓ ${done} utterance${done === 1 ? "" : "s"} recorded` }) : null,
          h("div", { class: "nav" },
            h("button", { class: "ghost", disabled: step.index === 0 || !!rec, text: "‹ Back",
              onclick: () => act(() => api("/api/session/goto", { step: step.index - 1 })) }),
            h("button", { class: "ghost", disabled: step.index >= sess.steps.length - 1 || !!rec, text: "Skip ›",
              onclick: () => act(() => api("/api/session/goto", { step: step.index + 1 })) }))),
  );
}

function tickRecording() {
  const timer = $("#rec-timer");
  const bar = $("#rec-bar");
  if (timer && recDeadline !== null) {
    const left = Math.max(0, (recDeadline - performance.now()) / 1000);
    timer.textContent = `${Math.ceil(left)} s`;
    if (bar && recSeconds) bar.style.width = `${100 * (1 - left / recSeconds)}%`;
  }
  requestAnimationFrame(tickRecording);
}

function renderFree(sess) {
  const lab = $("#fr-label"), intent = $("#fr-intent"), spk = $("#fr-speaker");
  if (lab.dataset.init !== sess.id) {
    lab.replaceChildren(...Object.entries(LABELS).map(([k, v]) => h("option", { value: k, text: v })));
    intent.replaceChildren(h("option", { value: "", text: "(cascade decides)" }),
      ...S.intents.map((i) => h("option", { value: i, text: i })));
    const speakers = [...sess.speakers];
    if (sess.speakers.length > 1) speakers.push(sess.speakers.join("+"));
    spk.replaceChildren(...speakers.map((s) => h("option", { value: s, text: s })), h("option", { value: "media", text: "media" }));
    lab.dataset.init = sess.id;
  }
  intent.disabled = lab.value !== "assistant_directed";
  const rec = S.recording;
  const freeLive = rec && rec.step === null;
  const btn = $("#fr-toggle");
  btn.textContent = freeLive ? `■ Stop (${Math.round(rec.elapsed)} s)` : "● Record";
  btn.className = freeLive ? "danger" : "secondary";
  btn.disabled = (!!rec && !freeLive) || !S.ready;
}

function renderHeard(sess) {
  const list = $("#heard-list");
  const segs = [...sess.segments].reverse();
  list.replaceChildren(...(segs.length ? segs.map((r) => {
    let warn = null;
    if (r.g_addressed === "assistant_directed" && r.t_accept === false) warn = "The cascade didn't think this was for Juno";
    if (r.g_addressed !== "assistant_directed" && r.t_accept) warn = "The cascade would have answered this";
    return h("li", {},
      h("span", { class: "dotc", style: `background:${COLORS[r.g_addressed]}` }),
      h("div", {},
        h("div", { class: "t", text: r.transcript ? `“${r.transcript}”` : "(nothing intelligible)" }),
        h("div", { class: "sub2", text: [LABELS[r.g_addressed], r.g_intent, r.speaker, `${r.duration} s`].filter(Boolean).join(" · ") }),
        warn ? h("div", { class: "sub2 warn", text: warn }) : null),
      h("button", { title: "discard this utterance", "aria-label": "discard", text: "×",
        onclick: () => act(() => api("/api/session/discard", { id: r.id })) }));
  }) : [h("li", { class: "hint", style: "display:block", text: "Nothing yet. Record a prompt." })]));
}

// -- RESULTS ------------------------------------------------------------------------------

async function loadLists() {
  const [m, s] = await Promise.all([api("/api/models"), api("/api/sessions")]).catch((e) => { toast(e.message, true); return [models, sessions]; });
  models = m; sessions = s;
  renderLists();
  render();
}

function renderLists() {
  const checked = new Set([...document.querySelectorAll("#session-list input:checked")].map((i) => i.value));
  $("#session-list").replaceChildren(...(sessions.length ? sessions.map((s) => h("label", { class: "pick" },
    h("input", { type: "checkbox", value: s.path, checked: checked.has(s.path), onchange: renderResultsControls }),
    h("div", {},
      h("div", { class: "name", text: s.name }),
      h("div", { class: "desc", text: `${s.rows} utterances · ${Object.entries(s.counts || {}).map(([k, v]) => `${LABELS[k] || k} ${v}`).join(", ")}` }),
      h("div", { class: "desc", text: `${s.kind || "session"} · ${s.room || "?"} · consent: ${s.consent || "?"} · ${(s.collected_at || "").replace("T", " ")}` })))) :
    [h("p", { class: "hint", text: "No sessions yet. Record one in the Collect tab, or save labels from Try it." })]));

  const chosen = document.querySelector("#model-list input:checked")?.value || (S && S.model ? S.model.path : models[0]?.path);
  $("#model-list").replaceChildren(...(models.length ? models.map((m) => h("label", { class: "pick" },
    h("input", { type: "radio", name: "ev-model", value: m.path, checked: m.path === chosen, onchange: renderResultsControls }),
    h("div", {},
      h("div", { class: "name", text: m.name }),
      m.error ? h("div", { class: "desc", text: m.error }) : h("div", { class: "desc" },
        `${m.targets || "?"} labels · `, h("code", { text: shortEnc(m.encoder) }),
        ` · held-out: ${pct(m.holdout.stt_avoided)} without STT, ${pct(m.holdout.false_activation, 1)} false activations`,
        m.distributable ? "" : " · not distributable")))) :
    [h("p", { class: "hint", text: "No models yet. Train one below, or with python -m juno_core.slu train." })]));
  renderResultsControls();
}

function renderResultsControls() {
  const picked = [...document.querySelectorAll("#session-list input:checked")].map((i) => i.value);
  const model = document.querySelector("#model-list input:checked")?.value;
  $("#ev-run").disabled = !picked.length || !model;
  const sel = $("#tr-encoder");
  const encs = new Set(models.map((m) => m.encoder).filter(Boolean));
  for (const s of sessions) if (picked.includes(s.path)) for (const e of s.encoders || []) encs.add(e);
  if (S) for (const e of S.encoders) encs.add(e);
  const list = [...encs].sort((a, b) => (b.includes("@L17:stats") - a.includes("@L17:stats")));
  if (sel.dataset.for !== JSON.stringify(list)) {
    const prev = sel.value;
    sel.replaceChildren(...list.map((e) => h("option", { value: e, text: shortEnc(e), selected: e === prev })));
    sel.dataset.for = JSON.stringify(list);
  }
  const running = S && S.job && S.job.state === "running";
  $("#tr-run").disabled = running || !sel.value;
}

function metric(n, label, extra, bad) {
  return h("div", { class: "metric" + (bad ? " bad" : "") }, h("div", { class: "n", text: n }),
    h("div", { class: "l", text: label }), extra ? h("div", { class: "ci", text: extra }) : null);
}
const rateText = (r) => (r && r.n ? `${r.k} of ${r.n} · up to ${pct(r.upper95, 1)}` : "no cases");

function renderReport(r, into) {
  const routes = r.routes || {};
  into.replaceChildren(
    h("div", { class: "metrics" },
      metric(pct(r.stt_avoided), "handled without STT", `${routes.act || 0} acted · ${routes.ignore || 0} ignored · ${routes.escalate || 0} escalated`),
      metric(pct(r.false_ignore.rate, 1), "false ignores", rateText(r.false_ignore), r.false_ignore.rate > 0.01),
      metric(pct(r.false_activation.rate, 1), "false activations", rateText(r.false_activation), r.false_activation.rate > 0.01),
      metric(pct(r.wrong_act.rate, 1), "wrong acts", rateText(r.wrong_act), r.wrong_act.rate > 0.01),
      metric(pct(r.addressed_accuracy, 1), "who-it-was-for accuracy", r.addressed_auc ? `AUC ${r.addressed_auc}` : ""),
      metric(pct(r.intent_accuracy, 1), "intent accuracy", r.typed_commands_acted ? `${pct(r.typed_commands_acted.rate)} of commands acted on` : ""),
    ),
    r.by_category ? h("div", {}, h("h4", { text: "By kind of utterance" }), h("table", { class: "t" },
      h("thead", {}, h("tr", {}, ...["kind", "n", "act", "ignore", "escalate", "who-for acc.", "wrong"].map((x) => h("th", { text: x })))),
      h("tbody", {}, ...Object.entries(r.by_category).map(([k, v]) => h("tr", {},
        h("td", { class: "mono", text: k }), h("td", { text: v.n }), h("td", { text: v.act }), h("td", { text: v.ignore }),
        h("td", { text: v.escalate }), h("td", { text: pct(v.addressed_accuracy) }), h("td", { text: v.wrong_decisions }))))) ) : null,
    r.intent_confusions && r.intent_confusions.length ? h("div", {}, h("h4", { text: "Most-confused intents" }), h("table", { class: "t" },
      h("thead", {}, h("tr", {}, h("th", { text: "said" }), h("th", { text: "heard as" }), h("th", { text: "n" }))),
      h("tbody", {}, ...r.intent_confusions.map((c) => h("tr", {}, h("td", { class: "mono", text: c.true }), h("td", { class: "mono", text: c.predicted }), h("td", { text: c.n })))))) : null,
    r.teacher_vs_gold ? h("p", { class: "hint", text: `For comparison, the cascade on these utterances: ${pct(r.teacher_vs_gold.addressed_binary_accuracy, 1)} right about who they were for, ${pct(r.teacher_vs_gold.missed.rate, 1)} of requests missed, ${pct(r.teacher_vs_gold.false_activation.rate, 1)} false activations.` }) : null,
  );
}

function renderJob(job) {
  if (!job) return;
  const log = $("#tr-log");
  log.hidden = false;
  log.textContent = job.log.join("\n") + (job.state === "running" ? "\n…" : "");
  log.scrollTop = log.scrollHeight;
  const out = $("#tr-out");
  if (job.state === "failed") {
    out.replaceChildren(h("p", { class: "hint", style: "color:var(--danger)", text: job.error }));
  } else if (job.state === "done" && job.report) {
    const box = h("div", {});
    renderReport({ ...job.report, by_category: null, intent_confusions: null }, box);
    out.replaceChildren(h("h4", { text: `Saved ${job.model}. Held-out results:` }), box,
      h("button", { class: "secondary", text: "Use this model in Try it", onclick: async () => {
        await act(() => api("/api/model", { path: job.model }), "Model loaded");
        loadLists();
      } }));
    loadLists();
  }
  if (S) { S.job = job; renderResultsControls(); }
}

// -- wiring -----------------------------------------------------------------------------------

function wire() {
  for (const tab of document.querySelectorAll(".tabs button")) {
    tab.addEventListener("click", () => {
      for (const t of document.querySelectorAll(".tabs button")) t.setAttribute("aria-selected", String(t === tab));
      for (const p of document.querySelectorAll(".tab")) p.hidden = p.id !== `tab-${tab.dataset.tab}`;
      if (tab.dataset.tab === "results") loadLists();
    });
  }

  $("#try-toggle").addEventListener("click", () => {
    const live = S && S.listening && S.purpose === "try";
    act(() => api(live ? "/api/listen/stop" : "/api/try/start", {}));
  });
  $("#model-select").addEventListener("change", (e) => {
    if (e.target.value) act(() => api("/api/model", { path: e.target.value }), (m) => `Loaded ${m.name}`);
  });
  $("#st-awaiting").addEventListener("change", (e) => act(() => api("/api/try/state", { awaiting_answer: e.target.checked })));
  $("#st-timer").addEventListener("change", (e) => act(() => api("/api/try/state", { timer_running: e.target.checked })));

  const dialog = $("#save-dialog");
  $("#save-labels").addEventListener("click", () => dialog.showModal());
  $("#save-form").addEventListener("submit", (e) => {
    if (e.submitter && e.submitter.value === "cancel") return;
    const consent = document.querySelector('input[name="sv-consent"]:checked').value;
    act(() => api("/api/try/save", { consent, room: $("#sv-room").value, speaker: $("#sv-speaker").value }),
      (r) => `Saved ${r.rows} labelled utterances to ${r.path}`);
    $("#sv-confirm").checked = false;
  });

  $("#cs-start").addEventListener("click", async () => {
    const encoders = [...document.querySelectorAll("#cs-encoders input:checked")].map((i) => i.value);
    const started = await act(() => api("/api/session/start", {
      speakers: $("#cs-speakers").value.split(","), room: $("#cs-room").value,
      consent: document.querySelector('input[name="cs-consent"]:checked').value,
      confirmed: $("#cs-confirm").checked, encoders,
    }));
    // Consent is per session: the next one has to be confirmed again.
    if (started) $("#cs-confirm").checked = false;
  });
  $("#sess-finish").addEventListener("click", () => act(() => api("/api/session/finish", {}),
    (r) => `Saved ${r.rows} utterances to ${r.path}`));
  $("#sess-abandon").addEventListener("click", () => {
    if (confirm("Abandon this session? Nothing recorded in it will be kept.")) act(() => api("/api/session/abandon", {}));
  });
  $("#fr-label").addEventListener("change", () => S && S.session && renderFree(S.session));
  $("#fr-toggle").addEventListener("click", () => {
    const freeLive = S && S.recording && S.recording.step === null;
    act(() => freeLive ? api("/api/session/stop", {}) : api("/api/session/free", {
      label: $("#fr-label").value, intent: $("#fr-label").value === "assistant_directed" ? $("#fr-intent").value : null,
      speaker: $("#fr-speaker").value,
    }));
  });

  $("#refresh-lists").addEventListener("click", loadLists);
  $("#ev-run").addEventListener("click", async () => {
    const picked = [...document.querySelectorAll("#session-list input:checked")].map((i) => i.value);
    const model = document.querySelector("#model-list input:checked")?.value;
    $("#ev-out").replaceChildren(h("p", { class: "hint", text: "Evaluating…" }));
    const res = await act(() => api("/api/evaluate", { model, sessions: picked, awaiting_answer: $("#ev-awaiting").checked }));
    if (res) renderReport(res.report, $("#ev-out"));
    else $("#ev-out").replaceChildren();
  });
  $("#tr-run").addEventListener("click", () => {
    const picked = [...document.querySelectorAll("#session-list input:checked")].map((i) => i.value);
    $("#tr-out").replaceChildren();
    act(() => api("/api/train", {
      name: $("#tr-name").value.trim(), encoder: $("#tr-encoder").value, targets: $("#tr-targets").value,
      sessions: picked, include_base: $("#tr-base").checked, allow_internal: $("#tr-internal").checked,
    }), "Training started");
  });
}

wire();
connect();
api("/api/models").then((m) => { models = m; render(); }).catch(() => {});
requestAnimationFrame(tickRecording);
