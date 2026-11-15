// Leaf photo -> POST /api/score (src/maize/serve.py) -> disease, severity and what to do.
import { BAND_COLOR, BANDS, DISEASE_COLOR, DISEASES, Disease, HEALTHY, bandFor } from "./advice.js";
import { Lang, get, initialLang, loadLang, t } from "./i18n.js";

interface ScoreResult {
  label: string; // "Healthy" or a Disease
  confidence: number;
  confident: boolean;
  score: number; // 0 = like a healthy leaf, 1 = as damaged as the worst diseased leaves
  model_version: string | null;
}

type View = "intro" | "loading" | "error" | "unsure" | "result";

const MAX_SIDE = 1024; // shrink on the phone: ~150 KB to upload instead of several MB
const TIMEOUT_MS = 90_000; // a sleeping free server can take about a minute to wake
const WAKE_HINT_MS = 5000;

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const frame = $("frame");
const photo = $<HTMLImageElement>("photo");
const ring = document.getElementById("ring") as unknown as SVGCircleElement;
const views: Record<View, HTMLElement> = {
  intro: $("v-intro"), loading: $("v-loading"), error: $("v-error"), unsure: $("v-unsure"), result: $("v-result"),
};

let upload: Blob | null = null;
let last: ScoreResult | null = null;
let errorKey = "error.generic";
let guideDisease: Disease = "MLND";

function el<K extends keyof HTMLElementTagNameMap>(tag: K, text = "", className = ""): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.textContent = text; // textContent only, so nothing is ever parsed as HTML
  node.className = className;
  return node;
}

function show(view: View): void {
  for (const [name, node] of Object.entries(views)) node.hidden = name !== view;
}

function progress(step: number): void {
  $("progress").querySelectorAll("li").forEach((li, i) => {
    li.className = i < step ? "done" : i === step ? "active" : "";
  });
}

function fail(key: string): void {
  errorKey = key;
  $("error-text").textContent = t(key);
  show("error");
}

// Decode on the device (also handles iPhone HEIC and rotation), shrink, re-encode as JPEG.
async function shrink(file: File): Promise<Blob> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = el("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise<Blob | null>((done) => canvas.toBlob(done, "image/jpeg", 0.9));
  if (!blob) throw new Error("encode failed");
  return blob;
}

async function score(blob: Blob): Promise<ScoreResult> {
  const body = new FormData();
  body.append("file", blob, "leaf.jpg");
  const res = await fetch("/api/score", { method: "POST", body, signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as ScoreResult;
}

function errorKeyFor(err: unknown): string {
  if (err instanceof DOMException && err.name === "TimeoutError") return "error.timeout";
  return err instanceof TypeError ? "error.offline" : "error.generic";
}

async function check(file: File): Promise<void> {
  show("loading");
  progress(0);
  try {
    upload = await shrink(file);
  } catch {
    fail("error.open");
    return;
  }
  URL.revokeObjectURL(photo.src);
  photo.src = URL.createObjectURL(upload);
  photo.hidden = false;
  $("shutter").hidden = true;
  frame.classList.add("has-photo");
  await analyse(upload);
}

async function analyse(blob: Blob): Promise<void> {
  show("loading");
  progress(1);
  frame.classList.add("scanning");
  const wake = setTimeout(() => ($("wake").hidden = false), WAKE_HINT_MS);
  try {
    last = await score(blob);
    progress(2);
    render(last, true);
  } catch (err) {
    fail(errorKeyFor(err));
  } finally {
    clearTimeout(wake);
    $("wake").hidden = true;
    frame.classList.remove("scanning");
  }
}

function render(r: ScoreResult, animate: boolean): void {
  if (!r.confident) {
    show("unsure");
    return;
  }
  const healthy = r.label === HEALTHY;
  const band = bandFor(r.label, r.score);
  const pct = healthy ? 0 : Math.round(r.score * 100);
  const name = healthy ? t("bands.healthy.label") : t(`diseases.${r.label}.name`);

  views.result.style.setProperty("--accent", BAND_COLOR[band]);
  $("chip").textContent = t(`bands.${band}.label`);
  $("headline").textContent = healthy ? t("result.healthy_title") : t("result.detected", { disease: name });
  $("confidence").textContent = t("result.confidence", { pct: Math.round(r.confidence * 100) });
  $("meaning").textContent = t(`bands.${band}.meaning`);
  $("action").textContent = t(`bands.${band}.action`);
  $("steps").replaceChildren(...get<string[]>(`bands.${band}.steps`).map((s) => el("li", s)));
  $("scale").replaceChildren(...BANDS.map((key) => {
    const seg = el("div", t(`bands.${key}.label`), key === band ? "seg on" : "seg");
    seg.style.setProperty("--c", BAND_COLOR[key]);
    return seg;
  }));
  $("learn-title").textContent = healthy ? t("result.learn_healthy") : t("result.learn", { disease: name });
  $("learn-hint").textContent = healthy ? t("result.learn_healthy_hint") : t("result.learn_hint");
  $("model").textContent = t("result.model", { version: r.model_version ?? "-" });
  if (!healthy) selectDisease(r.label as Disease);
  show("result");
  if (!animate) return;
  animateRing(pct);
  if (matchMedia("(max-width: 860px)").matches) views.result.scrollIntoView({ behavior: "smooth" });
}

function animateRing(pct: number): void {
  ring.style.strokeDashoffset = "100";
  requestAnimationFrame(() => requestAnimationFrame(() => (ring.style.strokeDashoffset = String(100 - pct))));
  const start = performance.now();
  const tick = (now: number): void => {
    const p = Math.min(1, (now - start) / 1100);
    $("pct").textContent = String(Math.round(pct * (1 - (1 - p) ** 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function fact(term: string, body: HTMLElement, className = ""): HTMLDivElement {
  const row = el("div", "", className);
  row.append(el("dt", term), body);
  return row;
}

function selectDisease(key: Disease): void {
  guideDisease = key;
  $("tabs").querySelectorAll<HTMLButtonElement>("button").forEach((tab) => {
    tab.setAttribute("aria-pressed", String(tab.dataset.key === key));
  });
  const steps = el("ul");
  steps.append(...get<string[]>(`diseases.${key}.steps`).map((s) => el("li", s)));
  const what = el("dd");
  what.append(steps);
  const facts = el("dl", "", "facts");
  facts.append(
    fact(t("guide.signs"), el("dd", t(`diseases.${key}.signs`))),
    fact(t("guide.cause"), el("dd", t(`diseases.${key}.cause`))),
    fact(t("guide.what"), what, "wide"),
    fact(t("guide.risk"), el("dd", t(`diseases.${key}.risk`)), "wide"),
  );
  const panel = $("disease");
  panel.style.setProperty("--accent", DISEASE_COLOR[key]);
  panel.replaceChildren(el("h3", t(`diseases.${key}.name`)), el("p", t(`diseases.${key}.kind`), "kind"), facts);
}

function buildTabs(): void {
  $("tabs").replaceChildren(...DISEASES.map((key) => {
    const tab = el("button", t(`diseases.${key}.short`), "tab");
    tab.type = "button";
    tab.dataset.key = key;
    tab.style.setProperty("--tab", DISEASE_COLOR[key]);
    tab.addEventListener("click", () => selectDisease(key));
    return tab;
  }));
  selectDisease(guideDisease);
}

async function setLang(lang: Lang): Promise<void> {
  await loadLang(lang);
  document.querySelectorAll<HTMLButtonElement>("[data-lang]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.lang === lang));
  });
  photo.alt = t("scan.take");
  $("error-text").textContent = t(errorKey);
  buildTabs();
  if (last) render(last, false);
}

async function share(): Promise<void> {
  if (!last) return;
  const band = bandFor(last.label, last.score);
  const text = t("result.share_text", {
    result: last.label === HEALTHY ? t("result.healthy_title") : t(`diseases.${last.label}.name`),
    band: t(`bands.${band}.label`),
    pct: last.label === HEALTHY ? 0 : Math.round(last.score * 100),
    action: t(`bands.${band}.action`),
  });
  if (navigator.share) {
    await navigator.share({ title: t("app.name"), text }).catch(() => undefined);
    return;
  }
  await navigator.clipboard.writeText(text);
  $("share").textContent = t("result.copied");
  setTimeout(() => ($("share").textContent = t("result.share")), 1500);
}

function reset(): void {
  last = null;
  photo.hidden = true;
  $("shutter").hidden = false;
  frame.classList.remove("has-photo");
  show("intro");
  $("analyse").scrollIntoView({ behavior: "smooth" });
}

for (const id of ["camera", "gallery"]) {
  const input = $<HTMLInputElement>(id);
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    input.value = ""; // allow picking the same photo again
    if (file) void check(file);
  });
}
frame.addEventListener("dragover", (e) => {
  e.preventDefault();
  frame.classList.add("over");
});
frame.addEventListener("dragleave", () => frame.classList.remove("over"));
frame.addEventListener("drop", (e) => {
  e.preventDefault();
  frame.classList.remove("over");
  const file = e.dataTransfer?.files[0];
  if (file) void check(file);
});
document.querySelectorAll<HTMLButtonElement>("[data-lang]").forEach((b) => {
  b.addEventListener("click", () => void setLang(b.dataset.lang as Lang));
});
$("retry").addEventListener("click", () => void (upload ? analyse(upload) : reset()));
$("share").addEventListener("click", () => void share());
$("again").addEventListener("click", reset);

show("intro");
void setLang(initialLang());
