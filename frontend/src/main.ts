// Leaf photo -> POST /api/score (src/maize/serve.py) -> damage level and what to do.
import { BANDS, Band, DISEASES, DiseaseKey, bandFor } from "./advice.js";

interface ScoreResult {
  score: number; // 0 = like a healthy leaf, 1 = as far from healthy as the worst diseased leaves
  within_healthy_range: boolean;
}

type View = "intro" | "loading" | "error" | "result";

const MAX_SIDE = 1024; // shrink on the phone: ~150 KB to upload instead of several MB (the model uses 128 px)
const TIMEOUT_MS = 90_000; // a sleeping free server can take about a minute to wake
const WAKE_HINT_MS = 5000;
const BAND_ORDER: Band[] = ["healthy", "mild", "moderate", "severe"];

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;
const frame = $("frame");
const photo = $<HTMLImageElement>("photo");
const shutter = $("shutter");
const ring = document.getElementById("ring") as unknown as SVGCircleElement;
const views: Record<View, HTMLElement> = {
  intro: $("v-intro"), loading: $("v-loading"), error: $("v-error"), result: $("v-result"),
};

let upload: Blob | null = null;
let shareText = "";

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

function fail(message: string): void {
  $("error-text").textContent = message;
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
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail;
    throw new Error(detail || `Server error (${res.status}). Please try again.`);
  }
  return (await res.json()) as ScoreResult;
}

function explain(err: unknown): string {
  if (err instanceof DOMException && err.name === "TimeoutError") {
    return "The server took too long to answer. Check your connection and try again.";
  }
  if (err instanceof TypeError) return "Could not reach the server. Check your internet and try again.";
  return err instanceof Error ? err.message : "Something went wrong. Please try again.";
}

async function check(file: File): Promise<void> {
  show("loading");
  progress(0);
  try {
    upload = await shrink(file);
  } catch {
    fail("Could not open that photo. Please use a JPEG or PNG image.");
    return;
  }
  URL.revokeObjectURL(photo.src);
  photo.src = URL.createObjectURL(upload);
  photo.hidden = false;
  shutter.hidden = true;
  frame.classList.add("has-photo");
  await analyse(upload);
}

async function analyse(blob: Blob): Promise<void> {
  show("loading");
  progress(1);
  frame.classList.add("scanning");
  const wake = setTimeout(() => ($("wake").hidden = false), WAKE_HINT_MS);
  try {
    const result = await score(blob);
    progress(2);
    render(result);
  } catch (err) {
    fail(explain(err));
  } finally {
    clearTimeout(wake);
    $("wake").hidden = true;
    frame.classList.remove("scanning");
  }
}

function render(r: ScoreResult): void {
  const band = bandFor(r.score, r.within_healthy_range);
  const b = BANDS[band];
  const pct = Math.round(r.score * 100);

  $("verdict").parentElement?.style.setProperty("--accent", b.color);
  $("chip").textContent = b.label;
  $("headline").textContent = b.title;
  $("meaning").textContent = b.meaning;
  $("action").textContent = b.action;
  $("steps").replaceChildren(...b.steps.map((s) => el("li", s)));
  $("scale").replaceChildren(...BAND_ORDER.map((key) => {
    const seg = el("div", BANDS[key].label, key === band ? "seg on" : "seg");
    seg.style.setProperty("--c", BANDS[key].color);
    return seg;
  }));
  shareText = `Maize Leaf Check: ${b.label} damage (${pct}%). Recommended action: ${b.action}.`;
  selectDisease(null);
  show("result");
  animateRing(pct);
  if (matchMedia("(max-width: 860px)").matches) views.result.scrollIntoView({ behavior: "smooth" });
}

function animateRing(pct: number): void {
  ring.style.strokeDashoffset = "100";
  requestAnimationFrame(() => requestAnimationFrame(() => (ring.style.strokeDashoffset = String(100 - pct))));
  const start = performance.now();
  const tick = (now: number): void => {
    const t = Math.min(1, (now - start) / 1100);
    $("pct").textContent = String(Math.round(pct * (1 - (1 - t) ** 3)));
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function fact(term: string, body: HTMLElement): HTMLDivElement {
  const row = el("div");
  row.append(el("dt", term), body);
  return row;
}

function selectDisease(key: DiseaseKey | null): void {
  $("tabs").querySelectorAll<HTMLButtonElement>("button").forEach((tab) => {
    tab.setAttribute("aria-pressed", String(tab.dataset.key === key));
  });
  const panel = $("disease");
  panel.hidden = key === null;
  if (key === null) return;

  const d = DISEASES[key];
  const steps = el("dd");
  const list = el("ul");
  list.append(...d.steps.map((s) => el("li", s)));
  steps.append(list);
  const source = el("dd");
  const link = el("a", d.source.name);
  Object.assign(link, { href: d.source.url, target: "_blank", rel: "noopener noreferrer" });
  source.append(link);
  const facts = el("dl", "", "facts");
  facts.append(
    fact("Signs to look for", el("dd", d.signs)),
    fact("Cause", el("dd", d.cause)),
    fact("What to do", steps),
    fact("Yield risk", el("dd", d.risk)),
    fact("Source", source),
  );
  panel.style.setProperty("--accent", d.color);
  panel.replaceChildren(el("h3", d.name), el("p", d.kind, "kind"), facts);
}

function buildTabs(): void {
  const keys = Object.keys(DISEASES) as DiseaseKey[];
  $("tabs").replaceChildren(...keys.map((key) => {
    const tab = el("button", DISEASES[key].short, "tab");
    tab.type = "button";
    tab.dataset.key = key;
    tab.style.setProperty("--tab", DISEASES[key].color);
    tab.addEventListener("click", () => selectDisease(key));
    return tab;
  }));
}

async function share(): Promise<void> {
  if (navigator.share) {
    await navigator.share({ title: "Maize Leaf Check", text: shareText }).catch(() => undefined);
    return;
  }
  await navigator.clipboard.writeText(shareText);
  $("share").textContent = "Copied";
  setTimeout(() => ($("share").textContent = "Share"), 1500);
}

function reset(): void {
  photo.hidden = true;
  shutter.hidden = false;
  frame.classList.remove("has-photo");
  show("intro");
  window.scrollTo({ top: 0, behavior: "smooth" });
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
$("retry").addEventListener("click", () => void (upload ? analyse(upload) : reset()));
$("share").addEventListener("click", () => void share());
$("again").addEventListener("click", reset);

buildTabs();
show("intro");
