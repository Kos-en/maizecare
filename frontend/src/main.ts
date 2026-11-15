// Uploads a photo to POST /api/score (src/maize/serve.py) and shows the result.
interface ScoreResult {
  score: number; // 0 = healthy-like, 1 = as far from healthy as the worst diseased leaves
  within_healthy_range: boolean;
}

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const file = $<HTMLInputElement>("file");
const pick = $<HTMLLabelElement>("pick");
const preview = $<HTMLImageElement>("preview");
const message = $<HTMLDivElement>("message");
const result = $<HTMLElement>("result");

function showMessage(text: string): void {
  message.textContent = text; // textContent, so server text is never parsed as HTML
  message.className = "card warn";
  message.hidden = false;
}

function render(r: ScoreResult): void {
  const pct = Math.round(r.score * 100);
  $("headline").textContent = r.within_healthy_range ? "Looks healthy" : `Severity ${pct}%`;
  $("fill").style.width = `${pct}%`;
  $("detail").textContent = r.within_healthy_range
    ? "This leaf is within the range of healthy leaves."
    : `Score ${r.score.toFixed(2)} (0 = healthy-like, 1 = very far from healthy).`;
  result.hidden = false;
}

async function onChange(): Promise<void> {
  const f = file.files?.[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
  preview.hidden = false;
  message.hidden = true;
  result.hidden = true;
  pick.classList.add("busy");
  try {
    const body = new FormData();
    body.append("file", f);
    const res = await fetch("/api/score", { method: "POST", body });
    if (!res.ok) {
      const detail = (await res.json().catch(() => ({ detail: "" }))).detail;
      throw new Error(detail || `Server error (${res.status})`);
    }
    render((await res.json()) as ScoreResult);
  } catch (err) {
    showMessage(err instanceof Error ? err.message : "Something went wrong. Please try again.");
  } finally {
    pick.classList.remove("busy");
    file.value = ""; // allow re-selecting the same photo
  }
}

file.addEventListener("change", () => void onChange());
