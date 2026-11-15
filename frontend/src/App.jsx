import { useState } from "react";

const ACTION_LABEL = { monitor: "Monitor", treat: "Treat", remove: "Remove" };

export default function App() {
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function onFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setBusy(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch("/api/predict", { method: "POST", body });
      if (!res.ok) throw new Error(res.status === 503 ? "The models are not loaded on the server yet." : `Server error (${res.status})`);
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <h1>Maize Leaf Check</h1>
      <p className="sub">Take or upload a photo of one maize leaf. The first request may take about a minute while the server wakes up.</p>

      <label className="button">
        {busy ? "Analysing…" : "Take or choose a photo"}
        <input type="file" accept="image/*" capture="environment" onChange={onFile} disabled={busy} hidden />
      </label>

      {preview && <img className="preview" src={preview} alt="Your leaf photo" />}
      {error && <div className="card warn">{error}</div>}

      {result?.status === "retake" && <div className="card warn"><strong>Please retake the photo.</strong><p>{result.reason}</p></div>}

      {result?.status === "ok" && (
        <div className="card">
          <h2>{result.disease === "Healthy" ? "Healthy leaf" : result.disease}</h2>
          <p className="sub">Confidence {(result.confidence * 100).toFixed(0)}%</p>

          {result.severity && (
            <>
              <h3>Severity: {result.severity.grade}</h3>
              <div className="bar"><div style={{ width: `${Math.round(result.severity.score * 100)}%` }} /></div>
              <p className="sub">Score {result.severity.score.toFixed(2)} (0 = healthy-like, 1 = very severe)</p>
            </>
          )}

          <div className={`advice ${result.advice.action}`}>
            <strong>{ACTION_LABEL[result.advice.action]}</strong>
            <p>{result.advice.text}</p>
          </div>

          <div className="maps">
            <figure><img src={`data:image/png;base64,${result.cam_png}`} alt="Where the model looked" /><figcaption>Where the model looked</figcaption></figure>
            {result.severity && (
              <figure><img src={`data:image/png;base64,${result.severity.heatmap_png}`} alt="Least healthy areas" /><figcaption>Least healthy areas</figcaption></figure>
            )}
          </div>
        </div>
      )}
    </main>
  );
}
