// Severity levels and disease colours. All wording is in public/i18n/<lang>.json.

export type Band = "healthy" | "mild" | "moderate" | "severe";
export type Disease = "MLND" | "MSV" | "MLB"; // classifier labels (folder names in the dataset)

export const HEALTHY = "Healthy";
export const BANDS: Band[] = ["healthy", "mild", "moderate", "severe"];
export const DISEASES: Disease[] = ["MLND", "MSV", "MLB"];

export const BAND_COLOR: Record<Band, string> = {
  healthy: "#4cc27a",
  mild: "#b5d33f",
  moderate: "#f2b33d",
  severe: "#ef5b4c",
};

export const DISEASE_COLOR: Record<Disease, string> = {
  MLND: "#ef5b4c",
  MSV: "#f2b33d",
  MLB: "#e0814a",
};

// The classifier decides healthy vs diseased; the autoencoder score (0-1) grades how bad it is.
export function bandFor(label: string, score: number): Band {
  if (label === HEALTHY) return "healthy";
  if (score < 0.35) return "mild";
  return score < 0.7 ? "moderate" : "severe";
}
