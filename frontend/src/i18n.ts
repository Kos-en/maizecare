// Translations live in public/i18n/<lang>.json (nested keys, e.g. "scan.camera").
// Elements with data-i18n="key" get their text from the active language.
export type Lang = "en" | "sw";
type Dict = { [key: string]: unknown };

const STORAGE_KEY = "lang";
let dict: Dict = {};

export function initialLang(): Lang {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "en" || saved === "sw") return saved;
  } catch {
    // storage blocked: fall back to the browser language
  }
  return navigator.language.toLowerCase().startsWith("sw") ? "sw" : "en";
}

export async function loadLang(lang: Lang): Promise<void> {
  dict = (await (await fetch(`i18n/${lang}.json`)).json()) as Dict;
  document.documentElement.lang = lang;
  document.title = t("app.title");
  document.querySelectorAll<HTMLElement>("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n ?? "");
  });
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // language choice just won't be remembered
  }
}

export function get<T>(key: string): T {
  return key.split(".").reduce<unknown>((node, part) => (node as Dict | undefined)?.[part], dict) as T;
}

export function t(key: string, vars: Record<string, string | number> = {}): string {
  return (get<string>(key) ?? key).replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? ""));
}
