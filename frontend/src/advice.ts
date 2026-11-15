// Advice shown to farmers. DRAFT: the proposal requires it to be agreed with an agricultural
// extension officer before release. Kept apart from the code so it can be reviewed and edited.
// Disease guidance is summarised from East African sources (linked in each entry).

export type Band = "healthy" | "mild" | "moderate" | "severe";
export type DiseaseKey = "mln" | "msv" | "nlb";

export interface SeverityAdvice {
  label: string;
  title: string;
  color: string;
  action: string;
  meaning: string;
  steps: string[];
}

export interface Disease {
  short: string;
  name: string;
  color: string;
  kind: string;
  signs: string;
  cause: string;
  steps: string[];
  risk: string;
  source: { name: string; url: string };
}

// score: 0 = like a healthy leaf, 1 = as far from healthy as the worst diseased leaves
export function bandFor(score: number, withinHealthyRange: boolean): Band {
  if (withinHealthyRange) return "healthy";
  if (score < 0.35) return "mild";
  return score < 0.7 ? "moderate" : "severe";
}

export const BANDS: Record<Band, SeverityAdvice> = {
  healthy: {
    label: "Healthy",
    title: "Looks healthy",
    color: "#3daa5c",
    action: "Monitor",
    meaning: "This leaf looks like the healthy maize leaves the model learned from.",
    steps: [
      "Keep checking a few plants across the field every week.",
      "Keep grass weeds down. They shelter the insects that spread maize viruses.",
    ],
  },
  mild: {
    label: "Mild",
    title: "Early signs",
    color: "#8aba3b",
    action: "Watch closely",
    meaning: "Small parts of this leaf look different from a healthy leaf.",
    steps: [
      "Mark this plant and check it again in 3 to 5 days.",
      "Check the plants around it for the same signs.",
      "Compare the signs with the diseases below.",
    ],
  },
  moderate: {
    label: "Moderate",
    title: "Clear damage",
    color: "#e6a817",
    action: "Act now",
    meaning: "A clear part of this leaf is damaged.",
    steps: [
      "Match the signs with a disease below and follow its steps.",
      "Ask an extension officer to confirm the disease before buying any product.",
      "Check the neighbouring plants today.",
    ],
  },
  severe: {
    label: "Severe",
    title: "Severe damage",
    color: "#d94040",
    action: "Remove plant",
    meaning: "Most of this leaf is damaged. The plant can spread disease to the rest of the field.",
    steps: [
      "Pull out the plant and destroy it away from the field (bury or burn it).",
      "Do not leave diseased plants or leaves lying in the field.",
      "Wash your hands and tools afterwards.",
      "Tell your extension officer, especially if many plants look like this.",
    ],
  },
};

export const DISEASES: Record<DiseaseKey, Disease> = {
  mln: {
    short: "MLN",
    name: "Maize Lethal Necrosis",
    color: "#d94040",
    kind: "Virus · spread by thrips, aphids and seed",
    signs:
      "Fine yellow specks and stripes from the base of young leaves that join into mottled patches, then dry out from the leaf edges. The youngest leaves can die (\"dead heart\") and cobs stay small.",
    cause:
      "Two viruses together (MCMV with SCMV), carried by thrips and aphids, and by contaminated seed, tools and crop residue.",
    steps: [
      "Scout often and pull out infected plants straight away.",
      "Plant certified seed of MLN-tolerant hybrids (over 20 are released in East Africa).",
      "Keep grass weeds down, and don't intercrop maize with cabbage, kale or onions (they raise thrips).",
      "Seed dressed with a systemic insecticide protects young plants. Ask an agro-dealer or extension officer.",
      "Leave the field without maize for at least 2 months, and rotate with beans or other legumes.",
      "Clean tools after working in an infected field.",
    ],
    risk: "In Kenya, losses of 23 to 100% were estimated in 2012–2013.",
    source: {
      name: "CIMMYT Nairobi: MLN and its management (2025)",
      url: "https://mln.cimmyt.org/wp-content/uploads/sites/39/2025/10/MAIZE-LETHAL-NECROSIS-MLN-AND-MANAGEMENT.pdf",
    },
  },
  msv: {
    short: "MSV",
    name: "Maize Streak Virus",
    color: "#e6a817",
    kind: "Virus · spread by leafhoppers",
    signs:
      "Narrow, unbroken pale-yellow streaks along the leaf veins, starting on the youngest leaves. Plants infected young stay stunted.",
    cause: "A virus carried by small leafhoppers (Cicadulina). It does not spread through seed.",
    steps: [
      "Plant an MSV-resistant variety. Ask your agro-dealer which ones resist streak in your area.",
      "Plant early in the season, at the same time as your neighbours.",
      "Pull out infected young plants early.",
      "Keep grass weeds down around the field.",
      "Seed dressing protects only young plants, so it cannot replace a resistant variety.",
    ],
    risk: "Losses of up to 90% have been seen on susceptible varieties in Uganda.",
    source: {
      name: "NARO Uganda: Uganda Journal of Agricultural Sciences",
      url: "https://journal.naro.go.ug/index.php/ujas/article/download/433/379/703",
    },
  },
  nlb: {
    short: "Leaf blight",
    name: "Northern Leaf Blight",
    color: "#c0522a",
    kind: "Fungus · spread by wind and rain",
    signs:
      "Long, cigar-shaped grey-green to tan spots, starting on the lower leaves and moving up the plant. Worse in cool, wet weather.",
    cause: "The fungus Exserohilum turcicum. It survives on old maize residue in the field.",
    steps: [
      "Plant a tolerant or resistant hybrid. This is the most reliable control.",
      "Rotate maize with another crop.",
      "Remove or bury maize residue after harvest.",
      "Use balanced fertiliser, and adjust planting dates as your extension officer advises.",
      "Use a fungicide only if an extension officer recommends one.",
    ],
    risk: "Losses of up to 60% have been reported in Kenya.",
    source: {
      name: "ILRI / ReSAKSS policy brief: Maize leaf diseases in Kenya (2019)",
      url: "https://www.slu.se/globalassets/slu.se/samverkan/slu-global/agrifose2030/pdf/pb-2019-ilri-resakss/maize-leaf-diseases--a-curse-to-food-security-in-kenya.pdf",
    },
  },
};
