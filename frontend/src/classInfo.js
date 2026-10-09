// Fixed categorical color per class (never reassigned/cycled), matching the
// first four slots of the validated palette. Each class keeps this color
// everywhere in the app so it functions as a stable visual identity.
export const CLASS_META = {
  glioma: {
    label: "Glioma",
    colorVar: "--series-1",
    description: "Tumour arising from glial (supportive) brain tissue.",
  },
  meningioma: {
    label: "Meningioma",
    colorVar: "--series-2",
    description: "Tumour arising from the meninges surrounding the brain.",
  },
  notumor: {
    label: "No Tumour",
    colorVar: "--series-3",
    description: "No tumour pattern detected in the scan.",
  },
  pituitary: {
    label: "Pituitary",
    colorVar: "--series-4",
    description: "Tumour arising from the pituitary gland.",
  },
};

export const CLASS_ORDER = ["glioma", "meningioma", "notumor", "pituitary"];
