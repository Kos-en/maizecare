"""Plain-language advice. PLACEHOLDER WORDING: the proposal requires the advice for each
class and grade to be agreed with an extension officer BEFORE release. Edit and sign off here."""

ADVICE = {
    "Healthy": ("monitor", "No disease detected. Keep checking your plants regularly."),
    ("MLND", "Mild"): ("monitor", "Early signs of maize lethal necrosis. Check neighbouring plants and ask your extension officer."),
    ("MLND", "Moderate"): ("treat", "Maize lethal necrosis is spreading. Contact your extension officer about control measures."),
    ("MLND", "Severe"): ("remove", "Severe maize lethal necrosis. Remove and destroy affected plants to protect the rest of the field."),
    ("MSV", "Mild"): ("monitor", "Early maize streak virus. Watch for leafhoppers and check nearby plants."),
    ("MSV", "Moderate"): ("treat", "Maize streak virus is developing. Ask your extension officer about leafhopper control."),
    ("MSV", "Severe"): ("remove", "Severe maize streak virus. Remove badly affected plants."),
    ("Blight", "Mild"): ("monitor", "Early leaf blight lesions. Check again in a few days, especially in wet weather."),
    ("Blight", "Moderate"): ("treat", "Leaf blight is spreading. Ask your extension officer about fungicide or resistant varieties."),
    ("Blight", "Severe"): ("remove", "Severe leaf blight. Remove heavily affected plants and avoid replanting maize here next season."),
}
DEFAULT = ("monitor", "Disease detected but the severity grade is uncertain. Ask your extension officer to check the plant.")
WITHIN_HEALTHY = ("monitor", "A disease was detected, but the leaf looks close to healthy. Check again soon.")


def advice_for(disease: str, grade: str | None) -> dict:
    if disease == "Healthy":
        action, text = ADVICE["Healthy"]
    elif grade == "Within healthy range":
        action, text = WITHIN_HEALTHY
    else:
        action, text = ADVICE.get((disease, grade), DEFAULT)
    return {"action": action, "text": text}
