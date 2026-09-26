"""
Short, non-clinical coaching text shown in the "Meaning" tab of the app.
MoodTune describes a detected expression, not a diagnosis (report,
Section 8.5: Ethical and Privacy Considerations) - the wording here is
kept supportive and explicitly non-medical.
"""

COACHING = {
    "surprise": {
        "headline": "Something caught your attention",
        "steps": [
            "Take a breath and notice what just surprised you.",
            "If it's a good surprise, let yourself enjoy it for a moment.",
        ],
        "affirmation": "Surprise means you're paying attention to the world around you.",
    },
    "fear": {
        "headline": "It's okay to feel uneasy",
        "steps": [
            "Notice where the tension sits in your body.",
            "Try slow, longer exhales to signal safety to your nervous system.",
        ],
        "affirmation": "Fear is information, not a verdict.",
    },
    "disgust": {
        "headline": "Something feels off right now",
        "steps": [
            "Name what specifically is bothering you, even just to yourself.",
            "Step away from the trigger for a minute if you can.",
        ],
        "affirmation": "It's fine to trust that reaction.",
    },
    "happiness": {
        "headline": "You seem to be in a good place",
        "steps": [
            "Notice what led to this moment so you can find it again.",
            "Let yourself enjoy it without rushing to the next thing.",
        ],
        "affirmation": "This moment is worth savoring.",
    },
    "sadness": {
        "headline": "It's okay to feel this way",
        "steps": [
            "Let the feeling be here without trying to fix it immediately.",
            "If it helps, place a hand where the sadness feels heaviest.",
            "Drink water and unclench your hands - grief lives in the body too.",
        ],
        "affirmation": "Sadness often means something mattered.",
    },
    "anger": {
        "headline": "Something crossed a line for you",
        "steps": [
            "Unclench your jaw and shoulders on purpose.",
            "If you can, put words to what specifically feels unfair.",
        ],
        "affirmation": "Anger often points at a boundary worth listening to.",
    },
    "neutral": {
        "headline": "You seem calm right now",
        "steps": [
            "A neutral moment is a good time to check in with yourself.",
            "No action needed - just noticing is enough.",
        ],
        "affirmation": "Calm is a state worth recognizing, not just passing through.",
    },
}


def get_coaching(emotion: str) -> dict:
    return COACHING.get(emotion, COACHING["neutral"])
