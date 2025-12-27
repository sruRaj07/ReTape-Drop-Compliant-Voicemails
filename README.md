Tunable parameters (what you can change to tune accuracy)

TARGET_SR (e.g. 16000) — audio SR.

FRAME_MS (e.g. 20) — frame size for VAD.

VAD_MODE (0–3) — aggression of VAD.

BEEP_MIN_FREQ / BEEP_MAX_FREQ (e.g. 850–1400) — beep band.

PEAK_TO_MEDIAN_RATIO — how “peaky” a tone must be to count as beep.

MIN_CONSECUTIVE_BEEP_FRAMES — how many consecutive frames required to declare a beep.

SILENCE_AFTER_SPEECH — how long silence must be after speech to be a candidate.

REQUIRED_SILENCE_AFTER_START — verification window after chosen start.

MAX_GREETING_TIME — fallback limit for greeting length.

USE_WHISPER / USE_GEMINI booleans — toggle ML verification.
