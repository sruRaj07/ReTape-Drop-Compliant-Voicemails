#!/usr/bin/env python3
"""
voicemail_submission.py

Single-file submission for "Drop Compliant Voicemail" assignment.

Features:
- Streams WAV audio files (resamples to 16k mono if needed).
- VAD (webrtcvad) to find speech/silence.
- Narrowband beep detection (peak-to-median in a small band) with consecutive-frame requirement.
- Builds beep candidates and silence candidates, applies priority rules:
    * if a beep occurs before we start playback -> start after beep
    * else if verified silence-after-speech -> start after silence
    * else fallback to MAX_GREETING_TIME
- Verifies candidate start has a required silence window (so company name is heard).
- Optional: transcribe greeting context with Whisper and classify with Google Gemini (improves confidence).
- Debug helpers to extract short clips around chosen starts.

Usage:
    python voicemail_submission.py file1.wav file2.wav ...

Dependencies:
    pip install numpy soundfile webrtcvad scipy openai-whisper google-genai
    brew install ffmpeg   # required by whisper

Set GEMINI_API_KEY environment variable if you want to use Google Gemini:
    export GEMINI_API_KEY="ya29.XXXX..."

Notes:
- Gemini and Whisper are optional. If Gemini fails, script uses robust DSP logic + keyword fallback.
- This script is tuned for voicemail-style recordings; tweak params near the top if needed.
"""

import sys, os, json, tempfile
from typing import Tuple, Dict, List
import numpy as np
import soundfile as sf
from scipy import signal
import webrtcvad

# ---------------- Configuration ----------------
TARGET_SR = 16000
FRAME_MS = 20
VAD_MODE = 2
SILENCE_AFTER_SPEECH = 0.7
MAX_GREETING_TIME = 12.0

# beep detection
BEEP_MIN_FREQ = 850.0
BEEP_MAX_FREQ = 1400.0
PEAK_TO_MEDIAN_RATIO = 10.0
MIN_CONSECUTIVE_BEEP_FRAMES = 2

# verification
REQUIRED_SILENCE_AFTER_START = 0.6
LOOKAHEAD_AFTER_SILENCE = 2.0
LOOKAHEAD_VERIFY = 3.0

# STT / LLM options
USE_WHISPER = True       # set False to skip local transcription
WHISPER_MODEL = "tiny"   # "tiny" or "base" etc.
USE_GEMINI = True        # set False to skip Gemini (will fallback to keyword rules)
GEMINI_MODEL = "gemini-2.5-flash"

# quick phrase list fallback
END_PHRASES = [
    "after the beep", "after the tone", "leave a message", "please leave a message",
    "please leave your name", "please leave your name and number", "i can't take your call",
    "i cannot take your call", "please leave your number", "message after the beep"
]

# ------------------------------------------------

# ---------- Utility functions --------------
def resample_to_target(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    new_samples = int(round(len(y) * float(target_sr) / orig_sr))
    return signal.resample(y, new_samples)

def to_int16_pcm(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind == 'i':
        return y.astype(np.int16)
    y_clipped = np.clip(y, -1.0, 1.0)
    return (y_clipped * 32767).astype(np.int16)

def frame_generator(frame_ms: int, audio: np.ndarray, sample_rate: int) -> List[bytes]:
    samples_per_frame = int(sample_rate * (frame_ms / 1000.0))
    total_samples = len(audio)
    offset = 0
    frames = []
    while offset + samples_per_frame <= total_samples:
        frame = audio[offset:offset+samples_per_frame]
        frames.append(frame.tobytes())
        offset += samples_per_frame
    if offset < total_samples:
        last = audio[offset:]
        pad_len = samples_per_frame - len(last)
        padded = np.concatenate([last, np.zeros(pad_len, dtype=np.int16)])
        frames.append(padded.tobytes())
    return frames

# ---------- beep detector (peak-based) --------------
def detect_beep_frame_peak(frame_int16: np.ndarray, sr: int) -> Tuple[bool, float, float]:
    x = frame_int16.astype(np.float32)
    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return False, 0.0, 0.0
    w = np.hamming(len(x))
    X = np.fft.rfft(x * w)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), d=1.0/sr)
    band_mask = (freqs >= BEEP_MIN_FREQ) & (freqs <= BEEP_MAX_FREQ)
    if not band_mask.any():
        return False, 0.0, 0.0
    band_mag = mag[band_mask]
    if band_mag.size == 0:
        return False, 0.0, 0.0
    local_peak_idx = int(np.argmax(band_mag))
    peak_val = float(band_mag[local_peak_idx])
    peak_freq = float(freqs[band_mask][local_peak_idx])
    baseline = float(np.median(band_mag)) + 1e-12
    ratio = peak_val / baseline
    is_tone = (ratio >= PEAK_TO_MEDIAN_RATIO)
    return is_tone, peak_freq, ratio

# ---------- verify candidate start --------------
def verify_start_time(pcm_int16: np.ndarray, sr: int, candidate_start: float,
                      required_silence: float = REQUIRED_SILENCE_AFTER_START,
                      lookahead_limit: float = LOOKAHEAD_VERIFY,
                      frame_ms: int = FRAME_MS, vad_mode: int = VAD_MODE) -> Tuple[bool, float, float]:
    vad = webrtcvad.Vad(vad_mode)
    samples_per_frame = int(sr * (frame_ms / 1000.0))
    total_samples = len(pcm_int16)
    candidate_sample = int(candidate_start * sr)
    candidate_sample = max(0, min(candidate_sample, total_samples-1))

    def is_silent_window_from(sample_idx):
        end_sample = sample_idx + int(required_silence * sr)
        if end_sample > total_samples:
            return False, None
        t = sample_idx
        while t < end_sample:
            frame = pcm_int16[t: t + samples_per_frame]
            if len(frame) < samples_per_frame:
                frame = np.pad(frame, (0, samples_per_frame - len(frame)), 'constant')
            if vad.is_speech(frame.tobytes(), sample_rate=sr):
                return False, (t + samples_per_frame) / sr
            t += samples_per_frame
        return True, None

    ok, info = is_silent_window_from(candidate_sample)
    if ok:
        return True, candidate_start, 0.95
    search_end_sample = candidate_sample + int(lookahead_limit * sr)
    search_end_sample = min(search_end_sample, total_samples)
    scan = candidate_sample + int(0.05 * sr)
    while scan < search_end_sample:
        ok, info = is_silent_window_from(scan)
        if ok:
            new_start = scan / sr
            distance = new_start - candidate_start
            conf = max(0.3, 0.95 - (distance / lookahead_limit) * 0.6)
            return True, new_start, conf
        if info is not None:
            scan = int(info * sr) + int(0.02 * sr)
        else:
            scan += int(0.05 * sr)
    return False, candidate_start, 0.0

# ---------- Whisper STT + Gemini (optional) ----------
# Whisper transcription (local)
_whisper_model_cache = None
def transcribe_with_whisper(audio_np_float32: "np.ndarray", sr: int, model_name: str = WHISPER_MODEL) -> str:
    if not USE_WHISPER:
        return ""
    global _whisper_model_cache
    try:
        import whisper
    except Exception:
        return ""
    if _whisper_model_cache is None:
        _whisper_model_cache = whisper.load_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpf:
        sf.write(tmpf.name, audio_np_float32, sr)
        res = _whisper_model_cache.transcribe(tmpf.name, language='en', task='transcribe')
    text = res.get("text", "").strip().lower()
    return text

# Gemini classification (uses google-genai client)
def classify_with_gemini(transcript: str) -> dict:
    # fallback keyword-based classifier
    def fallback_keywords(t):
        t = t.lower()
        hits = [p for p in END_PHRASES if p in t]
        return {"end_of_greeting": bool(hits), "expects_beep": any("beep" in p or "tone" in p for p in hits), "confidence": 0.6 if hits else 0.0, "raw": t}
    if not USE_GEMINI:
        return fallback_keywords(transcript)
    try:
        from google import genai
        client = genai.Client()
        prompt = (
            "You are a compact JSON classifier. Return ONLY JSON with keys:\n"
            '  "end_of_greeting": true/false,\n'
            '  "expects_beep": true/false,\n'
            '  "confidence": 0.0-1.0\n'
            "Transcript:\n" + json.dumps(transcript) + "\n\n"
        )
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        raw_text = ""
        try:
            raw_text = response.text
        except Exception:
            try:
                raw_text = response.candidates[0].content.parts[0].text
            except Exception:
                raw_text = str(response)
        jtext = raw_text.strip()
        start = jtext.find('{')
        end = jtext.rfind('}')
        if start != -1 and end != -1 and end > start:
            jtext = jtext[start:end+1]
        parsed = json.loads(jtext)
        eog = bool(parsed.get("end_of_greeting", False))
        expects_beep = bool(parsed.get("expects_beep", False))
        conf = float(parsed.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        return {"end_of_greeting": eog, "expects_beep": expects_beep, "confidence": conf, "raw": raw_text}
    except Exception as exc:
        fb = fallback_keywords(transcript)
        fb['raw_error'] = str(exc)
        return fb

# ---------- main processing ----------
def process_stream_file(path: str) -> dict:
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    if sr != TARGET_SR:
        data = resample_to_target(data, sr, TARGET_SR)
        sr = TARGET_SR
    pcm = to_int16_pcm(data)
    frames_bytes = frame_generator(FRAME_MS, pcm, sr)
    vad = webrtcvad.Vad(VAD_MODE)

    frame_duration = FRAME_MS / 1000.0
    time_pointer = 0.0
    greeting_has_speech = False
    last_speech_time = None
    silence_candidate_time = None
    silence_started_at = None

    beep_runs = []
    beep_candidate_start = None
    beep_candidate_frames = 0
    beep_candidate_peak = 0.0
    consecutive_nonbeep_after_candidate = 0

    debug_frames = []

    for i, frame_bytes in enumerate(frames_bytes):
        is_speech = vad.is_speech(frame_bytes, sample_rate=sr)
        frame_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
        is_tone, peak_freq, ratio = detect_beep_frame_peak(frame_int16, sr)

        if is_tone:
            if beep_candidate_start is None:
                beep_candidate_start = time_pointer
                beep_candidate_frames = 1
                beep_candidate_peak = ratio
            else:
                beep_candidate_frames += 1
                if ratio > beep_candidate_peak:
                    beep_candidate_peak = ratio
            consecutive_nonbeep_after_candidate = 0
        else:
            if beep_candidate_start is not None:
                consecutive_nonbeep_after_candidate += 1
                if consecutive_nonbeep_after_candidate >= 1:
                    run_dur = time_pointer - beep_candidate_start
                    if beep_candidate_frames >= MIN_CONSECUTIVE_BEEP_FRAMES and run_dur >= (FRAME_MS/1000.0)*(MIN_CONSECUTIVE_BEEP_FRAMES - 0.5):
                        beep_runs.append({'start': beep_candidate_start, 'end': time_pointer, 'frames': beep_candidate_frames, 'peak_ratio': beep_candidate_peak, 'peak_freq': peak_freq})
                    beep_candidate_start = None
                    beep_candidate_frames = 0
                    beep_candidate_peak = 0.0
                    consecutive_nonbeep_after_candidate = 0

        if is_speech:
            greeting_has_speech = True
            last_speech_time = time_pointer + frame_duration
            silence_started_at = None
        else:
            if greeting_has_speech:
                if silence_started_at is None:
                    silence_started_at = time_pointer
                if last_speech_time is not None:
                    silence_duration = (time_pointer + frame_duration) - last_speech_time
                    if silence_duration >= SILENCE_AFTER_SPEECH and silence_candidate_time is None:
                        silence_candidate_time = last_speech_time + 0.01
                        silence_detect_frame_time = time_pointer

        debug_frames.append({'t': round(time_pointer,3), 'is_speech': int(is_speech), 'is_tone_frame': int(is_tone), 'peak_freq': round(peak_freq,1), 'peak_ratio': round(ratio,3)})
        time_pointer += frame_duration
        if time_pointer >= MAX_GREETING_TIME:
            break

    if beep_candidate_start is not None:
        run_dur = time_pointer - beep_candidate_start
        if beep_candidate_frames >= MIN_CONSECUTIVE_BEEP_FRAMES:
            beep_runs.append({'start': beep_candidate_start, 'end': time_pointer, 'frames': beep_candidate_frames, 'peak_ratio': beep_candidate_peak, 'peak_freq': None})

    candidates = []
    for br in beep_runs:
        ratio = br.get('peak_ratio', 0.0)
        frames = br.get('frames', 1)
        peak_score = min(1.0, (ratio / PEAK_TO_MEDIAN_RATIO)) * min(1.0, frames / max(1, MIN_CONSECUTIVE_BEEP_FRAMES))
        candidate_start = br['end']
        candidates.append({'type':'beep','orig_start':br['start'],'orig_end':br['end'],'start':candidate_start,'score':peak_score,'peak_freq':br.get('peak_freq',None),'peak_ratio':ratio})

    if silence_candidate_time is not None:
        candidates.append({'type':'silence','orig':silence_candidate_time,'start':silence_candidate_time,'score':0.6})

    # decision logic with verification and STT+Gemini integration
    final_candidate = None
    reason = ""
    debug_info = {'beep_runs': beep_runs, 'silence_candidate_time': silence_candidate_time}

    # helper to run STT+Gemini on greeting context (returns gemini_result dict)
    def analyze_with_stt_gemini(candidate_start):
        # extract last 8s before candidate
        context_sec = 8.0
        end_sample = int(candidate_start * sr)
        start_sample = max(0, end_sample - int(context_sec * sr))
        seg = pcm[start_sample:end_sample].astype('float32') / 32767.0
        transcript = ""
        if USE_WHISPER:
            try:
                transcript = transcribe_with_whisper(seg, sr, model_name=WHISPER_MODEL)
            except Exception as e:
                transcript = ""
        gemini_res = classify_with_gemini(transcript)
        return transcript, gemini_res

    # prefer beep if present and appropriate relative to silence
    beep_candidates = [c for c in candidates if c['type']=='beep']
    beep_candidates_sorted = sorted(beep_candidates, key=lambda x: x['start'])
    chosen = None
    if silence_candidate_time is not None:
        cutoff = silence_candidate_time + LOOKAHEAD_AFTER_SILENCE
        beeps_before_cutoff = [b for b in beep_candidates_sorted if b['orig_end'] <= cutoff]
        if beeps_before_cutoff:
            chosen = max(beeps_before_cutoff, key=lambda x: x['score'])
            reason = f"beep_before_silence_cutoff (beep {chosen['orig_start']:.3f}->{chosen['orig_end']:.3f})"
    else:
        if beep_candidates_sorted:
            chosen = max(beep_candidates_sorted, key=lambda x: x['score'])
            reason = "beep_earliest_high_score"

    if chosen is not None:
        ok, accepted_start, conf_verify = verify_start_time(pcm, sr, chosen['start'], required_silence=REQUIRED_SILENCE_AFTER_START, lookahead_limit=LOOKAHEAD_VERIFY)
        # gather STT/Gemini to update confidence
        transcript, gemini_res = analyze_with_stt_gemini(accepted_start)
        gem_conf = float(gemini_res.get('confidence', 0.0))
        final_confidence = round(conf_verify * (0.35 + 0.65 * gem_conf) * chosen['score'], 3)
        final_candidate = {'type':'beep','accepted_start':accepted_start,'confidence':final_confidence,'chosen':chosen,'transcript':transcript,'gemini':gemini_res}
        debug_info['post_analysis'] = {'transcript': transcript, 'gemini': gemini_res}
    elif silence_candidate_time is not None:
        ok, accepted_start, conf_verify = verify_start_time(pcm, sr, silence_candidate_time, required_silence=REQUIRED_SILENCE_AFTER_START, lookahead_limit=LOOKAHEAD_VERIFY)
        transcript, gemini_res = analyze_with_stt_gemini(accepted_start)
        gem_conf = float(gemini_res.get('confidence', 0.0))
        final_confidence = round(conf_verify * (0.35 + 0.65 * gem_conf) * 0.6, 3)
        if ok and final_confidence >= 0.5:
            final_candidate = {'type':'silence','accepted_start':accepted_start,'confidence':final_confidence,'transcript':transcript,'gemini':gemini_res}
            reason = "silence_after_speech_verified"
        else:
            # fallback to best beep if any
            if beep_candidates_sorted:
                chosen = max(beep_candidates_sorted, key=lambda x: x['score'])
                ok, accepted_start, conf_verify = verify_start_time(pcm, sr, chosen['start'], required_silence=REQUIRED_SILENCE_AFTER_START, lookahead_limit=LOOKAHEAD_VERIFY)
                transcript, gemini_res = analyze_with_stt_gemini(accepted_start)
                gem_conf = float(gemini_res.get('confidence', 0.0))
                final_confidence = round(conf_verify * (0.35 + 0.65 * gem_conf) * chosen['score'], 3)
                final_candidate = {'type':'beep','accepted_start':accepted_start,'confidence':final_confidence,'chosen':chosen,'transcript':transcript,'gemini':gemini_res}
                reason = "fallback_to_beep"
            else:
                ok2, accepted_start2, conf2 = verify_start_time(pcm, sr, MAX_GREETING_TIME, required_silence=0.4, lookahead_limit=1.0)
                final_candidate = {'type':'fallback','accepted_start':accepted_start2,'confidence':round(conf2,3)}
                reason = "timeout_fallback"
    else:
        if beep_candidates_sorted:
            chosen = max(beep_candidates_sorted, key=lambda x: x['score'])
            ok, accepted_start, conf_verify = verify_start_time(pcm, sr, chosen['start'], required_silence=REQUIRED_SILENCE_AFTER_START, lookahead_limit=LOOKAHEAD_VERIFY)
            transcript, gemini_res = analyze_with_stt_gemini(accepted_start)
            gem_conf = float(gemini_res.get('confidence', 0.0))
            final_confidence = round(conf_verify * (0.35 + 0.65 * gem_conf) * chosen['score'], 3)
            final_candidate = {'type':'beep','accepted_start':accepted_start,'confidence':final_confidence,'chosen':chosen,'transcript':transcript,'gemini':gemini_res}
            reason = "beep_no_silence"
        else:
            ok2, accepted_start2, conf2 = verify_start_time(pcm, sr, MAX_GREETING_TIME, required_silence=0.4, lookahead_limit=1.0)
            final_candidate = {'type':'fallback','accepted_start':accepted_start2,'confidence':round(conf2,3)}
            reason = "timeout_fallback"

    out = {
        'file': path,
        'start': round(float(final_candidate['accepted_start']), 3),
        'type': final_candidate.get('type'),
        'confidence': final_candidate.get('confidence'),
        'reason': reason,
        'debug_frames_sample': debug_frames[:10],
        'debug': debug_info
    }
    if 'transcript' in final_candidate:
        out['transcript'] = final_candidate['transcript']
        out['gemini'] = final_candidate['gemini']
    return out

# ---------- demo helper: extract clip ----------
def extract_clip(path: str, center: float, before=0.5, after=1.5, outname="clip.wav"):
    data, sr = sf.read(path, dtype='float32')
    if data.ndim == 2: data = data.mean(axis=1)
    start = int(max(0, (center - before) * sr))
    end = int(min(len(data), (center + after) * sr))
    sf.write(outname, data[start:end], sr)
    return outname, start/sr, end/sr

# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python voicemail_submission.py file1.wav file2.wav ...")
        sys.exit(1)
    files = sys.argv[1:]
    results = []
    for f in files:
        print(f"Processing {f} ...")
        try:
            res = process_stream_file(f)
            results.append(res)
            print(f"File: {res['file']}")
            print(f"Chosen start: {res['start']} s")
            print(f"Type: {res['type']}, Reason: {res['reason']}, Confidence: {res['confidence']}")
            if 'transcript' in res:
                t = res['transcript'][:200].replace("\n"," ")
                print("Transcript (truncated):", t)
                print("Gemini:", res.get('gemini'))
            # write small clip for demo
            clip_name = f"{os.path.splitext(os.path.basename(f))[0]}_demo_clip.wav"
            extract_clip(f, res['start'], before=0.6, after=1.5, outname=clip_name)
            print("Saved demo clip:", clip_name)
        except Exception as e:
            print("Error:", e)
    # write demo output file
    with open("demo_output.txt", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nWrote demo_output.txt (JSON)")

if __name__ == "__main__":
    main()