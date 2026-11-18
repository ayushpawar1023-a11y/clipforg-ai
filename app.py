
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel

APP_NAME = "ClipForge AI"
OUTPUT_DIR = Path("/tmp/clipforge")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the transcription model once when the server starts.
model = WhisperModel("base", device="cpu", compute_type="int8")

KEYWORDS = {
    "risk", "trading", "trader", "discipline", "psychology",
    "mistake", "loss", "profit", "strategy", "funded",
    "challenge", "account", "money", "market", "success",
    "failure", "patience", "emotional", "risk management",
    "trading psychology", "discipline"
}

def run(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return result.stdout

def get_duration(video):
    output = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video)
    ])
    return float(output.strip())

def timestamp(seconds):
    ms = int(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

def transcribe(video):
    segments, info = model.transcribe(
        str(video),
        beam_size=3,
        vad_filter=True,
        word_timestamps=True
    )

    results = []
    for segment in segments:
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "words": [
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": w.word
                }
                for w in (segment.words or [])
            ]
        })

    return results, info.language

def score_window(segments, start, end):
    text = " ".join(
        s["text"] for s in segments
        if s["start"] < end and s["end"] > start
    ).lower()

    words = re.findall(r"[a-zA-Z]+", text)
    keyword_score = sum(
        1 for keyword in KEYWORDS if keyword in text
    )

    # Prefer windows containing meaningful speech.
    speech_score = min(len(words) / 35, 2.0)
    duration_score = min(len(text) / 180, 1.0)

    return keyword_score * 2 + speech_score + duration_score

def find_highlights(segments, duration, clip_length, count):
    if duration < clip_length:
        raise ValueError(
            f"Video is only {duration:.1f} seconds. "
            f"Choose a shorter clip length."
        )

    candidates = []
    step = max(5, clip_length // 4)

    for start in range(0, int(duration - clip_length) + 1, step):
        end = start + clip_length
        score = score_window(segments, start, end)
        candidates.append((score, start, end))

    candidates.sort(reverse=True)
    selected = []

    for score, start, end in candidates:
        if all(
            abs(start - item[1]) >= clip_length * 0.8
            for item in selected
        ):
            selected.append((score, start, end))
        if len(selected) == count:
            break

    return sorted(selected, key=lambda item: item[1])

def make_srt(segments, start, end, output):
    lines = []
    index = 1

    for segment in segments:
        if segment["end"] <= start or segment["start"] >= end:
            continue

        # Word-level captions, grouped into short readable phrases.
        words = segment["words"]
        if words:
            groups = [
                words[i:i + 5]
                for i in range(0, len(words), 5)
            ]
            for group in groups:
                a = max(start, group[0]["start"])
                b = min(end, group[-1]["end"])
                text = "".join(w["text"] for w in group).strip()
                if b > a and text:
                    lines.extend([
                        str(index),
                        f"{timestamp(a-start)} --> {timestamp(b-start)}",
                        text,
                        ""
                    ])
                    index += 1
        else:
            a = max(start, segment["start"])
            b = min(end, segment["end"])
            if b > a:
                lines.extend([
                    str(index),
                    f"{timestamp(a-start)} --> {timestamp(b-start)}",
                    segment["text"],
                    ""
                ])
                index += 1

    Path(output).write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

def render_clip(video, start, end, index, segments, hook, overlay):
    clip_dir = OUTPUT_DIR / next(tempfile._get_candidate_names())
    clip_dir.mkdir(parents=True, exist_ok=True)

    srt = clip_dir / "captions.srt"
    output = clip_dir / f"clipforge_{index}.mp4"

    make_srt(segments, start, end, srt)

    filters = (
        "scale=720:1280:force_original_aspect_ratio=increase,"
        "crop=720:1280,"
        f"subtitles={srt}:force_style="
        "'FontName=Arial,FontSize=20,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=3,Shadow=1,"
        "Alignment=2,MarginV=100'"
    )

    command = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(end - start),
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart"
    ]

    # Optional campaign overlay and opening hook.
    if overlay.strip():
        command += [
            "-vf",
            filters + ",drawtext=text='" +
            overlay.replace("'", "") +
            "':fontcolor=white:fontsize=24:"
            "box=1:boxcolor=black@0.65:"
            "x=(w-text_w)/2:y=h-190"
        ]

    command += [str(output)]
    run(command)

    return output

def create_clips(
    video,
    number,
    clip_length,
    hook,
    overlay,
    progress=gr.Progress()
):
    if video is None:
        raise gr.Error("Please upload a video first.")

    source = Path(video)
    duration = get_duration(source)

    if duration > 3600:
        raise gr.Error("For this free prototype, use videos under 60 minutes.")

    progress(0.1, desc="Transcribing video with AI...")
    segments, language = transcribe(source)

    if not segments:
        raise gr.Error("No speech detected. Try a video with clear audio.")

    progress(0.4, desc="Finding highlight moments...")
    highlights = find_highlights(
        segments,
        duration,
        int(clip_length),
        int(number)
    )

    if not highlights:
        raise gr.Error("Could not find suitable clip windows.")

    outputs = []
    for i, (score, start, end) in enumerate(highlights, 1):
        progress(
            0.4 + 0.55 * i / len(highlights),
            desc=f"Rendering clip {i}/{len(highlights)}..."
        )
        output = render_clip(
            source, start, end, i,
            segments, hook, overlay
        )
        outputs.append(output)

    zip_path = OUTPUT_DIR / "clipforge_results.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for output in outputs:
            archive.write(output, output.name)

    report = (
        f"Language detected: {language}\n"
        f"Source duration: {duration:.1f} seconds\n"
        f"Clips created: {len(outputs)}\n\n"
        + "\n".join(
            f"Clip {i}: {start:.1f}s – {end:.1f}s | score {score:.2f}"
            for i, (score, start, end) in enumerate(highlights, 1)
        )
    )

    return str(zip_path), report

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title=APP_NAME
) as demo:

    gr.Markdown(
        """
        # ⚡ ClipForge AI
        ### Turn long videos into vertical Shorts.

        Upload a video, detect highlight moments, add captions
        and export MP4 clips.
        """
    )

    video = gr.File(
        label="Upload your source video",
        file_types=[".mp4", ".mov", ".mkv", ".webm"],
        type="filepath"
    )

    with gr.Row():
        number = gr.Dropdown(
            [1, 2, 3, 5],
            value=3,
            label="Number of clips"
        )
        clip_length = gr.Dropdown(
            [20, 30, 45, 60],
            value=30,
            label="Clip duration (seconds)"
        )

    hook = gr.Textbox(
        label="Opening hook (optional)",
        placeholder="Example: THE BIGGEST TRADING MISTAKE"
    )

    overlay = gr.Textbox(
        label="Campaign overlay",
        value="DISCOUNT CODE: FPMOVEMENT"
    )

    create = gr.Button(
        "🚀 Generate AI Clips",
        variant="primary"
    )

    download = gr.File(label="Download your clips ZIP")
    report = gr.Textbox(label="AI processing report", lines=7)

    create.click(
        create_clips,
        inputs=[video, number, clip_length, hook, overlay],
        outputs=[download, report]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)
