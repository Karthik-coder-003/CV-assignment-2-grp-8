#!/usr/bin/env python3
"""
CV Assignment 2 – Group 8
Event Detection and Video Summarization Pipeline
Run as a script OR copy-paste sections into a Jupyter notebook cell by cell.
"""

# ─────────────────────────────────────────────
# CELL 1 – Imports
# ─────────────────────────────────────────────
import os, re, json, math, csv, shutil, random, subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# CELL 2 – Paths and Config
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR        = PROJECT_ROOT / "data"
VIDEO_DIR       = DATA_DIR / "videos"
ANNOTATION_DIR  = DATA_DIR / "annotations"
PROMPT_DIR      = DATA_DIR / "prompts"

OUT_DIR         = PROJECT_ROOT / "outputs"
EVENT_DIR       = OUT_DIR / "events"
RETRIEVAL_DIR   = OUT_DIR / "retrieval"
EVAL_DIR        = OUT_DIR / "evaluation"
SUMMARY_DIR     = OUT_DIR / "summaries"
LOG_DIR         = OUT_DIR / "logs"
FEATURE_DIR     = OUT_DIR / "features"
AGREEMENT_DIR   = OUT_DIR / "agreements"

for d in [VIDEO_DIR, ANNOTATION_DIR, PROMPT_DIR, EVENT_DIR, RETRIEVAL_DIR,
          EVAL_DIR, SUMMARY_DIR, LOG_DIR, FEATURE_DIR, AGREEMENT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VIDEO_IDS = ["TVSUM_29", "TVSUM_30", "TVSUM_31", "TVSUM_32"]

PROMPT_VARIANTS = {
    "prompt_a": "Retrieve all salient events of the video in chronological order. Return only a numbered list. Each event should be one short sentence.",
    "prompt_b": "Return only the events necessary to understand the video. Keep them chronological and concise in numbered format.",
    "prompt_c": "Return a numbered chronological list of salient events with approximate timestamps in the form: Event (MM:SS - MM:SS).",
}

USE_GPU = torch is not None and torch.cuda.is_available()
DEVICE  = "cuda" if USE_GPU else "cpu"
print("Project root:", PROJECT_ROOT)
print("Device:", DEVICE)


# ─────────────────────────────────────────────
# CELL 3 – Install (uncomment if needed)
# ─────────────────────────────────────────────
# import subprocess, sys
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#     "transformers", "sentence-transformers", "av",
#     "opencv-python", "pandas", "numpy", "scikit-learn"], check=True)
# # For lighthouse (CG-DETR):
# subprocess.run([sys.executable, "-m", "pip", "install", "-q",
#     "git+https://github.com/line/lighthouse.git"], check=True)


# ─────────────────────────────────────────────
# CELL 4 – Utilities
# ─────────────────────────────────────────────
def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

def mmss_to_sec(text):
    mm, ss = text.strip().split(":")
    return int(mm) * 60 + int(ss)

def sec_to_mmss(seconds):
    seconds = max(0, float(seconds))
    return f"{int(seconds//60):02d}:{int(round(seconds%60)):02d}"

def ffprobe_duration(video_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except Exception:
        return None


# ─────────────────────────────────────────────
# CELL 5 – Annotation Loading
# ─────────────────────────────────────────────
ANNOT_LINE = re.compile(
    r"^(.*?),\s*(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\s*,\s*(-?2|-?1|0|1|2)\s*$"
)

def parse_annotation_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip().lstrip("•").strip()
            if not line or line.lower().startswith("event description"):
                continue
            m = ANNOT_LINE.match(line)
            if not m:
                print(f"[WARN] Cannot parse line {line_no} in {Path(path).name}: {line}")
                continue
            desc, start, end, subj = m.groups()
            rows.append({
                "description": desc.strip(),
                "start_sec": mmss_to_sec(start),
                "end_sec":   mmss_to_sec(end),
                "subjectivity": int(subj),
                "duration_sec": mmss_to_sec(end) - mmss_to_sec(start),
            })
    return rows

def load_annotations_for_video(video_id):
    vid_num = video_id.split("_")[-1]
    files = sorted(ANNOTATION_DIR.glob(f"*{vid_num}*.txt"))
    return {f.stem: parse_annotation_file(f) for f in files}


# ─────────────────────────────────────────────
# CELL 6 – Frame Sampling
# ─────────────────────────────────────────────
from PIL import Image

def sample_uniform_frames(video_path, num_frames=16):
    if cv2 is None:
        raise ImportError("opencv-python required")
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
    frames, times = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        times.append(idx / fps)
    cap.release()
    return frames, times

def sample_segmented_frames(video_path, segments=4, frames_per_segment=4):
    duration = ffprobe_duration(video_path)
    if not duration:
        raise ValueError("ffprobe failed")
    if cv2 is None:
        raise ImportError("opencv-python required")
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_frames, out_times = [], []
    for s in range(segments):
        for t in np.linspace(s * duration / segments, (s+1) * duration / segments,
                             frames_per_segment, endpoint=False):
            idx = min(int(t * fps), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                out_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                out_times.append(float(t))
    cap.release()
    return out_frames, out_times


# ─────────────────────────────────────────────
# CELL 7 – Prompt Logging
# ─────────────────────────────────────────────
def log_prompt_run(video_id, model_name, strategy_name, prompt_text, extra=None):
    log_path = PROMPT_DIR / "prompt_log.json"
    logs = load_json(log_path) if log_path.exists() else []
    logs.append({"video_id": video_id, "model_name": model_name,
                 "strategy_name": strategy_name, "prompt_text": prompt_text,
                 "extra": extra or {}})
    save_json(logs, log_path)


# ─────────────────────────────────────────────
# CELL 8 – Event Parsing
# ─────────────────────────────────────────────
def parse_numbered_events(text):
    events = []
    for line in str(text).splitlines():
        line = line.strip()
        m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line) or re.match(r"^\s*[-•]\s*(.+)$", line)
        if m:
            events.append(m.groups()[-1].strip())
    return [re.sub(r"\s+", " ", e).strip(" -•") for e in events if len(e) > 3]

def parse_events_with_timestamps(text):
    pattern = re.compile(r"^\s*(\d+)[.)]\s*(.*?)\s*\((\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\)\s*$")
    rows = []
    for line in str(text).splitlines():
        m = pattern.match(line.strip())
        if m:
            _, event, start, end = m.groups()
            rows.append({"event": event.strip(),
                         "start_sec": mmss_to_sec(start),
                         "end_sec":   mmss_to_sec(end)})
    return rows


# ─────────────────────────────────────────────
# CELL 9 – SmolVLM2 Backend
# ─────────────────────────────────────────────
class SmolVLMExtractor:
    def __init__(self, model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct"):
        self.model_name = model_name
        self.ready = False
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=dtype).to(DEVICE)
            self.ready = True
            print("[OK] SmolVLM2 loaded")
        except Exception as e:
            print(f"[WARN] SmolVLM2 not loaded: {e}")

    def generate(self, video_path, prompt, sampling="uniform", num_frames=16, chunked=False):
        if not self.ready:
            raise RuntimeError("SmolVLM2 not loaded")
        frames, times = (sample_segmented_frames(video_path, 4, max(1, num_frames//4))
                         if chunked else sample_uniform_frames(video_path, num_frames))
        content = [{"type": "image", "url": im} for im in frames]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(DEVICE)
        ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return {"raw_output": text,
                "parsed_events": parse_numbered_events(text),
                "parsed_events_with_ts": parse_events_with_timestamps(text),
                "frame_times": times}


# ─────────────────────────────────────────────
# CELL 10 – QwenVL Backend
# ─────────────────────────────────────────────
class QwenVLExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ"):
        self.model_name = model_name
        self.ready = False
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype).to(DEVICE)
            self.ready = True
            print("[OK] QwenVL loaded")
        except Exception as e:
            print(f"[WARN] QwenVL not loaded: {e}")

    def generate(self, video_path, prompt, sampling="uniform", num_frames=16, chunked=False):
        if not self.ready:
            raise RuntimeError("QwenVL not loaded")
        frames, times = (sample_segmented_frames(video_path, 4, max(1, num_frames//4))
                         if chunked else sample_uniform_frames(video_path, num_frames))
        content = [{"type": "image", "image": im} for im in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        try:
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt_text], images=frames, return_tensors="pt", padding=True).to(DEVICE)
            ids = self.model.generate(**inputs, max_new_tokens=512)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        except Exception as e:
            text = f"Qwen generation failed: {e}"
        return {"raw_output": text,
                "parsed_events": parse_numbered_events(text),
                "parsed_events_with_ts": parse_events_with_timestamps(text),
                "frame_times": times}


# ─────────────────────────────────────────────
# CELL 11 – Semantic Event Coverage
# ─────────────────────────────────────────────
class SemanticMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer(model_name)
            self.util  = util
        except Exception as e:
            print(f"[WARN] SemanticMatcher unavailable: {e}")

    def similarity_matrix(self, preds, gts):
        emb_p = self.model.encode(preds, convert_to_tensor=True)
        emb_g = self.model.encode(gts,   convert_to_tensor=True)
        return self.util.cos_sim(emb_p, emb_g).cpu().numpy()

    def greedy_match(self, preds, gts, threshold=0.45):
        if not preds or not gts or self.model is None:
            return []
        sim, used, rows = self.similarity_matrix(preds, gts), set(), []
        for i, pred in enumerate(preds):
            j     = int(np.argmax(sim[i]))
            score = float(sim[i, j])
            matched = score >= threshold and j not in used
            rows.append({"predicted_event": pred,
                         "annotated_event": gts[j] if matched else None,
                         "similarity": score, "matched": matched})
            if matched:
                used.add(j)
        return rows

def event_coverage_stats(matches, n_gt):
    matched   = sum(1 for m in matches if m["matched"])
    precision = matched / len(matches) if matches else 0.0
    recall    = matched / n_gt if n_gt else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return {"matched": matched, "n_pred": len(matches), "n_gt": n_gt,
            "precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────────────────────────
# CELL 12 – Inter-Annotator Agreement
# ─────────────────────────────────────────────
def interval_iou(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0

def compute_inter_annotator_agreement(video_id):
    ann  = load_annotations_for_video(video_id)
    keys = list(ann.keys())
    if len(keys) < 2:
        return {"video_id": video_id, "error": "Need 2 annotation files"}
    matcher = SemanticMatcher()
    events_a, events_b = ann[keys[0]], ann[keys[1]]
    if not events_a or not events_b or matcher.model is None:
        return {"video_id": video_id, "error": "Empty annotations or no matcher"}
    sim = matcher.similarity_matrix(
        [e["description"] for e in events_a],
        [e["description"] for e in events_b])
    aligned, used_b = [], set()
    for i, ea in enumerate(events_a):
        j  = int(np.argmax(sim[i]))
        eb = events_b[j]
        if j not in used_b and (float(sim[i,j]) >= 0.45 or
                interval_iou(ea["start_sec"], ea["end_sec"], eb["start_sec"], eb["end_sec"]) >= 0.3):
            aligned.append({"a": ea, "b": eb,
                             "semantic_similarity": float(sim[i,j]),
                             "temporal_iou": interval_iou(ea["start_sec"], ea["end_sec"],
                                                          eb["start_sec"], eb["end_sec"])})
            used_b.add(j)
    labels_a = [x["a"]["subjectivity"] for x in aligned]
    labels_b = [x["b"]["subjectivity"] for x in aligned]
    kappa = None
    if labels_a:
        from sklearn.metrics import cohen_kappa_score
        kappa = float(cohen_kappa_score(labels_a, labels_b))
    out = {"video_id": video_id, "annotator_a": keys[0], "annotator_b": keys[1],
           "n_aligned_pairs": len(aligned),
           "mean_temporal_iou": float(np.mean([x["temporal_iou"] for x in aligned])) if aligned else None,
           "cohens_kappa_subjectivity": kappa}
    save_json(out, AGREEMENT_DIR / f"{video_id}_agreement.json")
    print(f"[{video_id}] Agreement: kappa={kappa:.3f}, aligned={len(aligned)}")
    return out


# ─────────────────────────────────────────────
# CELL 13 – Query Variants for Retrieval
# ─────────────────────────────────────────────
def generate_query_variants(event_text):
    e = event_text.strip().rstrip(".")
    variants = [e, f"the moment when {e.lower()}",
                f"a scene where {e.lower()}",
                f"the event in which {e.lower()}",
                f"video segment showing {e.lower()}"]
    seen, out = set(), []
    for v in variants:
        if v.lower() not in seen:
            out.append(v); seen.add(v.lower())
    return out


# ─────────────────────────────────────────────
# CELL 14 – CG-DETR Backend (Lighthouse)
# ─────────────────────────────────────────────
class CGDETRBackend:
    def __init__(self, ckpt_path="results/cg_detr/qvhighlight/clip/best.ckpt",
                 device="cpu", feature_name="clip"):
        self.ready = False
        self.model = None
        try:
            from lighthouse.models import CGDETRPredictor
            self.model = CGDETRPredictor(ckpt_path, device=device, feature_name=feature_name)
            self.ready = True
            print("[OK] CG-DETR loaded")
        except Exception as e:
            print(f"[WARN] CG-DETR unavailable: {e}")

    def encode_video(self, video_path):
        return self.model.encode_video(str(video_path))

    def predict_query(self, encoded_video, query):
        pred = self.model.predict(query, encoded_video)
        windows = pred.get("pred_relevant_windows", []) if isinstance(pred, dict) else []
        return [{"start": float(w[0]), "end": float(w[1]),
                 "score": float(w[2]), "query": query}
                for w in windows if len(w) >= 3]


# ─────────────────────────────────────────────
# CELL 15 – Moment-DETR Backend
# ─────────────────────────────────────────────
class MomentDETRBackend:
    """
    Point REPO_PATH to your local moment_detr clone.
    If not available, a deterministic fallback placeholder is used so the
    rest of the pipeline can still run.
    """
    REPO_PATH = None  # e.g. Path("/home/user/moment_detr")

    def __init__(self, repo_path=None):
        self.ready = False
        self.predictor = None
        path = Path(repo_path) if repo_path else self.REPO_PATH
        if path and path.exists():
            try:
                import sys; sys.path.insert(0, str(path))
                # from run_on_video.run import MomentDETRPredictor
                # self.predictor = MomentDETRPredictor(...)
                # self.ready = True
                print("[INFO] Moment-DETR repo found; wire up predictor above.")
            except Exception as e:
                print(f"[WARN] Moment-DETR import failed: {e}")

    def predict_query(self, video_path, query, fallback_duration=None):
        if self.ready and self.predictor:
            return self.predictor.localize_moment(video_path=str(video_path), query_list=[query])
        dur = fallback_duration or ffprobe_duration(video_path) or 60.0
        center = abs(hash(query)) % max(int(dur) - 5, 1)
        return [{"start": float(center), "end": float(center + 4.0),
                 "score": 0.01, "query": query,
                 "note": "placeholder – replace with real Moment-DETR predictor"}]


# ─────────────────────────────────────────────
# CELL 16 – Segment Post-Processing
# ─────────────────────────────────────────────
def temporal_iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0

def merge_overlapping_segments(segs, iou_threshold=0.3):
    if not segs: return []
    segs = sorted(segs, key=lambda x: x["start"])
    merged = [segs[0].copy()]
    for s in segs[1:]:
        last = merged[-1]
        if temporal_iou(last, s) >= iou_threshold or s["start"] <= last["end"]:
            last["end"]   = max(last["end"], s["end"])
            last["score"] = max(last.get("score", 0), s.get("score", 0))
        else:
            merged.append(s.copy())
    return merged

def remove_short_segments(segs, min_duration=1.0):
    return [s for s in segs if (s["end"] - s["start"]) >= min_duration]

def enforce_chronological_consistency(segs):
    out, prev_end = [], -1
    for s in sorted(segs, key=lambda x: x["start"]):
        s = s.copy()
        if s["start"] < prev_end:
            s["start"] = prev_end
        if s["end"] > s["start"]:
            out.append(s); prev_end = s["end"]
    return out

def fuse_query_results(query_results, mode="max"):
    all_segs = [s for qs in query_results for s in qs]
    if mode == "max":
        best = max(all_segs, key=lambda x: x.get("score", 0), default=None)
        return [best] if best else []
    merged = merge_overlapping_segments(all_segs)
    return enforce_chronological_consistency(remove_short_segments(merged))


# ─────────────────────────────────────────────
# CELL 17 – Retrieval Evaluation & Failure Analysis
# ─────────────────────────────────────────────
def evaluate_retrieval(video_id, predicted_segments, gt_events, method_name):
    rows = []
    for seg in predicted_segments:
        best_iou, best_gt = 0.0, None
        for gt in gt_events:
            iou = temporal_iou(seg, {"start": gt["start_sec"], "end": gt["end_sec"]})
            if iou > best_iou:
                best_iou, best_gt = iou, gt
        rows.append({"video_id": video_id, "method": method_name,
                     "query": seg.get("query", ""), "start": seg["start"], "end": seg["end"],
                     "score": seg.get("score"), "best_iou": best_iou,
                     "matched_gt": best_gt["description"] if best_gt else None,
                     "matched_subjectivity": best_gt["subjectivity"] if best_gt else None})
    df = pd.DataFrame(rows)
    summary = {}
    if len(df):
        summary = {"mean_iou": float(df["best_iou"].mean()),
                   "recall@0.3": float((df["best_iou"] >= 0.3).mean()),
                   "recall@0.5": float((df["best_iou"] >= 0.5).mean())}
    return df, summary

def failure_analysis_table(eval_df):
    df = eval_df.copy()
    df["failure_type"] = np.where(df["best_iou"] < 0.3, "missed_or_bad_boundary", "acceptable")
    df["query_length"] = df["query"].fillna("").str.split().str.len()
    return df[["video_id","method","query","query_length","best_iou","failure_type","matched_gt"]]


# ─────────────────────────────────────────────
# CELL 18 – Summary Video Generation
# ─────────────────────────────────────────────
def extract_subclip(video_path, start, end, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(video_path),
           "-c:v", "libx264", "-c:a", "aac", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)

def concat_clips(clip_paths, output_path):
    list_file = Path(output_path).with_suffix(".txt")
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-c:v", "libx264", "-c:a", "aac", str(output_path)]
    subprocess.run(cmd, check=True, capture_output=True)

def build_summary_video(video_id, video_path, segments, top_k=None):
    segs = sorted(segments, key=lambda x: x["start"])
    if top_k:
        segs = sorted(segs, key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        segs = sorted(segs, key=lambda x: x["start"])
    clip_paths = []
    for i, seg in enumerate(segs, start=1):
        label = f"clip_{i:02d}_{sec_to_mmss(seg['start']).replace(':','-')}_{sec_to_mmss(seg['end']).replace(':','-')}"
        clip_path = SUMMARY_DIR / video_id / f"{label}.mp4"
        try:
            extract_subclip(video_path, seg["start"], seg["end"], clip_path)
            clip_paths.append(clip_path)
        except Exception as e:
            print(f"[WARN] Clip extraction failed for {label}: {e}")
    final = SUMMARY_DIR / f"{video_id}_summary.mp4"
    if clip_paths:
        try:
            concat_clips(clip_paths, final)
        except Exception as e:
            print(f"[WARN] Concat failed: {e}")
    return final, clip_paths


# ─────────────────────────────────────────────
# CELL 19 – Full Pipeline Runner
# ─────────────────────────────────────────────
def run_full_pipeline(video_id,
                      cgdetr_ckpt="results/cg_detr/qvhighlight/clip/best.ckpt",
                      cgdetr_device="cpu",
                      moment_detr_repo=None,
                      num_frames=16):
    print(f"\n{'='*60}\nRunning pipeline for {video_id}\n{'='*60}")

    # 1. Inter-annotator agreement
    agreement = compute_inter_annotator_agreement(video_id)

    # 2. VLM event detection
    smol_vlm = SmolVLMExtractor()
    qwen_vlm = QwenVLExtractor()
    vlm_results, candidate_events = [], []

    for model, strategies in [
        (smol_vlm, [("smol_uniform_a",  "prompt_a", "uniform",    False),
                    ("smol_segment_b",  "prompt_b", "segmented",  True),
                    ("smol_uniform_c",  "prompt_c", "uniform",    False)]),
        (qwen_vlm, [("qwen_segment_a",  "prompt_a", "segmented",  True),
                    ("qwen_segment_c",  "prompt_c", "segmented",  True)]),
    ]:
        if not model.ready:
            continue
        for strategy, prompt_key, sampling, chunked in strategies:
            video_path = VIDEO_DIR / f"{video_id}.mp4"
            prompt = PROMPT_VARIANTS[prompt_key]
            log_prompt_run(video_id, model.model_name, strategy, prompt,
                           {"sampling": sampling, "num_frames": num_frames, "chunked": chunked})
            result = model.generate(video_path, prompt, sampling=sampling,
                                    num_frames=num_frames, chunked=chunked)
            result.update({"video_id": video_id, "model_name": model.model_name,
                            "strategy_name": strategy, "prompt_key": prompt_key})
            save_json(result, EVENT_DIR / f"{video_id}_{strategy}.json")
            vlm_results.append(result)
            candidate_events.extend(result.get("parsed_events", []))

    # 2b. Semantic coverage evaluation
    matcher = SemanticMatcher()
    ann     = load_annotations_for_video(video_id)
    for vr in vlm_results:
        preds = vr.get("parsed_events", [])
        for annotator, gt_events in ann.items():
            gt_texts = [e["description"] for e in gt_events]
            matches  = matcher.greedy_match(preds, gt_texts)
            stats    = event_coverage_stats(matches, len(gt_texts))
            save_csv([{**m, "video_id": video_id, "annotator": annotator} for m in matches],
                     EVAL_DIR / f"{video_id}_{vr['strategy_name']}_{annotator}_semantic_matches.csv")
            print(f"  [{vr['strategy_name']}] recall={stats['recall']:.2f} F1={stats['f1']:.2f}")

    # fallback: use annotation events if VLM not available
    if not candidate_events and ann:
        first = list(ann.values())[0]
        candidate_events = [e["description"] for e in first]

    # deduplicate
    seen, final_events = set(), []
    for e in candidate_events:
        k = e.lower().strip()
        if k and k not in seen:
            final_events.append(e); seen.add(k)

    # 3. Retrieval
    retrieval_rows = []
    cgdetr   = CGDETRBackend(ckpt_path=cgdetr_ckpt, device=cgdetr_device)
    mdetr    = MomentDETRBackend(repo_path=moment_detr_repo)

    for event in final_events:
        queries = generate_query_variants(event)
        for backend_name, backend_fn in [
            ("CG-DETR",      lambda q: cgdetr.predict_query(cgdetr.encode_video(VIDEO_DIR / f"{video_id}.mp4"), q) if cgdetr.ready else []),
            ("Moment-DETR",  lambda q: mdetr.predict_query(VIDEO_DIR / f"{video_id}.mp4", q))
        ]:
            qr = [backend_fn(q) for q in queries]
            fused = fuse_query_results(qr, mode="vote")
            for seg in fused:
                seg.update({"event": event, "backend": backend_name})
                retrieval_rows.append(seg)

    save_json(retrieval_rows, RETRIEVAL_DIR / f"{video_id}_all_retrieval.json")

    # 4. Evaluation
    if ann and retrieval_rows:
        gt_events = list(ann.values())[0]
        dfs = []
        for method in set(r["backend"] for r in retrieval_rows):
            rows = [r for r in retrieval_rows if r["backend"] == method]
            df, summary = evaluate_retrieval(video_id, rows, gt_events, method)
            if len(df):
                df.to_csv(EVAL_DIR / f"{video_id}_{method.lower().replace('-','_')}_iou.csv", index=False)
                failure_analysis_table(df).to_csv(
                    EVAL_DIR / f"{video_id}_{method.lower().replace('-','_')}_failures.csv", index=False)
            save_json(summary, EVAL_DIR / f"{video_id}_{method.lower().replace('-','_')}_iou_summary.json")
            print(f"  [{method}] mean IoU={summary.get('mean_iou', 0):.3f} recall@0.5={summary.get('recall@0.5', 0):.3f}")
            dfs.append(df)

    # 5. Build summary video
    summary_info = {}
    if retrieval_rows:
        selected = enforce_chronological_consistency(
            remove_short_segments(
                merge_overlapping_segments(
                    sorted(retrieval_rows, key=lambda x: x.get("score", 0), reverse=True)[:8]),
                min_duration=1.0))
        final_path, clips = build_summary_video(video_id, VIDEO_DIR / f"{video_id}.mp4", selected)
        summary_info = {"summary_path": str(final_path), "n_clips": len(clips)}

    report = {"video_id": video_id, "agreement": agreement,
              "n_candidate_events": len(final_events), "summary_info": summary_info}
    save_json(report, LOG_DIR / f"{video_id}_pipeline_report.json")
    return report


# ─────────────────────────────────────────────
# CELL 20 – Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Change checkpoint paths to match your local setup
    for vid in VIDEO_IDS:
        run_full_pipeline(
            vid,
            cgdetr_ckpt="results/cg_detr/qvhighlight/clip/best.ckpt",
            cgdetr_device=DEVICE,
            moment_detr_repo=None,  # e.g. "/home/user/moment_detr"
            num_frames=16
        )
