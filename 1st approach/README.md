# 🚗💥 Accident@CVPR — Video Accident Detection Pipeline

> **Detect, localize, and classify traffic accidents in dashcam/CCTV footage using a multi-stage deep learning pipeline.**

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full System Architecture](#2-full-system-architecture)
3. [Model-by-Model Breakdown](#3-model-by-model-breakdown)
4. [Step-by-Step Execution Flow](#4-step-by-step-execution-flow)
5. [Output Explanation](#5-output-explanation)
6. [Why This Design Was Chosen](#6-why-this-design-was-chosen)
7. [Training Details](#7-training-details)

---

## 1. Project Overview

### What Does This Project Do?

This project takes a **traffic video** (dashcam footage, CCTV, etc.) and automatically answers three questions:

| Question | Output |
|---|---|
| **When** did the accident happen? | `accident_time` — a timestamp in seconds |
| **Where** in the frame did it happen? | `center_x`, `center_y` — normalised coordinates (0.0 to 1.0) |
| **What kind** of accident was it? | `type` — e.g., `rear-end`, `head-on`, `t-bone`, `sideswipe`, `single` |

### The Goal

Given a set of unlabelled test videos, predict the accident time, spatial location of impact, and collision type for each video — and output a `submission.csv` file in competition format.

### High-Level Pipeline

```
Video File (.mp4)
      │
      ▼
 ┌─────────────────────┐
 │  Frame Extraction   │  ← Decode video into individual frames
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  GroundingDINO      │  ← Detect vehicles in every 2nd frame
 │  (Object Detector)  │
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  ByteTrack          │  ← Assign consistent IDs, record trajectories
 │  (Multi-Object      │
 │   Tracker)          │
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  Interaction        │  ← Score frame pairs: IoU + closing speed
 │  Scoring            │    + deceleration → pick top-3 candidates
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  VideoMAE-v2        │  ← Extract 768-dim temporal features from
 │  (Feature           │    16-frame clips (only near candidates)
 │   Extractor)        │
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  TriDet Head        │  ← Predict exact accident timestamp
 │  (Temporal          │    using multi-scale 1D CNN
 │   Localizer)        │
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  Spatial            │  ← Find impact point from colliding
 │  Localization       │    bounding box intersection
 └─────────────────────┘
      │
      ▼
 ┌─────────────────────┐
 │  Trajectory         │  ← Classify accident type from
 │  Classifier         │    pre-collision velocity vectors
 └─────────────────────┘
      │
      ▼
 submission.csv
 (accident_time, center_x, center_y, type)
```

---

## 2. Full System Architecture

### End-to-End Data Flow

Here is exactly how data moves through the system from raw video to final prediction:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT: Video (.mp4)                           │
│           e.g. 30 FPS, 1280×720, 20 seconds = 600 frames            │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ load_frames()   │   Decode all frames into
                    │                 │   RGB numpy arrays [H, W, 3]
                    └────────┬────────┘
                             │  List of ~600 RGB frames
                    ┌────────▼──────────────────┐
                    │  run_tracking()            │
                    │  ┌──────────────────────┐  │
                    │  │  detect_objects()    │  │   Every 15th frame (2 FPS),
                    │  │  GroundingDINO       │  │   detect vehicles → bboxes
                    │  └──────────┬───────────┘  │
                    │             │               │
                    │  ┌──────────▼───────────┐  │
                    │  │  ByteTrack           │  │   Link detections → tracks
                    │  │  .update_with_       │  │   Output: track_id, bbox,
                    │  │   detections()       │  │   velocity per object
                    │  └──────────────────────┘  │
                    └────────┬──────────────────-┘
                             │  det dict + trajectories dict
                    ┌────────▼────────────────────┐
                    │ compute_interaction_scores() │   For every detection frame,
                    │                              │   score all vehicle pairs
                    └────────┬─────────────────────┘
                             │  List of (frame_idx, score, pair)
                    ┌────────▼────────────────────┐
                    │ pick_top_candidates()        │   Keep top-3 non-overlapping
                    │                              │   high-score frames
                    └────────┬─────────────────────┘
                             │  candidate frame indices
                    ┌────────▼────────────────────┐
                    │ build_feature_sequence()     │   Extract 16-frame clips
                    │                              │   around candidates only,
                    │  VideoMAE-v2 backbone        │   ROI-crop to vehicles,
                    │                              │   → [T, 768] feature matrix
                    └────────┬─────────────────────┘
                             │  feats [T, 768] + stamps [T]
                    ┌────────▼────────────────────┐
                    │ tridet_predict_time()        │   TriDet 1D-CNN scores each
                    │  TriDet Head                 │   clip → accident_time (sec)
                    └────────┬─────────────────────┘
                             │  accident_time (float, seconds)
          ┌──────────────────┼────────────────────┐
          │                  │                     │
 ┌────────▼────────┐ ┌───────▼──────────┐ ┌───────▼──────────────┐
 │ find_colliding_ │ │ impact_point_    │ │ classify_from_       │
 │ pair()          │ │ from_boxes()     │ │ trajectories()       │
 │                 │ │                  │ │                      │
 │ IoU + proximity │ │ Intersection of  │ │ Velocity angle       │
 │ scoring near    │ │ colliding bbox   │ │ → accident type      │
 │ accident frame  │ │ → (px, py)       │ │                      │
 └────────┬────────┘ └───────┬──────────┘ └───────┬──────────────┘
          │                  │                     │
          └──────────────────┼─────────────────────┘
                             │
                    ┌────────▼────────────────────┐
                    │  OUTPUT: submission.csv      │
                    │  path, accident_time,        │
                    │  center_x, center_y, type    │
                    └─────────────────────────────-┘
```

---

## 3. Model-by-Model Breakdown

---

### 3.1 GroundingDINO — Zero-Shot Vehicle Detector

#### What is it?
GroundingDINO is a state-of-the-art object detection model that takes both an **image** and a **text description** as input. Instead of being trained to detect only a fixed set of classes, it understands natural language — so you can ask it to find *"car. bus. truck. motorcycle. rickshaw."* without any additional training.

#### Why is it used here?
Traffic videos contain many different vehicle types (rickshaws, auto-rickshaws, motorbikes, etc.) that standard detectors like YOLO may miss if they weren't trained on those classes. GroundingDINO generalises to any vehicle type described in text.

#### What problem does it solve?
**Vocabulary gap** — traditional detectors have a fixed list of 80 or 91 classes. GroundingDINO has no fixed class limit; it detects whatever you describe.

#### Input / Output

```
Input:
  - image:       RGB frame, resized so longest edge ≤ 640px
  - text prompt: "car. bus. truck. motorcycle. bicycle. rickshaw. van. auto rickshaw."

Output:
  - boxes:   [N, 4]  — (x1, y1, x2, y2) bounding boxes in pixel coordinates
  - scores:  [N]     — confidence scores per detection (0.0 → 1.0)
```

#### How it works (intuitive)
Imagine showing someone a photo and saying *"point to every car and bus you see."* GroundingDINO does exactly this. It uses a **vision transformer** to understand the image and a **text encoder** (BERT-based) to understand your words, then fuses both streams together to decide where each described object is. The fusion mechanism lets visual regions "attend to" relevant words, so the word *"rickshaw"* activates the corresponding region of the image.

> **Key setting:** Frames are processed at **2 FPS** (every ~15th frame at 30 FPS). This is fast enough to track vehicles and find accidents, while skipping the heavy computation of running the detector on every single frame.

---

### 3.2 ByteTrack — Multi-Object Tracker

#### What is it?
ByteTrack is a multi-object tracking algorithm. Its job is to watch the detections across frames and decide: *"the car in box A at frame 10 is the same car as the car in box B at frame 25."* It assigns a persistent **track ID** to each vehicle.

#### Why is it used here?
Detecting vehicles frame-by-frame gives you a list of boxes with no identity. You need to know that *Track #5* is a specific car you've been following for 3 seconds, so you can compute its speed, direction, and acceleration over time.

#### What problem does it solve?
**Identity persistence** — connecting detections across time so each vehicle has a continuous trajectory.

#### Input / Output

```
Input:
  - sv.Detections object: bboxes [N, 4] + confidence scores [N]

Output:
  - Updated sv.Detections with tracker_id assigned to each box
  - Internally maintained trajectories:
      trajectories[track_id] = {
          'frames':      [frame_idx, ...],
          'bboxes':      [[x1,y1,x2,y2], ...],
          'velocities':  [[vx, vy], ...]     ← pixels/second
      }
```

#### How it works (intuitive)
ByteTrack uses **IoU matching** (overlap between predicted and detected boxes) to link detections. Most trackers throw away low-confidence detections. ByteTrack's key insight is to use **even low-confidence detections** as "bytes" — they might be noisy, but they help maintain tracks through occlusions. It uses a Kalman filter to predict where each object *will be* in the next frame, and matches predictions to actual detections.

---

### 3.3 Interaction Scoring — Candidate Window Selection

#### What is it?
Not a neural network — this is a **hand-crafted scoring function** that identifies the most suspicious moments in the video (i.e., the frames most likely to contain or lead up to an accident).

#### Why is it used here?
Running VideoMAE on every single frame of a 20-second video would take forever. Instead, we find the top 3 most suspicious windows first, and only run the expensive feature extractor around those.

#### What problem does it solve?
**Computational efficiency + coarse temporal localisation** — narrow the search space from the whole video to 3 candidate windows.

#### The Score Formula

For each pair of vehicles `(i, j)` at each frame:

```
score = 1.5 × IoU(box_i, box_j)          ← Are they overlapping? (collision proxy)
      + 1.0 × (prev_distance - now_distance)   ← Are they getting closer?
      + 0.6 × (decel_i + decel_j)         ← Are they braking?
```

| Component | What it captures |
|---|---|
| **IoU** | Boxes overlapping → vehicles touching |
| **Closing distance** | Vehicles converging → pre-collision approach |
| **Deceleration** | Sudden speed drop → emergency braking |

#### Input / Output

```
Input:  det dict, trajectories dict, fps
Output: List of (frame_idx, score, vehicle_pair_tuple) — one entry per detection frame
        Top-3 candidates after non-maximum suppression (min 2s gap between picks)
```

---

### 3.4 VideoMAE-v2 — Temporal Feature Extractor

#### What is it?
VideoMAE-v2 (Video Masked Autoencoder) is a **video understanding model** from MCG-NJU. It was pre-trained by masking 90% of video patches and learning to reconstruct them — this forces it to learn rich spatiotemporal representations without needing labels.

#### Why is it used here?
A single frame can't tell you if an accident is happening — you need to understand **motion patterns over time**. VideoMAE-v2 watches 16 consecutive frames and compresses everything it sees into a single 768-dimensional feature vector that captures motion, appearance, and temporal dynamics.

#### What problem does it solve?
**Temporal understanding** — converting a clip of 16 frames into a compact feature vector that summarises what's happening in that clip.

#### Input / Output

```
Input:
  - 16 RGB frames, each resized to 224×224
  - Shape: [16, 224, 224, 3]  (processed as a batch by the processor)

Output:
  - Feature vector: [768]  — mean-pooled over all spatial/temporal tokens
  - Across T clips: [T, 768]  — one vector per clip window
```

#### How it works (intuitive)
Think of VideoMAE as reading a flipbook. It watches 16 frames and asks: *"What kinds of motion patterns are present? Are things crashing? Swerving? Braking?"* It was trained on millions of videos and learned to recognise these patterns without being told what accidents look like — because its pretraining forced it to deeply understand all kinds of motion.

**ROI Cropping trick:** Before feeding frames to VideoMAE, the pipeline crops to the **bounding box region** of detected vehicles (with a 20px margin). This focuses the model's attention on the vehicles themselves rather than wasting capacity on sky, buildings, and road markings.

---

### 3.5 TriDet Head — Temporal Action Localiser

#### What is it?
TriDet is a lightweight **1D convolutional neural network** that sits on top of VideoMAE features. It takes the sequence of feature vectors `[T, 768]` and predicts, for each clip window:
1. How likely is an accident happening in this clip? (`confidence`)
2. How far is the accident start from this clip's centre? (`dt_start`)
3. How far is the accident end from this clip's centre? (`dt_end`)

#### Why is it used here?
VideoMAE gives us features, but doesn't tell us *when* the accident is. TriDet is the "decoder" that converts those features into precise temporal boundaries.

#### What problem does it solve?
**Precise temporal localisation** — pinpointing the exact second of the accident within the video, beyond just saying "it's somewhere in this 6-second window."

#### Input / Output

```
Input:
  - Feature sequence: [1, T, 768]  (batch=1, T clip windows, 768-dim each)

Output:
  - conf:  [1, T]    — sigmoid probability of accident in each clip
  - dt:    [1, T, 2] — ReLU offsets: [dt_start, dt_end] in seconds from clip centre
```

#### Architecture

```
[T, 768]
    │
    ▼  permute → [768, T]
┌───────────────────────────────────────────┐
│  3 × Conv1d(768→256, kernel=3) + ReLU    │  ← Backbone: extract patterns
└───────────────────────────────────────────┘
    │  [256, T]
    ├──────────────────────────────────────────────────┐
    │                         │                        │
    ▼                         ▼                        ▼
Conv1d(dilation=1)      Conv1d(dilation=2)      Conv1d(dilation=4)
  [3, T]                  [3, T]                  [3, T]
    └──────────────────────── average ─────────────────┘
                              │  [3, T]
                    ┌─────────┴──────────┐
                    │                    │
                 conf [T]              dt [T, 2]
                 (sigmoid)            (relu)
```

**Multi-scale dilation explained:** Using dilations 1, 2, and 4 means the model sees patterns at different time scales simultaneously — short impacts (dilation=1), medium build-ups (dilation=2), and longer collision sequences (dilation=4). Averaging them gives robust multi-scale temporal reasoning.

---

### 3.6 Spatial Localisation — Impact Point Finder

#### What is it?
A rule-based geometric computation that estimates *where in the frame* the collision happened.

#### How it works

```
Step 1: Find the colliding vehicle pair near the predicted accident frame
        → Try IoU matching first (best overlapping pair)
        → Fall back to proximity + acceleration scoring if IoU=0

Step 2: Compute impact point from the two bounding boxes
        → If boxes overlap: centroid of the intersection rectangle
        → If boxes don't overlap: midpoint between the closest edges

Step 3: Normalise to [0, 1] range
        center_x = impact_pixel_x / frame_width
        center_y = impact_pixel_y / frame_height
```

#### Input / Output

```
Input:  Two bounding boxes [x1, y1, x2, y2] of colliding vehicles
Output: impact_point [px, py] in pixel space → normalised (cx, cy)
```

---

### 3.7 Trajectory-Based Classifier — Accident Type

#### What is it?
A rule-based classifier that uses the **pre-collision velocity vectors** of the two colliding vehicles to determine the type of collision.

#### Collision Types

| Type | Description | Velocity Signature |
|---|---|---|
| `head-on` | Vehicles travelling toward each other | Relative angle > 135° |
| `rear-end` | One vehicle hits another from behind | Relative angle < 30° |
| `t-bone` | Side impact at an intersection | Relative angle 60°–120° |
| `sideswipe` | Glancing parallel collision | Relative angle 30°–60°, similar headings |
| `single` | Only one vehicle involved (e.g., hitting a wall) | Both speeds < 3 px/s threshold |

#### Decision Logic

```python
if speed1 < 3.0 and speed2 < 3.0:
    → "single"           # Both nearly stationary

elif one vehicle is nearly stationary:
    → Check moving vehicle's angle:
      sin(angle) > 0.7   → "t-bone"
      else               → "rear-end"

else:  # Both moving
    cos_angle = dot(vel1, vel2) / (|vel1| × |vel2|)
    relative_angle = degrees(arccos(cos_angle))

    > 135°        → "head-on"
    60° – 120°    → "t-bone"
    30° – 60°     → "sideswipe"
    < 30°         → "rear-end"
```

---

## 4. Step-by-Step Execution Flow

### 4.1 Training Phase (on CARLA Synthetic Data)

The TriDet head is the only learnable component trained in this pipeline. It is trained on synthetic videos generated by the **CARLA driving simulator**, which come with ground-truth accident timestamps.

```
For each synthetic video in labels.csv:

  1. Load video + read ground truth:
       - accident_frame (frame number of the accident)
       - Compute accident_time = accident_frame / fps

  2. Run tracking pipeline:
       - detect_objects() on every ~15th frame
       - build trajectories with velocities

  3. Extract VideoMAE features:
       - build_feature_sequence() → [T, 768] feature matrix + timestamps

  4. Build training targets:
       - For each clip stamp t:
           target_conf[t] = 1.0  if (accident_time - 0.75s) ≤ t ≤ (accident_time + 0.75s)
                          = 0.0  otherwise
           target_dt[t]   = [t - t_start, t_end - t]  ← distance to accident boundaries

  5. Forward pass through TriDet:
       - pred_conf [T], pred_dt [T, 2]

  6. Compute loss:
       - Focal loss on confidence (class imbalance: most clips are non-accident)
       - L1 loss on temporal offsets (only for positive/accident clips)

  7. Backpropagate and update weights (AdamW optimizer)
```

> **Note:** GroundingDINO and VideoMAE-v2 are **frozen** — their weights are not updated during training. Only the TriDet head (tiny ~500K parameter network) is trained.

### 4.2 Inference Phase (on Test Videos)

```
1. load_frames(video_path, max_seconds=20)
   → List of RGB frames + fps

2. run_tracking(frames, fps)
   → det (frame-indexed detection results)
   → trajectories (per-track position/velocity history)

3. compute_interaction_scores(det, trajectories, fps)
   → Scored list of (frame_idx, score, vehicle_pair)

4. pick_top_candidates(scores, fps, n=3, min_gap=2s)
   → Top-3 candidate frame indices

5. build_feature_sequence(frames, fps, det, trajectories,
                           clip_stride=1s, candidate_frames=top3)
   → feats [T, 768], stamps [T]   ← only ~18 clips instead of 20×fps

6. tridet_predict_time(feats, stamps)
   → accident_time (seconds)

7. find_colliding_pair(det, trajectories, acc_frame, fps)
   → (tid1, tid2) or None

8. impact_point_from_boxes(bb1, bb2)
   → (center_x, center_y)  normalised

9. classify_from_trajectories(det, trajectories, accident_time, fps)
   → accident_type string
```

---

## 5. Output Explanation

### submission.csv Format

```csv
path,accident_time,center_x,center_y,type
videos/video_001.mp4,4.217,0.5312,0.4871,rear-end
videos/video_002.mp4,7.850,0.3100,0.6200,t-bone
videos/video_003.mp4,2.003,0.4900,0.5010,head-on
```

| Column | Type | Description |
|---|---|---|
| `path` | string | Relative path to the video file |
| `accident_time` | float (seconds) | Predicted moment of collision |
| `center_x` | float [0.0–1.0] | Horizontal position of impact (0=left, 1=right) |
| `center_y` | float [0.0–1.0] | Vertical position of impact (0=top, 1=bottom) |
| `type` | string | Collision category |

### Understanding the Confidence Score (TriDet)

The TriDet head outputs a `conf` value per clip:

```
conf = 0.9+  → Model is very confident an accident occurs in this clip
conf = 0.5   → Uncertain — could go either way
conf = 0.1-  → Model believes this clip is normal (no accident)
```

When predicting the final timestamp, the top-5 highest-confidence clips are weighted by their confidence and averaged:

```python
accident_time = Σ (conf[i] × predicted_center_time[i]) / Σ conf[i]
```

This **soft voting** produces a smoother, more robust prediction than just picking the single highest-confidence clip.

### Understanding Temporal Boundaries

Each TriDet prediction also outputs `dt_start` and `dt_end`:

```
clip_centre_time = 5.0s
dt_start = 0.6s
dt_end   = 0.8s

Predicted accident window: [5.0 - 0.6, 5.0 + 0.8] = [4.4s, 5.8s]
Predicted accident centre: (4.4 + 5.8) / 2 = 5.1s
```

### Interpreting Spatial Coordinates

```
center_x = 0.5, center_y = 0.5  →  Dead centre of the frame
center_x = 0.2, center_y = 0.7  →  Left side, lower portion of frame
center_x = 0.8, center_y = 0.3  →  Right side, upper portion of frame
```

---

## 6. Why This Design Was Chosen

### Design Rationale

| Design Choice | Why |
|---|---|
| **GroundingDINO** instead of YOLO | Zero-shot → handles rickshaws, auto-rickshaws, unusual vehicle types without retraining |
| **2 FPS detection** instead of 30 FPS | 15× speed-up with minimal accuracy loss; vehicles move slowly relative to frame rate |
| **Candidate-first feature extraction** | Running VideoMAE on a full video at 30 FPS would require ~600 model forward passes; candidates reduce this to ~18 |
| **ROI cropping before VideoMAE** | Forces the model to focus on vehicles, not background; improves feature quality |
| **Frozen pretrained models** | GroundingDINO + VideoMAE have billions of parameters; fine-tuning them on a small dataset would overfit |
| **TriDet (tiny trainable head)** | Only ~500K parameters → fast to train, unlikely to overfit, easy to adapt to new data |
| **Rule-based classifier** | Velocity geometry is interpretable, reliable, and requires no labelled accident-type training data |
| **Focal loss** | Accident clips are rare (1–2 out of 20 per video) → standard BCE loss would be dominated by easy negatives; focal loss down-weights them |

---

## 7. Training Details

### What is Trained?

Only the **TriDet head** is trained. All other components (GroundingDINO, VideoMAE-v2, ByteTrack) are used as frozen pre-trained models.

### Training Data

- **Source:** CARLA synthetic driving simulator videos with ground truth accident timestamps
- **Max samples:** 60 videos (configurable via `TRAIN_SYNTH_MAX`)
- **Split:** Random sample from `labels.csv`

### Loss Functions

The TriDet head is trained with two simultaneous losses:

#### 1. Focal Loss (Classification)

```
Standard BCE:   loss = -[y·log(p) + (1-y)·log(1-p)]

Focal variant:  loss = -(1 - p_t)² × [y·log(p) + (1-y)·log(1-p)]
                         └────────┘
                         Modulating factor: down-weights easy examples
```

**Why Focal Loss?** In a 20-second video, only ~1.5 seconds contain an accident. That means ~93% of clips are "normal." Without focal loss, the model learns to predict everything as 0 (no accident) and still achieves 93% accuracy — but that's useless. Focal loss forces the model to pay attention to the rare accident clips.

#### 2. L1 Regression Loss (Temporal Offsets)

```
loss_reg = |predicted_dt_start - true_dt_start|
         + |predicted_dt_end   - true_dt_end|
```

This is applied **only to positive clips** (where `target_conf = 1.0`), so the model only learns to predict offsets for clips it knows contain an accident.

#### Combined Loss

```
total_loss = focal_loss + 1.0 × regression_loss
```

The coefficient `1.0` balances the two tasks equally.

### Optimiser

```python
optimizer = AdamW(
    tridet.parameters(),
    lr=1e-3,
    weight_decay=1e-4    # L2 regularisation to prevent overfitting
)
```

### Training Loop Summary

```
for video in synthetic_videos[:60]:
    1. Extract VideoMAE features [T, 768]
    2. Build target_conf [T] and target_dt [T, 2]
    3. Forward: pred_conf, pred_dt = tridet(features)
    4. loss = focal(pred_conf, target_conf) + L1(pred_dt, target_dt) [masked]
    5. loss.backward()
    6. optimizer.step()
```
## Acknowledgements

| Model / Tool | Source |
|---|---|
| GroundingDINO | [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) |
| VideoMAE-v2 | [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) |
| ByteTrack | [supervision](https://github.com/roboflow/supervision) by Roboflow |
| TriDet | Architecture inspired by [TriDet: Temporal Action Detection](https://arxiv.org/abs/2303.07347) |
| CARLA Simulator | [carla.org](https://carla.org/) — synthetic training data |

---

*README written for the ACCIDENT@CVPR competition submission.*
