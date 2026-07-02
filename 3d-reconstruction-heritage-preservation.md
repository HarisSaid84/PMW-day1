# What I'm Learning About 3D Scene Reconstruction — and Why It Matters for Heritage Preservation

I'm a Computer Science undergraduate at FAST-NUCES Islamabad, and over the past few weeks I've been going down a rabbit hole: **AI-based 3D scene reconstruction** — the process of turning flat photos or video into a 3D model you can actually walk around in. This post is a snapshot of what I've learned so far, why I think it matters far beyond tech demos, and what I'm planning to try next.

## My PMW Track: AI-Based 3D Scene Reconstruction

My PMW (PreserveMy.World) track is **AI Based 3D Scene Reconstruction** — figuring out how ordinary phone footage of a place can be turned into a 3D scene worth revisiting, studying, or sharing. That's a deceptively simple sentence for a genuinely hard computer vision problem, and it's the thread I'm pulling on this month.

## Why 3D Reconstruction Matters for Heritage Preservation

A photo tells you a place existed. It doesn't tell you where the doorway was, how the light fell across a courtyard at noon, or how it felt to actually stand inside a room before it changed. That gap — between *documentation* and *experience* — is exactly the problem heritage preservation runs into, and it's exactly what 3D reconstruction is built to close.

This is where **PreserveMy.World (PMW)** comes in. PMW turns ordinary phone footage into walkable 3D worlds — for heritage sites, family homes, classrooms, and travel memories — without requiring expensive scanning gear. The workflow is intentionally simple: someone walks slowly through a place with a phone, uploads the footage, and the system extracts frames, reconstructs the scene, and generates a preview that can be shared as a public archive page. Sites like Lahore's Badshahi Masjid or a mountain route at Fairy Meadows become something you can step back into, not just look at.

That matters because heritage is fragile in ways that outpace how fast we document it. Buildings get renovated, conflict damages monuments, natural sites erode season by season, and family spaces disappear the moment a house is sold or torn down. Once a place changes, a flat photo is the only thing left. A walkable 3D reconstruction, on the other hand, preserves scale, spatial relationships, and context — the things that actually let future students, researchers, or family members understand *how a place was*, not just *that it was*. That's the core reason I wanted my PMW track to be specifically about the AI side of reconstruction: better models mean lower barriers to entry, so preservation doesn't stay limited to famous monuments with big budgets.

## What I've Learned So Far

A few concepts kept coming up as I read through papers and survey articles on this topic:

- **Structure from Motion (SfM):** Estimating camera positions and a sparse 3D point cloud from a set of overlapping images — the classic first stage in most photogrammetry pipelines (this is what tools like COLMAP do).
- **Multi-View Stereo (MVS):** Taking that sparse point cloud and densifying it into a much more detailed surface, using pixel correspondences across many views.
- **Monocular depth estimation:** Neural networks that predict a depth map from a *single* image — useful when you don't have enough overlapping views for classical photogrammetry.
- **Neural Radiance Fields (NeRF):** Instead of building an explicit mesh, NeRF learns a continuous function that predicts color and density at any point in space, letting you render novel viewpoints that were never actually captured.
- **3D Gaussian Splatting:** A newer alternative to NeRF that represents a scene as a large set of 3D Gaussians instead of a neural network, which can be rendered in real time — a big deal for anything meant to run on a phone or in a browser.

The pattern I noticed: classical photogrammetry (SfM + MVS) is well understood and reliable but can be finicky with sparse or low-texture footage — exactly the kind of casual, walk-through phone video PMW is designed around. Newer neural methods (NeRF, Gaussian Splatting) tend to handle those messier, more casual captures better, which is probably why they're where a lot of the current research energy is going.

## Method I Want to Explore Next

The method I'm most interested in trying next is **3D Gaussian Splatting**. A few reasons:

1. It renders in real time, which matters if PMW's end goal is something people can actually *walk through* rather than wait to load.
2. It tends to handle imperfect, casually-captured footage — phone video with inconsistent lighting and motion — better than classical MVS pipelines.
3. There's a growing set of open-source implementations, so it's realistic for me to get a working pipeline running on a personal dataset without needing research-lab compute.

I'm also keeping monocular depth estimation on my list as a lighter-weight fallback for single-image or low-overlap situations, since not every submitted memory will have enough frames for a full multi-view reconstruction.

## Experiment Evidence — What's Next This Week

This post is documenting research and reading, not a working pipeline. So instead of screenshots, here's the honest plan for this week:

- **Step 1:** Set up COLMAP locally (or on Colab, if my machine's GPU turns out to be a bottleneck) and run it on a short phone video of a local landmark, just to get a baseline sparse point cloud and camera poses.
- **Step 2:** Feed that COLMAP output into an open-source 3D Gaussian Splatting implementation and see what a first, rough reconstruction looks like.
- **Step 3:** Document everything — terminal output, run times, and rendered screenshots — in a follow-up post, including what broke and what I had to debug.


## Closing Thoughts

3D reconstruction sits at an interesting intersection for me — it's a genuinely hard AI/computer vision problem, but the payoff isn't abstract. Projects like PreserveMy.World show what it looks like when the research actually reaches people trying to hold onto a place before it changes. That's the part I want my PMW track to contribute to.

*— Haris Said*
