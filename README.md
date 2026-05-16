# Real-Time 2D Object Detection

Classical computer-vision pipeline that recognizes household objects from a
top-down webcam feed in real time, using hand-rolled morphology, region
segmentation, and shape-based features. No deep learning, no GPU. Just
thresholding, the grassfire transform, Hu moments, and a scaled-Euclidean
nearest-neighbor classifier. **93.1% accuracy on 12 classes** (54 / 58 trials).

Built in C++ with OpenCV for *CS 5330: Pattern Recognition and Computer Vision*
at Northeastern University (Spring 2022). Most of the building blocks
(grassfire growing/shrinking, region segmentation, k-means, KNN) are
implemented from scratch rather than pulled from `cv::` calls, because the
goal was to understand the math, not to invoke library functions.

## Demo

A 3.2 MB demo video lives in this repo at
[`Video Result.mp4`](./Video%20Result.mp4) and the full project report (with
intermediate images for every pipeline stage) is at
[`Project Report.pdf`](./Project%20Report.pdf).

## Why this exists

Modern object detection has collapsed into "fine-tune a CNN, point it at a
GPU, ship." That works, and it's what I'd reach for in production. But the
classical pipeline still earns its keep when:

- the deploy target is a CPU with no accelerator
- training data is tiny (here: 5 reference frames per object)
- you need to *explain* every misclassification, not just point at attention maps
- the scene is constrained (top-down, light background, clean lighting)

Working through the pipeline by hand gave me intuition for everything that
sits inside today's CNN feature extractors. Convolution, pooling, region
proposals, even non-max suppression all trace back to operations like
thresholding, morphology, and connected components. This repo is the from-scratch
version of those primitives.

## The pipeline

```
Webcam frame (BGR)
        |
        v
HSV threshold (S < 55, V > 80)                      <- background suppression
        |
        v
Grassfire shrink(1) -> grow(9) -> shrink(11) -> grow(3)   <- noise + holes
        |
        v
connectedComponentsWithStats + area filter (> 500 px)     <- regions
        |
        v
Custom segmentation: drop boundary blobs                  <- keep ROIs
        |
        v
findContours -> minAreaRect (oriented bounding box)       <- shape
        |
        v
Features per region:
  - aspect ratio (short / long side of OBB)
  - percentage filled (contour area / OBB area)
  - log|Hu moment| for moments 1..6
        |
        v
Three classifiers in parallel:
  - Scaled-Euclidean nearest neighbor (primary)
  - KNN, k = 2
  - K-means, k = 3 (trained on a 3-object subset)
        |
        v
Annotated frame: contour (blue), OBB (red), centroid + axis (yellow),
                 predicted label (magenta)
```

The 8-dim feature vector is invariant to translation, rotation, and scale by
construction. Aspect ratio comes from the oriented bounding box (rotation
invariant), percentage-filled is a ratio (scale invariant), and Hu moments
are translation/rotation/scale invariant by definition. I take `log|hu_i|`
because raw Hu moments span many orders of magnitude.

## Quickstart

You need OpenCV 4.x and a C++ compiler. The source files don't ship with a
CMakeLists, so the simplest path is to compile both targets directly with
`pkg-config`:

```bash
cd "Project Files"

# Live detection from webcam (device 0)
g++ vidDisplay.cpp csv_util.cpp -o objdet `pkg-config --cflags --libs opencv4`

# Training mode: feed a single image, append its features to data.csv
g++ Training_mode.cpp csv_util.cpp -o train `pkg-config --cflags --libs opencv4`
```

Run:

```bash
./objdet                  # opens the webcam and prints predictions per frame
./train                   # interactive: asks for image path, db file, label
```

`data.csv`, `knn.csv`, and `kmeans.csv` ship pre-populated with 12 objects so
the detector works out of the box.

> Note: `objectdetection.hpp` `#include`s `objectdetection.cpp` directly
> (single-translation-unit shortcut from coursework). Don't link
> `objectdetection.cpp` separately or you'll get multiple-definition errors.

## Results

Confusion matrix from 5 trials per object (Key only got 3 trials because
specular reflection on the polished metal got rejected as background twice;
that failure mode is itself a useful finding):

|                        | Predicted correctly | Trials |
| ---------------------- | ------------------- | ------ |
| Beats Ear Buds Case    | 5                   | 5      |
| Spatula                | 5                   | 5      |
| Key                    | 3                   | 3      |
| Wallet                 | 4                   | 5      |
| Glass                  | 5                   | 5      |
| Trimmer                | 5                   | 5      |
| Indian Candle Holder   | 4                   | 5      |
| Scissors               | 4                   | 5      |
| Spoon                  | 5                   | 5      |
| Heart Candy            | 5                   | 5      |
| Hair Comb              | 4                   | 5      |
| Hair Comb Horizontal   | 5                   | 5      |
| **Total**              | **54**              | **58** |

**Overall accuracy: 93.1%.** Seven of twelve classes were perfect; the other
four each missed exactly once, with the correct class as a close second in
the distance metric.

Raw data: [`Project Files/5 Features/confusion_matrix.csv`](./Project%20Files/5%20Features/confusion_matrix.csv).

## What's in the repo

```
Real-Time-2D-Object-Detection/
|-- Project Files/
|   |-- objectdetection.cpp        # pipeline: threshold, morph, segment, features, classify
|   |-- objectdetection.hpp        # declarations + single-TU include shortcut
|   |-- vidDisplay.cpp             # main loop, captures webcam frames
|   |-- Training_mode.cpp          # offline training: image -> feature vector -> data.csv
|   |-- csv_util.cpp / .hpp        # CSV read/write (provided by course staff, attributed)
|   |-- data.csv / knn.csv         # 12-object reference features for NN + KNN
|   |-- kmeans.csv                 # 3-object subset for k-means demo
|   |-- 1 Original/ ... 5 Features/   # 60 stage-by-stage output frames
|   |-- Morphological Transforms/  # distance-transform visualizations
|   |-- Train/, TrainingImages/    # source images for the reference set
|-- Project Report.pdf             # full writeup with stage-by-stage images
|-- Video Result.mp4               # 3.2 MB live-detection demo
```

## Tech

- **Language:** C++ (C++14)
- **Vision:** OpenCV 4.x (cv::Mat, HSV conversion, findContours, minAreaRect, HuMoments, connectedComponentsWithStats)
- **From-scratch:** HSV thresholding, grassfire grow/shrink (two-pass Manhattan distance transform), region segmentation with area + border filtering, scaled-Euclidean distance, KNN, k-means
- **Classifiers:** nearest neighbor (primary), KNN (k=2), k-means (k=3)
- **Tested on:** Ubuntu 20.04, integrated webcam + DroidCam

## Things I'd change today

- Wrap the build in a CMakeLists so it's one `cmake -B build && cmake --build build` away
- Replace the `.cpp` include from `.hpp` with a proper TU split
- Add a small CLI flag layer (`--training`, `--db <path>`, `--threshold-s <int>`) instead of recompiling
- For metallic objects (Key, Scissors), add a second pass on a saturation-only threshold and union the masks; specular highlights were the single biggest accuracy killer
- Swap the hand-rolled k-means for `cv::kmeans` if speed ever mattered (it didn't, here)

## Credit

- `csv_util.cpp / .hpp` were provided by Prof. Bruce Maxwell as starter code for the CS 5330 project series; attribution is preserved in the file header.
- Everything else is mine.

## License

MIT. See [LICENSE](./LICENSE).
