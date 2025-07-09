# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip3 install -r requirements.txt
python3 setup.py develop

# Install additional dependencies
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox
```

### Training
```bash
# Train ablation model (MOT17 half train + CrowdHuman)
python3 tools/train.py -f exps/example/mot/yolox_x_ablation.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth

# Train MOT17 test model (MOT17 + CrowdHuman + Cityperson + ETHZ)
python3 tools/train.py -f exps/example/mot/yolox_x_mix_det.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth

# Train MOT20 test model (MOT20 + CrowdHuman)
python3 tools/train.py -f exps/example/mot/yolox_x_mix_mot20_ch.py -d 8 -b 48 --fp16 -o -c pretrained/yolox_x.pth
```

### Tracking and Evaluation
```bash
# Run ByteTrack tracking
python3 tools/track.py -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar -b 1 -d 1 --fp16 --fuse

# Run other trackers
python3 tools/track_sort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_deepsort.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse
python3 tools/track_motdt.py -f exps/example/mot/yolox_x_ablation.py -c pretrained/bytetrack_ablation.pth.tar -b 1 -d 1 --fp16 --fuse

# MOTA evaluation
python3 tools/mota.py

# NTU-MTMC evaluation
python3 tools/ntu_mota.py

# Interpolation post-processing
python3 tools/interpolation.py
```

### Demo
```bash
# Standard demo
python3 tools/demo_track.py video -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result

# NTU-MTMC demo (single camera)
PYTHONPATH=/root/ByteTrack:$PYTHONPATH python3 tools/demo_track.py video \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --path NTU-MTMC/test/Cam1/Cam1.MP4 \
  --fp16 --fuse --save_result
```

### Multi-Camera Tracking (MCT)
```bash
# Quick test with 3 cameras (first 100 frames)
python3 tools/demo_multi_camera_track.py --quick_test --cameras Cam1,Cam2,Cam3

# Full processing with all cameras
python3 tools/demo_multi_camera_track.py \
  --cameras all \
  --save_video \
  --save_results \
  --progress_bar

# Custom camera selection
python3 tools/demo_multi_camera_track.py \
  --cameras Cam1,Cam2,Cam3,Cam4,Cam5 \
  --save_video \
  --save_results \
  --progress_bar

# Performance optimized
python3 tools/demo_multi_camera_track.py \
  --cameras all \
  --fp16 \
  --fuse \
  --save_results

# Custom model configuration
python3 tools/demo_multi_camera_track.py \
  -f exps/example/mot/yolox_m_mix_det.py \
  -c pretrained/bytetrack_m_mot17.pth.tar \
  --cameras Cam1,Cam2,Cam3 \
  --save_video \
  --save_results

# With Fast-Reid integration
python3 tools/demo_multi_camera_track.py \
  --cameras Cam1,Cam2,Cam3 \
  --reid_config path/to/fast_reid_config.yml \
  --reid_model path/to/fast_reid_model.pth \
  --save_results
```

### Data Preparation
```bash
# Convert datasets to COCO format
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_mot20_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
python3 tools/convert_cityperson_to_coco.py
python3 tools/convert_ethz_to_coco.py

# Mix datasets for training
python3 tools/mix_data_ablation.py
python3 tools/mix_data_test_mot17.py
python3 tools/mix_data_test_mot20.py

# Organize NTU dataset
python3 organize_ntu_dataset.py
```

## Architecture Overview

### Core Tracking System
- **BYTETracker** (`yolox/tracker/byte_tracker.py`): Main tracking algorithm that associates every detection box, not just high-confidence ones
- **STrack** (`yolox/tracker/byte_tracker.py`): Single track representation with Kalman filter state
- **Matching** (`yolox/tracker/matching.py`): IoU distance, embedding distance, and cost matrix functions

### Multi-Camera Tracking System
- **MultiCameraTracker** (`yolox/tracker/multi_camera_tracker.py`): Coordinates multiple BYTETracker instances for cross-camera tracking
- **GlobalTrackManager** (`yolox/tracker/global_track_manager.py`): Manages global track IDs and cross-camera associations
- **CrossCameraAssociator** (`yolox/tracker/cross_camera_association.py`): Handles complex cross-camera association logic
- **ReidExtractor** (`yolox/tracker/reid_extractor.py`): Fast-Reid integration for appearance feature extraction

### Detection Model
- **YOLOX** (`yolox/models/yolox.py`): Object detection model with FPN backbone
- **YOLOPAFPN** (`yolox/models/yolo_pafpn.py`): Path Aggregation Feature Pyramid Network
- **YOLOXHead** (`yolox/models/yolo_head.py`): Detection head with classification and regression

### Alternative Trackers
- **SORT** (`yolox/sort_tracker/sort.py`): Simple Online and Realtime Tracking
- **DeepSORT** (`yolox/deepsort_tracker/deepsort.py`): SORT with deep appearance features
- **MOTDT** (`yolox/motdt_tracker/motdt_tracker.py`): MOT with Detection and Tracking

### Experiment System
- **BaseExp** (`yolox/exp/base_exp.py`): Base experiment configuration class
- **Exp** (`yolox/exp/yolox_base.py`): YOLOX-specific experiment configuration
- **Configurations** (`exps/`): Pre-defined experiment configs for different models and datasets

### Data Pipeline
- **MOTDataset** (`yolox/data/datasets/mot.py`): Dataset loader for MOT format
- **MosaicDetection** (`yolox/data/datasets/mosaicdetection.py`): Mosaic data augmentation
- **TrainTransform/ValTransform** (`yolox/data/data_augment.py`): Data augmentation pipelines

### Evaluation
- **MOTEvaluator** (`yolox/evaluators/mot_evaluator.py`): MOT metrics evaluation (MOTA, IDF1, etc.)
- **COCOEvaluator** (`yolox/evaluators/coco_evaluator.py`): COCO-style detection evaluation

## Key Components

### Tracking Algorithm
The ByteTrack algorithm works by:
1. Associating high-confidence detections with existing tracks
2. Using low-confidence detections to recover lost tracks
3. Applying Kalman filter for motion prediction
4. Using IoU distance for data association

### Model Variants
- **Standard models**: yolox_x, yolox_l, yolox_m, yolox_s (different sizes)
- **Light models**: yolox_nano, yolox_tiny (optimized for speed)
- **Specialized configs**: MOT17, MOT20, ablation studies

### Dataset Support
- **MOT17/MOT20**: Standard MOT Challenge datasets
- **CrowdHuman**: Dense crowd detection dataset
- **Cityperson**: Person detection in city scenes
- **ETHZ**: European multi-object tracking dataset
- **NTU-MTMC**: Multi-target multi-camera tracking dataset

### Deployment Options
- **ONNX**: Export to ONNX format (`deploy/ONNXRuntime/`)
- **TensorRT**: GPU optimization (`deploy/TensorRT/`)
- **ncnn**: Mobile deployment (`deploy/ncnn/`)
- **DeepStream**: NVIDIA streaming platform (`deploy/DeepStream/`)

## NTU-MTMC Specific Workflow

### Single Camera Tracking
1. Organize dataset: `python3 organize_ntu_dataset.py`
2. Run tracking per camera: Use `tools/demo_track.py` with camera-specific paths
3. Organize results: `python3 organize_ntu_results.py`
4. Evaluate: `python3 tools/ntu_mota.py`

### Multi-Camera Tracking (MCT)
1. **Quick Start**: `python3 tools/demo_multi_camera_track.py --quick_test --cameras Cam1,Cam2,Cam3`
2. **Full Processing**: `python3 tools/demo_multi_camera_track.py --cameras all --save_video --save_results --progress_bar`
3. **Output Structure**:
   ```
   MCT_outputs/
   ├── tracking_results/
   │   ├── Cam1.txt          # MOT format results per camera
   │   ├── Cam2.txt
   │   └── ...
   ├── cross_camera_associations.txt  # Cross-camera associations
   ├── Cam1_mct_output.mp4   # Visualization videos
   └── ...
   ```

### MCT Command Line Arguments
- `--cameras`: Camera selection (comma-separated or 'all')
- `--quick_test`: Process first 100 frames only
- `--save_video`: Save visualization videos
- `--save_results`: Save MOT format tracking results
- `--progress_bar`: Show progress during processing
- `--fp16`: Use half-precision for faster inference
- `--fuse`: Fuse model layers for optimization
- `--reid_config`: Path to Fast-Reid config file
- `--reid_model`: Path to Fast-Reid model file

## Common File Locations
- **Pretrained models**: `pretrained/`
- **Output results**: `YOLOX_outputs/`
- **Dataset structure**: `datasets/mot/`
- **Tracking results**: `YOLOX_outputs/*/track_results/`

## System Prompt Rules
```shell
### Role definition ###
You are a senior MLE from FAANG company and I'm a junior from NTUEE who is doing code review

### Task definition ###
Think hard to solve my task and follow the following reminders, clean code rules and problem solving rules.
use Context7. use Serena
Use a more precise search_replace tool, which allows AI to perform precise string replacement.

### Reminders ###
1. You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

2. If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

3. You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

### Clean Code Rules ###

Code is clean if it can be understood easily – by everyone
 on the team. Clean code can be read and enhanced by a developer other
than its original author. With understandability comes readability,
changeability, extensibility and maintainability.

---

**General rules**

1. Follow standard conventions.
2. Keep it simple stupid. Simpler is always better. Reduce complexity as much as possible.
3. Boy scout rule. Leave the campground cleaner than you found it.
4. Always find root cause. Always look for the root cause of a problem.

**Design rules**

1. Keep configurable data at high levels.
2. Prefer polymorphism to if/else or switch/case.
3. Separate multi-threading code.
4. Prevent over-configurability.
5. Use dependency injection.
6. Follow Law of Demeter. A class should know only its direct dependencies.

**Understandability tips**

1. Be consistent. If you do something a certain way, do all similar things in the same way.
2. Use explanatory variables.
3. Encapsulate boundary conditions. Boundary conditions are hard to keep track of. Put the processing for them in one place.
4. Prefer dedicated value objects to primitive type.
5. Avoid logical dependency. Don't write methods which works correctly depending on something else in the same class.
6. Avoid negative conditionals.

**Names rules**

1. Choose descriptive and unambiguous names.
2. Make meaningful distinction.
3. Use pronounceable names.
4. Use searchable names.
5. Replace magic numbers with named constants.
6. Avoid encodings. Don't append prefixes or type information.

**Functions rules**

1. Small.
2. Do one thing.
3. Use descriptive names.
4. Prefer fewer arguments.
5. Have no side effects.
6. Don't use flag arguments. Split method into several independent methods that can be called from the client without the flag.

**Comments rules**

1. Always try to explain yourself in code.
2. Don't be redundant.
3. Don't add obvious noise.
4. Don't use closing brace comments.
5. Don't comment out code. Just remove.
6. Use as explanation of intent.
7. Use as clarification of code.
8. Use as warning of consequences.

**Source code structure**

1. Separate concepts vertically.
2. Related code should appear vertically dense.
3. Declare variables close to their usage.
4. Dependent functions should be close.
5. Similar functions should be close.
6. Place functions in the downward direction.
7. Keep lines short.
8. Don't use horizontal alignment.
9. Use white space to associate related things and disassociate weakly related.
10. Don't break indentation.

**Objects and data structures**

1. Hide internal structure.
2. Prefer data structures.
3. Avoid hybrids structures (half object and half data).
4. Should be small.
5. Do one thing.
6. Small number of instance variables.
7. Base class should know nothing about their derivatives.
8. Better to have many functions than to pass some code into a function to select a behavior.
9. Prefer non-static methods to static methods.

**Tests**

1. One assert per test.
2. Readable.
3. Fast.
4. Independent.
5. Repeatable.

**Code smells**

1. Rigidity. The software is difficult to change. A small change causes a cascade of subsequent changes.
2. Fragility. The software breaks in many places due to a single change.
3. Immobility. You cannot reuse parts of the code in other projects because of involved risks and high effort.
4. Needless Complexity.
5. Needless Repetition.
6. Opacity. The code is hard to understand.


### Problem solving rule ###
Based on Oliver Crook's "Ten Simple Rules for Solving Any Problem," here are
the key problem-solving principles in concise bullet points:

- **Rule 1: Know That You Are Stuck**
    - Recognize when you're facing a problem
    - Identify signs of being stuck (not knowing how to proceed, repeatedly trying without progress)
    - Acknowledge the solution isn't immediate
    - State explicitly that you're stuck
- **Rule 2: Understand The Problem**
    - Rephrase the problem in simpler language
    - Draw pictures or diagrams
    - Check if you have enough information
    - Refactor the problem structure for better understanding
- **Rule 3: Picture The Solution**
    - Visualize what the solution might look like
    - Use this visualization to work backward
    - Recognize when you've found the answer
- **Rule 4: Simplifying And Specialising**
    - Break down complex problems into smaller steps
    - Start with simpler versions of the problem
    - Consider specific scenarios or special cases
- **Rule 5: Think Like A Scientist**
    - Generate and rank hypotheses by plausibility
    - Start with the simplest explanations
    - Eliminate unreasonable possibilities
    - Create and execute a plan
- **Rule 6: Guess And Check**
    - Try something when stuck, even with little confidence
    - Be comfortable with feeling uncomfortable
    - Check if your attempt helped you learn anything new
- **Rule 7: Auxiliary Problems**
    - Look for analogous or related problems
    - Consider variations that might be easier to solve
    - Draw on existing solutions from similar problems
- **Rule 8: Go For A Walk**
    - Change your environment to shift your thinking
    - Allow yourself to be distracted
    - Explain your problem to someone else
- **Rule 9: Meditate**
    - Manage emotions like frustration and anger
    - Reframe negative emotions positively
    - Practice mindfulness to maintain clarity
- **Rule 10: Reflection**
    - Review your problem-solving process
    - Identify what triggered breakthrough moments
    - Analyze which strategies worked and which didn't
    - Learn to predict effective strategies for future problems
```