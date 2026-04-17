# Age and Gender Detection: A Blurring-Based Approach

## Overview
This project predicts **gender** and **age range** from facial images using pre-trained OpenCV/Caffe models, and studies how image blurring affects prediction confidence.

The work is implemented in the notebook:
- `Age_and_Gender_Detection_A_Blurring_Based_Approach.ipynb`

## Objectives
1. Build a system for face-based age and gender prediction.
2. Compare predictions **before** and **after** blurring.
3. Analyze whether Gaussian + bilateral blurring improves confidence for low-confidence cases.

## Models and Techniques
### Face Detection
- OpenCV DNN face detector:
  - `opencv_face_detector.pbtxt`
  - `opencv_face_detector_uint8.pb`

### Age Prediction
- Caffe model:
  - `age_deploy.prototxt`
  - `age_net.caffemodel`

### Gender Prediction
- Caffe model:
  - `gender_deploy.prototxt`
  - `gender_net.caffemodel`

### Blurring
- Gaussian Blur: `cv.GaussianBlur(image, (5, 5), 0)`
- Bilateral Filter: `cv.bilateralFilter(image, 9, 75, 75)`

## Notebook Workflow
The notebook runs in a Google Colab-style setup and follows these main steps:

1. Clone source repository:
   - `git clone https://github.com/misbah4064/age_and_gender_detection.git`
2. Download model weights archive via `gdown`.
3. Unzip model files into `modelNweight/`.
4. Upload one or more images using Colab file upload.
5. Detect faces and predict age/gender:
   - Before blurring
   - After blurring
6. Display result images and comparison tables.
7. Save outputs:
   - Annotated images in `output_images/`
   - CSV summaries:
     - `age_gender_before_blurring.csv`
     - `age_gender_after_blurring.csv`

## Dependencies
The notebook uses:
- Python 3.x
- OpenCV (`cv2`)
- pandas
- gdown
- unzip utility
- Google Colab helpers:
  - `google.colab.files`
  - `google.colab.patches.cv2_imshow`

## Running in Google Colab
1. Open the notebook in Colab.
2. Run cells in order.
3. Upload test images when prompted.
4. Review before/after prediction outputs and generated CSV files.

## Running Locally (Notes)
The notebook is Colab-oriented (`google.colab` imports and upload/display APIs). For local execution, replace:
- `files.upload()` with local file path loading
- `cv2_imshow(...)` with `cv.imshow(...)`, Matplotlib, or file writes

Also ensure model files are available in:
- `modelNweight/`

## Output Interpretation
Each detected face receives:
- Predicted gender
- Predicted age bracket
- Confidence-based accuracy metric (average of age and gender confidence)

The second objective repeatedly applies blur until confidence crosses a threshold (85%) for challenging cases, then compares results.
