# Lane Marking Detection and Lane Condition Analysis using Semantic Segmentation and Business Intelligence

🚗 A deep learning and Business Intelligence framework that transforms lane segmentation outputs into actionable road-condition indicators for smart road safety monitoring and maintenance.

This project extends traditional lane detection by converting semantic segmentation outputs into interpretable metrics that support infrastructure monitoring and decision-making. Using a VGG16-UNet architecture, lane markings are detected from highway road images and further analyzed to compute road-quality indicators such as Lane Visibility, Infrastructure Degradation Index (IDI) and Lane Status.

Unlike conventional lane detection systems that stop at identifying lane boundaries, this framework integrates Business Intelligence dashboards and automated alert generation to support real-world road maintenance and safety management. 

---

## Problem Statement

Most lane detection systems focus primarily on:

- Detecting lane markings
- Identifying lane boundaries
- Achieving higher segmentation accuracy

However, road authorities require answers to practical questions such as:

- Which roads have deteriorating lane markings?
- Which lanes require immediate maintenance?
- Which road segments pose safety risks?
- How can lane health be monitored at scale?

Traditional manual inspection methods are:

- Time-consuming
- Subjective
- Costly
- Difficult to scale

This project addresses these limitations by combining semantic segmentation with Business Intelligence and automated road-condition assessment. 
---

## Dataset

### TuSimple Lane Detection Dataset

Dataset Statistics

| Component | Count |
|------------|------------:|
| Road Images | 3,626 |
| Binary Lane Masks | 3,626 |
| Training Images | 2,900 |
| Testing Images | 726 |
| Original Resolution | 1280 × 720 |
| Model Input Resolution | 224 × 224 |

The dataset consists of real-world highway driving scenes captured using front-facing vehicle cameras under varying road and environmental conditions. 

---

## Proposed Framework

### Stage 1: Data Preparation

- Extract representative frames from video clips
- Parse JSON lane annotations
- Generate binary ground-truth segmentation masks
- Remove invalid lane coordinates
- Create supervised learning dataset

---

### Stage 2: Image Preprocessing

- Resize images to 224 × 224
- Normalize pixel values
- Generate lane masks
- Create training and testing splits
- Batch and shuffle data for efficient training

---

### Stage 3: Lane Segmentation

A VGG16-UNet Encoder-Decoder architecture is used for semantic segmentation.

#### Encoder (VGG16)

- Feature extraction
- Edge detection
- Curve detection
- Lane boundary learning

#### Decoder (UNet)

- Feature reconstruction
- Spatial information recovery
- Lane boundary refinement
- Pixel-level classification

The model predicts whether each pixel belongs to:

- Lane
- Non-Lane (Background)

---

### Stage 4: Post Processing

The generated probability maps are converted into clean lane masks using:

- Thresholding
- Noise Removal
- Skeletonization
- Connected Component Analysis

This ensures meaningful lane structures for downstream analysis. 
---

### Stage 5: Lane Condition Assessment

Instead of stopping at segmentation, the framework extracts meaningful lane-quality indicators.

#### Lane Visibility

Measures how clearly a lane is visible within an image.

#### Infrastructure Degradation Index (IDI)

Quantifies lane deterioration.

- Low IDI - Better lane quality
- High IDI - Poor lane quality

#### Lane Status Classification

Road segments are classified as:

- GOOD
- MODERATE
- POOR
- CRITICAL

These indicators transform segmentation masks into actionable maintenance metrics. 

---

### Stage 6: Alert Generation

When lane visibility falls below predefined thresholds:

- Automatic alerts are triggered
- Geographic coordinates are attached
- Timestamp is recorded
- Authorities can be notified for maintenance

This creates a proof-of-concept smart road monitoring system. 
---

### Stage 7: Business Intelligence Integration

The generated KPIs are integrated into Microsoft Power BI.

Dashboard Features:

- Road Health Score
- Lane Visibility Analysis
- Infrastructure Degradation Index
- Road Condition Distribution
- Geospatial Monitoring
- Maintenance Insights

This enables data-driven road safety management and infrastructure planning. 

---

## Technologies Used

### Deep Learning

- TensorFlow
- Keras
- VGG16
- UNet
- CNN

### Computer Vision

- OpenCV
- Semantic Segmentation
- Image Processing

### Data Processing

- Pandas
- NumPy
- Scikit-Image

### Visualization & Analytics

- Microsoft Power BI

### Programming Language

- Python

---

## Performance Metrics

| Metric | Score |
|----------|----------:|
| Dice Coefficient | 0.747 |
| Precision | 0.734 |
| Recall | 0.765 |
| IoU | 0.603 |

The model achieved reliable segmentation performance suitable for lane-condition assessment and KPI generation.

---

## Key Contributions

✅ Lane marking semantic segmentation using VGG16-UNet

✅ Automated lane visibility computation

✅ Infrastructure Degradation Index (IDI) generation

✅ Road-condition classification framework

✅ Automated alert generation

✅ Geospatial road monitoring

✅ Business Intelligence dashboard integration

✅ Decision-support system for road maintenance

---

## Results

The proposed framework successfully:

- Detected lane markings using semantic segmentation
- Generated interpretable lane-health indicators
- Classified road conditions into GOOD, MODERATE, POOR and CRITICAL categories
- Triggered alerts for degraded road segments
- Enabled geospatial road-condition monitoring
- Integrated AI outputs into a Business Intelligence environment for practical decision-making

The framework bridges the gap between computer vision outputs and real-world infrastructure management. 

