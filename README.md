# AI Methods for Classification of Production Errors on Camera Images of Additive Manufacturing Processes

> **A Systematic Review** | Seminar Paper & Presentation  
> Universität Koblenz · Web and Data Science · February 2026

---

## Table of Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Research Objectives](#research-objectives)
- [Background](#background)
  - [Additive Manufacturing Processes](#additive-manufacturing-processes)
  - [Common AM Defects](#common-am-defects)
  - [Camera-Based Monitoring in AM](#camera-based-monitoring-in-am)
- [AI Methods Reviewed](#ai-methods-reviewed)
  - [Traditional Image Processing](#1-traditional-image-processing)
  - [Classical Machine Learning](#2-classical-machine-learning)
  - [Deep Learning Approaches](#3-deep-learning-approaches)
- [Comparative Analysis](#comparative-analysis)
- [Key Results & Case Studies](#key-results--case-studies)
- [Strengths & Weaknesses](#strengths--weaknesses)
- [Challenges & Future Research Directions](#challenges--future-research-directions)
- [Key Findings & Conclusions](#key-findings--conclusions)
- [Methodology](#methodology)
- [Repository Contents](#repository-contents)
- [References](#references)
- [Author](#author)
- [Supervisor](#supervisor)

---

## Overview

This repository contains all materials produced for the seminar paper and presentation on **AI-based defect detection in Additive Manufacturing (AM)**. The work conducts a systematic review of state-of-the-art artificial intelligence approaches for classifying production errors from camera images captured during the AM build process — covering publications from 2018 to 2025.

The review spans three generations of AI methods: traditional image processing, classical machine learning, and modern deep learning architectures (CNNs, YOLO-based object detectors, and semantic segmentation networks). It evaluates these approaches not only on classification accuracy but also on practical deployment factors including dataset requirements, computational cost, real-time capability, and robustness to domain shift.

**Keywords:** `Additive Manufacturing` · `Defect Detection` · `Deep Learning` · `Convolutional Neural Networks` · `YOLO` · `Computer Vision` · `In-Situ Monitoring` · `Transfer Learning` · `Quality Assurance`

---

## Abstract

Additive Manufacturing (AM) has emerged as a transformative production technology, enabling complex geometries and design freedom beyond conventional manufacturing methods. However, AM processes are inherently susceptible to defects such as porosity, melt-pool irregularities, and layer delamination, which undermine part quality and reliability.

This seminar paper systematically reviews state-of-the-art AI approaches for camera-based defect classification in AM, encompassing traditional image processing techniques, classical machine learning methods, and deep learning architectures.

The reviewed literature indicates that deep learning approaches — particularly Convolutional Neural Networks (CNNs) and YOLO-based object detection models — consistently outperform traditional and classical methods in defect classification and localization under controlled experimental conditions, with classification accuracy improvements from 65–75% (traditional methods) up to 92–97% (deep learning). Nevertheless, challenges including limited availability of annotated datasets, domain shift across machines and materials, limited interpretability, and integration complexity continue to constrain industrial deployment.

This review identifies key research gaps and outlines directions for advancing AI-driven defect detection to improve quality assurance and support wider industrial adoption of additive manufacturing.

---

## Research Objectives

### Primary Objective
Critically review AI methods for classifying production errors from camera images in AM, focusing on deep learning approaches applied to real-time visual data. The scope encompasses CNN architectures, YOLO-based object detection, and semantic segmentation networks, examining publications from 2018–2025.

### Secondary Objective
Compare AI techniques systematically across:
- Classification accuracy and detection precision
- Computational requirements and inference speed
- Data annotation burden and dataset requirements
- Robustness to domain shift across machines and materials

### Tertiary Objective
Synthesize insights highlighting open research directions for advancing AI-enabled visual monitoring — particularly regarding generalization across systems, dataset diversity, and industrial integration.

---

## Background

### Additive Manufacturing Processes

Additive Manufacturing builds components through layer-by-layer material deposition from digital models. Despite differences in mechanism, all AM processes involve repeated thermal cycling and layer-wise bonding, introducing complex thermo-mechanical interactions that make processes susceptible to defect formation.

| AM Process | Thermal Characteristics | Resolution | Key Characteristics |
|---|---|---|---|
| **Material Extrusion (FDM)** | Heated thermoplastic extrusion | Moderate | Molten thermoplastic deposited via nozzle |
| **Powder Bed Fusion (SLM/SLS)** | High-temperature melting with rapid cooling | High | Laser-based melting producing dense metal parts |
| **Vat Photopolymerization** | UV or laser-induced curing | High lateral | Excellent surface finish; shrinkage challenges |
| **Directed Energy Deposition** | Focused energy melting | Variable | Repair and large component fabrication |

**Key application sectors:** Aerospace (lightweight structures, complex components), Automotive (rapid prototyping, customized parts), Biomedical (patient-specific implants, prosthetics).

---

### Common AM Defects

Defects arise from unstable process parameters, non-uniform energy distribution, material inconsistencies, and insufficient interlayer adhesion. Their visual manifestations captured through camera imaging are the primary input to AI classification systems.

| Defect Type | Physical Characteristics | Visual Signature | Structural Impact |
|---|---|---|---|
| **Porosity** | Spherical/irregular voids (µm to 100s µm) | Dark circular/irregular voids | Reduces strength and fatigue resistance significantly |
| **Lack of Fusion** | Incomplete bonding between tracks/layers | Irregular bonding patterns | Leaves weak interfaces in the material |
| **Surface Roughness** | Striated patterns/textured appearances | Textured/striated patterns | Affects fatigue performance, indicates process instability |
| **Layer Shifting** | Spatial discontinuities between layers | Visible spatial discontinuities | Compromises dimensional accuracy |
| **Delamination** | Separation between layers | Separation gaps/cracks | Structural failure risk |
| **Dimensional Error** | Deviations from CAD specifications | Detectable through edge detection | Functional incompatibility |
| **Thermal Distortion** | Warping/curvature due to residual stresses | Visible geometric deformation | Geometric non-conformance |

---

### Camera-Based Monitoring in AM

In-situ process monitoring has become central to AM research as manufacturers transition from reactive post-process inspection (e.g., CT scanning, destructive sectioning) toward proactive real-time quality control.

**Why camera-based systems?**
- Non-contact and non-destructive observation
- Relatively low cost compared to other sensor types
- High-resolution spatial and temporal information capture
- Enables early process instability identification
- Potential for real-time corrective actions during build

**Camera modalities used in practice:**

| Modality | Capabilities | Relative Cost | Limitations |
|---|---|---|---|
| **2D RGB Camera** | Surface geometry, layer completeness | Low | Lighting sensitivity, no depth information |
| **Thermal / IR Imaging** | Temperature distribution, melt pool dynamics | Medium–High | Calibration required, emissivity sensitivity |
| **Coaxial Imaging** | Direct melt pool observation | Variable | Limited field of view |
| **3D Imaging** | Geometric reconstruction, depth information | High | Higher computation, lower temporal resolution |

**Key challenges** in camera-based monitoring: lighting variation, metallic surface reflections creating glare, motion blur, sensor noise, depth-of-field limitations, and environmental factors. Raw images require automated AI analysis to distinguish real defects from imaging artifacts.

---

## AI Methods Reviewed

### 1. Traditional Image Processing

**Methodology:** Sequential deterministic pipeline — image acquisition → preprocessing (grayscale conversion, noise reduction, contrast enhancement) → feature extraction (edges, contours, texture statistics, geometric properties) → threshold-based classification.

**Common techniques:** Binary thresholding, morphological operations (erosion, dilation), edge detection algorithms (Sobel, Canny).

**Performance Profile:**

| Metric | Value |
|---|---|
| Typical accuracy | 65–75% |
| Processing speed | >100 fps on CPU |
| Interpretability | High (fully transparent rules) |
| Data requirements | None (rule-based) |
| Generalization | Low — sensitive to lighting & environment |

**Key limitation:** Requires frequent manual recalibration. Fixed rules that work in controlled lab conditions fail in real manufacturing environments where conditions vary. Accuracy of 65–75% is insufficient for safety-critical applications.

---

### 2. Classical Machine Learning

**Methodology:** Data-driven classification using manually engineered visual features (texture descriptors, histograms, frequency-domain features, geometric measurements) fed into supervised learning algorithms.

**Algorithms reviewed:** Support Vector Machines (SVM with RBF kernel), Random Forests, Gradient Boosting Machines, k-Nearest Neighbor.

| Method | Performance Trend | Characteristics |
|---|---|---|
| **SVM (RBF kernel)** | Moderate accuracy (80–87%) | Multi-class defect categorization in powder bed fusion |
| **Random Forest** | Moderate to high accuracy | Interpretable feature importance rankings |
| **Gradient Boosting** | Effective for non-linear relationships | Sensitive to feature engineering quality and dataset size |
| **k-Nearest Neighbor** | Performance dependent on feature scaling | Computationally inefficient for large datasets |

**Performance Profile:**

| Metric | Value |
|---|---|
| Typical accuracy | 80–87% |
| Processing speed | 20–50 fps |
| Interpretability | Moderate (feature importance available) |
| Data requirements | Labeled examples needed |
| Generalization | Limited — features optimized for one process often fail on others |

**Key limitation:** Manual feature engineering creates performance ceilings. Features optimized for one AM process or material often fail on others due to defect morphology differences. Models struggle with complex morphologies and non-stationary process conditions.

---

### 3. Deep Learning Approaches

Deep learning eliminates manual feature engineering through end-to-end learning of hierarchical feature representations directly from raw images.

#### Convolutional Neural Networks (CNNs)

CNNs learn spatial filters through backpropagation — convolutional layers extract 2D spatial patterns, pooling layers reduce dimensionality, and activation functions introduce non-linearity enabling complex pattern recognition.

**Popular architectures reviewed:**
- **ResNet** — Residual connections enabling stable training of 50+ layer networks
- **VGG** — Sequential convolutional blocks; widely used baseline
- **MobileNet** — Optimized for resource-constrained and edge hardware

**Performance:** Classification accuracy frequently exceeds 92–96%, outperforming traditional methods by 20–30 percentage points and classical ML by 10–15 points.

#### YOLO-Based Object Detection

YOLO (You Only Look Once) frameworks enable **simultaneous localization and classification** — predicting bounding box coordinates and class probabilities in a single forward pass, making them uniquely suited for real-time in-situ monitoring.

- **YOLOv5 and YOLOv8** demonstrate >96–97% mean average precision (mAP) at 11–15 fps on GPU hardware
- Provides both defect category labels and precise spatial coordinates
- Single-stage regression approach — the image is processed once, not multiple times

#### Semantic Segmentation (U-Net)

Pixel-level classification providing precise defect boundary delineation, achieving 93–95% Intersection over Union (IoU). Particularly useful when defect size, shape, and spatial extent are critical quality indicators.

#### Transfer Learning

Pre-trained models (e.g., from ImageNet) adapted to AM defect detection:
- Reduces labeled data requirements by **10–50×**
- Accelerates training convergence significantly
- Enables strong performance with datasets of hundreds rather than thousands of examples

**Performance Profile (Deep Learning):**

| Metric | Value |
|---|---|
| Typical accuracy (CNN) | 92–97% |
| Typical mAP (YOLO) | >90–97% |
| Processing speed (YOLO) | 11–15 fps on GPU |
| Interpretability | Low (black-box) |
| Data requirements | 500–several thousand labeled images |

---

## Comparative Analysis

| Approach | Typical Accuracy | Speed | Interpretability | Data Required | Domain Generalization |
|---|---|---|---|---|---|
| **Traditional Image Processing** | 65–75% | >100 fps (CPU) | High | None | Very Low |
| **Classical ML (SVM, RF)** | 80–87% | 20–50 fps | Moderate | Labeled samples | Low |
| **Deep Learning (CNN)** | 92–97% | Moderate | Low | 500–several thousand images | Moderate |
| **Object Detection (YOLO)** | >90–97% mAP | 11–15 fps (GPU) | Low | Large annotated dataset | Moderate |

**Accuracy progression:** Traditional methods → Classical ML → Deep Learning represents a step-wise improvement of approximately 15–22 percentage points at each stage, for a total improvement of ~30 percentage points from 65–75% to 92–97%.

**The core trade-off:** As accuracy improves, data requirements and computational cost increase, and interpretability decreases. Optimal approach depends on application constraints: available training data, computational resources, interpretability requirements, and accuracy-speed trade-offs.

---

## Key Results & Case Studies

Five representative studies from the reviewed literature demonstrate the practical performance of AI methods on real AM imaging data:

| Study | Dataset Size | Imaging Modality | Method | Key Performance |
|---|---|---|---|---|
| **Wang et al.** | >3,500 annotated instances | RGB | YOLOv8 | mAP@0.5 >90%, overall mAP >97% across 4 defect categories, ~15 fps |
| **Baumgartl et al.** | Not specified | Thermal IR (off-axis) | Deep learning classifier | Real-time defect identification with k-fold cross-validation |
| **Jia et al.** | Hundreds of images | RGB | Transfer learning (ResNet, VGG) | 10–15% accuracy improvement vs. training from scratch (ceramic AM) |
| **Smoqi et al.** | Process-specific | Thermal/melt-pool images | Physics-informed ML + CNN | F1 >0.95 (physics features); F1 0.89–0.97 (CNN) |
| **Yang et al.** | CFD simulation + experimental | Thermal imaging (>100 Hz) | Hybrid neural network | Exceeded performance of either data source alone |

### Highlighted Study: Wang et al. — YOLOv8 for Real-Time Detection

This study is particularly significant for industrial relevance:
- Trained on **>3,500 manually annotated instances** across four defect categories
- Achieved **mAP@0.5 >90%** and overall **mAP >97%**
- Processed at approximately **15 frames per second** on GPU hardware — compatible with continuous in-situ monitoring
- Demonstrates that real-time, high-accuracy defect detection with spatial localization is practically achievable with current hardware

### Highlighted Study: Jia et al. — Transfer Learning with Small Datasets

Validated the practical value of transfer learning for AM applications:
- Achieved strong classification performance using only **hundreds of images** (not thousands)
- Transfer learning from ResNet/VGG pre-trained on ImageNet improved accuracy by **10–15 percentage points** versus training from scratch
- Particularly valuable for ceramic AM, where labeled defect data is scarce

---

## Strengths & Weaknesses

### Strengths of Deep Learning Methods

- **High classification accuracy** (92–97%) when trained on well-curated datasets — outperforming traditional techniques by 20–30 percentage points in accuracy
- **Automatic feature extraction** from raw images without manual feature engineering — particularly valuable in AM where defect patterns are complex and process-dependent
- **Multi-scale architectures** capture both local defect features (individual pores) and global process variations across entire build layers
- **Real-time detection** demonstrated in controlled settings, indicating potential for in-situ monitoring and immediate process intervention
- **Transfer learning** substantially reduces data requirements and makes deep learning practical when large labeled datasets are unavailable

### Weaknesses & Deployment Barriers

- **Dataset dependency:** Strong dependence on the size, quality, and representativeness of training data. Models may achieve high validation accuracy on same-distribution data but show reduced performance on unseen parameters, material batches, or underrepresented defect types
- **Domain shift:** Limited transferability across machines, materials, and sensing setups. Variations in camera resolution, illumination, and thermal signatures significantly alter defect appearance. Retraining or fine-tuning is often required for new AM systems from different vendors — reducing practicality for large-scale heterogeneous fleet deployment
- **Black-box interpretability:** With millions of learned parameters, it is difficult to understand which image features drive classifications or diagnose misclassifications. This challenges debugging, regulatory validation, and industrial acceptance — particularly in aerospace and medical device manufacturing
- **Computational requirements:** Training demands GPU resources and can take hours to weeks. Deployment on resource-constrained embedded or edge platforms remains challenging

---

## Challenges & Future Research Directions

### Critical Challenges Limiting Industrial Adoption

**1. Dataset Scarcity**  
Scarcity of large-scale, publicly available, well-annotated defect image datasets is one of the most significant barriers. Manual annotation is labor-intensive and requires expert knowledge for accurate ground-truth labels — particularly problematic for deep learning methods requiring thousands of examples. Emerging approaches include synthetic data generation through physics-based simulation and semi-supervised learning leveraging large unlabeled datasets.

**2. Domain Shift & Cross-System Generalization**  
Variations in camera hardware, illumination, imaging geometry, material properties, and process parameters substantially alter visual defect appearance — causing trained models to degrade on different equipment or materials. Particularly problematic for industrial environments with multiple AM systems from different vendors. Domain adaptation techniques (adversarial training, few-shot learning) and multi-source pre-training show promise.

**3. Real-Time Edge Deployment**  
Real-time in-situ monitoring requires inference speeds compatible with high-frequency image acquisition. While YOLO-based models demonstrate real-time performance on GPU platforms in controlled settings, deployment on resource-constrained embedded or edge platforms remains challenging. Model optimization techniques (quantization, pruning) are actively explored.

**4. Interpretability & Regulatory Compliance**  
Safety-critical applications in aerospace and medical device manufacturing require decision transparency for regulatory validation. Current deep learning models operate largely as black boxes. Attention visualization, gradient-based saliency mapping, and model-agnostic explanation techniques (LIME, SHAP) offer partial solutions, but complete interpretability remains unresolved.

---

### Future Research Roadmap

```
SHORT-TERM (1–2 years)
├── Establish standardized benchmark datasets with comprehensive annotations
│   covering multiple imaging modalities, materials, and defect types
├── Develop lightweight model architectures optimized for edge deployment
└── Prioritize domain adaptation methodologies for cross-machine generalization

MEDIUM-TERM (2–5 years)
├── Integrate AI detection with closed-loop process control architectures
│   enabling automated corrective actions based on real-time quality feedback
├── Develop self-supervised learning approaches leveraging large volumes of
│   unlabeled image data generated during normal manufacturing operations
└── Multi-modal sensor fusion combining visual, thermal, and acoustic sensing

LONG-TERM (> 5 years)
├── Physics-informed neural networks integrating heat transfer, fluid dynamics,
│   and solidification phenomena into model structures
└── Fully autonomous quality assurance frameworks — lights-out manufacturing
    scenarios where AM systems continuously optimize their own operation
```

---

## Key Findings & Conclusions

1. **Deep learning is the clear performance leader.** CNN architectures and YOLO-based object detection consistently outperform traditional image processing and classical ML methods in accuracy, detection precision, and robustness — with classification accuracy improvements from 65–75% (rule-based) to 92–97% (deep learning).

2. **YOLO-based detection is most promising for industrial deployment.** Single-stage detectors achieve >90% mAP while maintaining inference speeds of 11–15 fps — compatible with real-time in-situ monitoring and enabling both defect classification and precise spatial localization within practical computational budgets.

3. **Transfer learning makes deep learning accessible.** Pre-trained models reduce labeled data requirements by 10–50×, enabling strong performance even with domain-specific datasets limited to hundreds of annotated examples.

4. **Lab accuracy does not equal industrial readiness.** Despite impressive controlled-condition results, persistent challenges — dataset scarcity, domain shift, black-box interpretability, and edge deployment constraints — continue to limit broad industrial adoption.

5. **The path forward requires coordinated effort** across benchmark dataset creation, domain adaptation research, computational optimization for edge hardware, and physics-informed model development.

> The technology works in controlled research settings. The challenge now is bridging the gap to robust, generalizable, and explainable industrial deployment.

---

## Methodology

### Literature Search

A structured search strategy identified peer-reviewed research combining AM processes, image-based monitoring, and machine learning / deep learning. Primary databases searched: **SpringerLink, ScienceDirect, IEEE Xplore, MDPI, IOPscience**.

**Search queries (examples):**
- `"additive manufacturing" AND "defect detection" AND "deep learning"`
- `"powder bed fusion" AND "camera-based monitoring"`
- `"convolutional neural network" AND "manufacturing defects"`
- `"YOLO" AND "quality control"`
- `"computer vision" AND "layer-wise monitoring"`

**Temporal scope:** Publications from 2014–2025, with priority given to 2023–2025 publications for YOLO-based detection, transfer learning, and domain adaptation advances.

### Inclusion Criteria (All Three Required)

1. Camera/optical imaging (RGB, thermal, coaxial) as the **primary data source** during or after AM
2. Machine learning or deep learning applied for **defect detection or classification**
3. **Quantitative evaluation metrics** (accuracy, mAP, IoU, F1-score) reported on real or experimental AM data

### Exclusion Criteria

- Papers on simulation, design optimization, or mechanical testing without visual inspection
- Studies on non-AM inspection (conventional machining, welding)
- Conceptual papers without implementation, qualitative demonstrations, or theoretical studies without validation

### Comparison Criteria

Studies were compared across four dimensions:
- **AI approach category** — Traditional / Classical ML / Deep Learning / Hybrid
- **Dataset characteristics** — Imaging modality, material type, defect categories, dataset size, annotation method, availability (proprietary vs. public)
- **Evaluation metrics** — Accuracy, precision, recall, F1-score, mAP, IoU (interpreted cautiously given varying experimental conditions)
- **Practical relevance** — Offline vs. real-time inference, hardware requirements, deployment constraints, scalability

---

## Repository Contents

```
├── README.md                                    ← This file
├── final_SubramanianPorselvaBharathi.pdf        ← Full seminar paper (17 pages)
├── slides_Ramya_SubramanianPorselvaBharathi.pptx← Full presentation slides (18 slides)
├── Summary_Ramya_SubramanianPorselvaBharathi.pptx← Summary slides (2 slides, for discussion event)
└── Preparation_for_Screencast.docx             ← Screencast narration script (slide-by-slide)
```

### Document Descriptions

| File | Description |
|---|---|
| `final_SubramanianPorselvaBharathi.pdf` | Complete seminar paper with full methodology, theoretical background, results, and references following academic formatting standards |
| `slides_Ramya_SubramanianPorselvaBharathi.pptx` | Full 18-slide presentation covering all sections from introduction through conclusions and future directions |
| `Summary_Ramya_SubramanianPorselvaBharathi.pptx` | Condensed 2-slide summary for the seminar discussion event, designed for a 2–3 minute recap |
| `Preparation_for_Screencast.docx` | Detailed slide-by-slide narration script used for the screencast recording, with transition cues |

---

## References

The following are the 16 peer-reviewed sources forming the basis of this systematic review:

1. Deshpande, S., Venugopal, V., Kumar, M., Anand, S. (2024). Deep learning-based image segmentation for defect detection in additive manufacturing: an overview. *International Journal of Advanced Manufacturing Technology*, 134, 2081–2105. https://doi.org/10.1007/s00170-024-14191-6

2. Wang, W., Wang, P., Zhang, H., Chen, X., Wang, G., Lu, Y., Chen, M., Liu, H., Li, J. (2024). A Real-Time Defect Detection Strategy for Additive Manufacturing Processes Based on Deep Learning and Machine Vision Technologies. *Micromachines*, 15(1), 28. https://doi.org/10.3390/mi15010028

3. Jia, X., Li, S., Wang, T., Liu, B., Cui, C., Li, W., Wang, G. (2024). High-Performance Defect Detection Methods for Real-Time Monitoring of Ceramic Additive Manufacturing Process Based on Small-Scale Datasets. *Processes*, 12(4), 633. https://doi.org/10.3390/pr12040633

4. Frazier, W.E. (2014). Metal Additive Manufacturing: A Review. *Journal of Materials Engineering and Performance*, 23, 1917–1928. https://doi.org/10.1007/s11665-014-0958-z

5. Grasso, M., Colosimo, B.M. (2017). Process defects and in situ monitoring methods in metal powder bed fusion: a review. *Measurement Science and Technology*, 28(4), 044005. https://doi.org/10.1088/1361-6501/aa5c4f

6. Scime, L., Beuth, J. (2018). A multi-scale convolutional neural network for autonomous anomaly detection and classification in a laser powder bed fusion additive manufacturing process. *Additive Manufacturing*, 24, 273–286. https://doi.org/10.1016/j.addma.2018.09.034

7. Everton, S.K., Hirsch, M., Stavroulakis, P., Leach, R.K., Clare, A.T. (2016). Review of in-situ process monitoring and in-situ metrology for metal additive manufacturing. *Materials & Design*, 95, 431–445. https://doi.org/10.1016/j.matdes.2016.01.099

8. Holzmond, O., Li, X. (2017). In situ real time defect detection of 3D printed parts. *Additive Manufacturing*, 17, 135–142. https://doi.org/10.1016/j.addma.2017.08.003

9. Baumgartl, H., Tomas, J., Buettner, R., Merkel, M. (2020). A deep learning-based model for defect detection in laser-powder bed fusion using in-situ thermographic monitoring. *Progress in Additive Manufacturing*, 5(3), 277–285. https://doi.org/10.1007/s40964-019-00108-3

10. Yang, W., Qiu, Y., Liu, W., Qiu, X., Bai, Q. (2023). Defect prediction in laser powder bed fusion with the combination of simulated melt pool images and thermal images. *Journal of Manufacturing Processes*, 106, 214–222. https://doi.org/10.1016/j.jmapro.2023.10.006

11. Smoqi, Z., Gaikwad, A., Bevans, B., Kobir, M.H., Craig, J., Abul-Haj, A., Peralta, A., Rao, P. (2022). Monitoring and prediction of porosity in laser powder bed fusion using physics-informed meltpool signatures and machine learning. *Journal of Materials Processing Technology*, 304, 117550. https://doi.org/10.1016/j.jmatprotec.2022.117550

12. Razvi, S.S., Feng, S., Narayanan, A., Lee, Y.-T.T., Witherell, P. (2019). A Review of Machine Learning Applications in Additive Manufacturing. *Proceedings of the ASME 2019 IDETC-CIE*, DETC2019-98415. https://doi.org/10.1115/DETC2019-98415

13. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *2016 IEEE CVPR*, pp. 779–788. https://doi.org/10.1109/CVPR.2016.91

14. Gobert, C., Reutzel, E.W., Petrich, J., Nassar, A.R., Phoha, S. (2018). Application of supervised machine learning for defect detection during metallic powder bed fusion additive manufacturing using high resolution imaging. *Additive Manufacturing*, 21, 517–528. https://doi.org/10.1016/j.addma.2018.04.005

15. Jin, Z., Zhang, Z., Demir, K., Gu, G.X. (2020). Machine Learning for Advanced Additive Manufacturing. *Matter*, 3(5), 1541–1556. https://doi.org/10.1016/j.matt.2020.08.023

16. Aminzadeh, M., Kurfess, T.R. (2019). Online quality inspection using Bayesian classification in powder-bed additive manufacturing from high-resolution visual camera images. *Journal of Intelligent Manufacturing*, 30, 2505–2523. https://doi.org/10.1007/s10845-018-1412-0

---

## Author

**Ramya Subramanian Porselva Bharathi**  
MS — Web and Data Science  
Universität Koblenz, Koblenz, Germany  
Matriculation Number: 224201309  
📧 ramyasp@uni-koblenz.de  
🔗 [LinkedIn](https://www.linkedin.com/in/ramya-sp)

---

## Supervisor

**Jan Schmidt**  
Universität Koblenz

---

*Submitted: February 2026 · Universität Koblenz · MS Web and Data Science*
