# 🐾 WildScan

WildScan is an AI-powered wildlife image detection and classification system.  
It combines **MegaDetector v5** (YOLOv5-based animal detector) with a variety of trained classification models, including our custom **ScratchResNet**, to identify species from camera trap images.

This project was developed by **Tyler Clinscales**, **Geoffrey Fadera**, and **Edwin Merchan** as part of the University of San Diego Applied Artificial Intelligence program.

---

## 📖 Background

WildScan was built to address a common challenge in wildlife conservation:  
how to efficiently process **large-scale camera trap datasets** without manually reviewing every image.  
By combining **state-of-the-art object detection** with **species-specific classification**,  
WildScan helps researchers:
- Quickly identify animals from millions of images.
- Reduce human error and processing time.
- Generate consistent, reproducible results.

---

## ⚙️ Tech Stack

- **PyTorch** – Deep learning framework for model training & inference  
- **YOLOv5 (MegaDetector v5)** – Animal detection model  
- **ScratchResNet** – Custom ResNet-inspired architecture trained from scratch  
- **Gradio** – Web-based user interface for live predictions  
- **Pandas / NumPy** – Data processing  
- **Pillow / torchvision** – Image preprocessing and augmentation  

---

## 🚀 Key Features

- **Automatic Animal Detection** – Locates wildlife in an image before classification.  
- **Species Identification** – Classifies animals into trained categories with confidence scores.  
- **Location-based Splits** – Evaluates generalization to new environments.  
- **Interactive Demo** – Upload an image and get instant detection + classification results.  
- **Extensible Pipeline** – Swap or upgrade models with minimal code changes.  

---

## 📊 Results (Teaser)

- **ScratchResNet Validation Accuracy:** ~74% on location-based split  
- Strong performance on visually distinct species (e.g., cat, coyote, raccoon)  
- Main confusions occur between visually similar species (e.g., fox vs coyote)  

Full performance details are available in the accompanying report.

---

## 🔮 Future Improvements

- Expand training to full 250K+ image Caltech dataset  
- Integrate **video inference** for continuous monitoring  
- Enhance augmentations with **CutMix**, **RandAugment**, and **Test-Time Augmentation**  
- Explore attention-based architectures for fine-grained classification  
- Add **bulk upload** and **automated reporting** to the demo app  

---

## 🚀 Running the Demo

A ready-to-use **WildScan Demo** is available in the `demo/` folder.  
It includes a Gradio interface for testing MegaDetector + ScratchResNet on your own images.

➡ **For setup and usage instructions, see**: [`demo/README.md`](demo/README.md)

---

## 📜 Credits

Developed by:
- **Tyler Clinscales**
- **Geoffrey Fadera**
- **Edwin Merchan**

University of San Diego — *WildScan*
