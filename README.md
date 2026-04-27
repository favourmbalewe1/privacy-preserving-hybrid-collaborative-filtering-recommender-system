# Privacy-Preserving Hybrid Collaborative Filtering Recommender System

**Master's Dissertation Project**  
**University of the West of Scotland (UWS)**  
**School of Computing, Engineering and Physical Sciences**  
**Student:** Favour Ugochukwu Mbalewe (B01819308)  
**Month/Year:** April 2026  
**Supervisor:** Dr. Yingbo Zhu

## Abstract
E-commerce recommender systems have revolutionised online shopping by providing personalised suggestions. However, traditional collaborative filtering methods require large-scale collection of personal data, conflicting with privacy regulations such as GDPR. This project developed a **hybrid collaborative filtering recommender system** that preserves privacy by combining user-based similarity, item-based similarity, and Support Vector Machine (SVM) classification. Differential privacy (Laplace mechanism) adds calibrated noise to similarity computations, while federated learning principles minimise centralised data exposure.

The system was implemented as a **standalone desktop application with a user-friendly Tkinter GUI**, including an adjustable privacy budget (ε slider). It was tested on a merged dataset from UCI Online Retail, Retailrocket, and MovieLens.

## Key Features
- Hybrid Collaborative Filtering (user-based + item-based cosine similarity)
- SVM Classification (RBF kernel, one-vs-one) for recommendation scoring
- Differential Privacy via `diffprivlib` (Laplace mechanism) with adjustable ε
- 9-dimensional feature engineering pipeline
- Tkinter GUI with four tabs (Load Data, Train Model, Metrics, Recommend)
- Packaged as a single Windows executable using PyInstaller
- Full reproducibility with fixed random seeds

## Tech Stack
- **Language:** Python 3.10
- **ML Libraries:** scikit-learn, NumPy, Pandas, scipy
- **Privacy:** diffprivlib (IBM)
- **GUI:** Tkinter + custom dark-blue theme
- **Packaging:** PyInstaller
- **Data:** UCI Online Retail + Retailrocket + MovieLens (merged corpus)

## Repository Contents
- `prepare_dataset.py` – Data acquisition and preprocessing
- `model_pipeline.py` – HybridCFPipeline class
- GUI application files
- Trained models (`hybrid_cf_model.pkl`)
- Full dissertation PDF
- Requirements.txt and environment setup

## Results Summary
- Excellent classification performance (accuracy, precision, recall, F1 ≈ 0.99 on sampled subset)
- Privacy-utility trade-off analysis via ε-sweep
- Real-time recommendations with confidence scores
- Runs efficiently on commodity hardware

## Dissertation
Full dissertation (PDF) is included in the repository.  
GitHub: https://github.com/favourmbalewe1/privacy-preserving-hybrid-collaborative-filtering-recommender-system

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the GUI: `python main.py` (or use the pre-built `.exe`)

## License
This project is for academic and research purposes. Please cite the dissertation if used in your work.

---

**Made with ❤️ for privacy-conscious recommender systems**
