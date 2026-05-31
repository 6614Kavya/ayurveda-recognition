# 🌿 Ayurvedic Medicinal Plant Recognition
### Machine Learning & Image Processing · University of Moratuwa FYP

**ML / Computer Vision**
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=black)

**Backend**
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-4B0082?style=flat&logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat&logo=pydantic&logoColor=white)

**Frontend**
![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)
![Axios](https://img.shields.io/badge/Axios-5A29E4?style=flat&logo=axios&logoColor=white)

**Database & Deployment**
![MongoDB](https://img.shields.io/badge/MongoDB_Atlas-47A248?style=flat&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-46E3B7?style=flat&logo=render&logoColor=black)

> A three-module computer vision system that identifies **33 Sri Lankan Ayurvedic medicinal plant species** from leaf and flower images — achieving **98.75% accuracy** using classical ML and deep learning pipelines, served via a FastAPI backend and React frontend.

---

## 📌 Overview

Sri Lanka has a rich tradition of Ayurvedic medicine (Hela Osu), yet there is no reliable automated tool to identify local medicinal plants. Misidentification of herbs can render treatments ineffective or even harmful. This research builds a machine learning system that allows anyone — from Ayurvedic practitioners to the general public — to identify a medicinal plant simply by uploading a photo of its leaf or flower.

The system is divided into three independent modules based on the type of input image:

| Module | Input | Plants Covered | Best Accuracy |
|--------|-------|---------------|---------------|
| Module 1 | Large single leaves | 18 species | **98.75%** (Random Forest) |
| Module 2 | Small / compound leaves | 15 species | **95%** (Random Forest) |
| Module 3 | Flowers | 10 species | **CNN + MobileNetV2** |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React + Vite (Frontend)               │
│         User uploads image  →  Axios POST request       │
└────────────────────────┬────────────────────────────────┘
                         │  multipart/form-data
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI + Uvicorn (Backend API)             │
│                                                         │
│  Pydantic validates file type (JPEG / PNG only)         │
│         │                                               │
│         ▼                                               │
│  Route to correct module:                               │
│  ├─ POST /predict/large-leaf  →  Module 1 (RF model)   │
│  ├─ POST /predict/small-leaf  →  Module 2 (RF model)   │
│  └─ POST /predict/flower      →  Module 3 (MobileNetV2)│
│         │                                               │
│         ▼                                               │
│  Motor (async) → MongoDB Atlas                          │
│  Fetch plant details (names, diseases, uses, warnings)  │
│         │                                               │
│         ▼                                               │
│  Return JSON { plant_name, confidence, uses, ... }      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           MongoDB Atlas (Plant Details Database)         │
│  Stores: Sinhala/English/scientific names, diseases,    │
│  preparation methods, warnings, plant part used         │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Repository Structure

```
ayurvedic-plant-recognition/
│
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── routers/
│   │   ├── large_leaf.py       # POST /predict/large-leaf
│   │   ├── small_leaf.py       # POST /predict/small-leaf
│   │   └── flower.py           # POST /predict/flower
│   ├── ml/
│   │   ├── preprocess.py       # Shared preprocessing utilities
│   │   ├── module1/            # Large leaf models (.joblib)
│   │   ├── module2/            # Small leaf models (.joblib)
│   │   └── module3/            # Flower CNN model (Keras .h5)
│   ├── db/
│   │   └── mongo.py            # Motor async MongoDB client
│   ├── schemas.py              # Pydantic request/response models
│   ├── .env                    # MONGODB_URI and secrets (never committed)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/         # React UI components
│   │   ├── api/                # Axios API calls
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## 🌱 Plant Species

<details>
<summary><b>Module 1 — Large Leaves (18 species)</b></summary>
</details>

<details>
<summary><b>Module 2 — Small / Compound Leaves (15 species)</b></summary>
</details>

<details>
<summary><b>Module 3 — Flowers (10 species)</b></summary>
</details>

---

## ⚙️ ML Pipeline

### Classical ML

```
Raw Image
    │
    ▼
Preprocessing
  ├─ Resize to 256×256
  ├─ BGR → RGB → Grayscale
  ├─ Normalization (0–255 range)
  ├─ Median filter (noise removal)
  └─ CLAHE (contrast enhancement)
    │
    ▼
Feature Extraction
    │
    ▼
Classification
  ├─ Support Vector Machine (SVM)
  ├─ Random Forest ✅ best
  ├─ K-Nearest Neighbors (KNN)
  ├─ Decision Tree
  └─ Logistic Regression
    │
    ▼
Output: plant species label → API returns MongoDB plant document
```

### Deep Learning (Transfer Learning)

```
Raw Image
    │
    ▼
Preprocessing + Augmentation
  (rotation, flip, zoom, brightness shift → 5× dataset expansion)
    │
    ▼
Feature Extraction via Pre-trained CNN
  ├─ VGG16
  ├─ VGG19
  ├─ ResNet50
  └─ MobileNetV2 ✅ best generalization
    │
    ▼
Custom Classifier Head (Adam optimizer · Categorical Cross-Entropy)
    │
    ▼
Output: species label → API returns MongoDB plant document
```

---

## 🚀 Getting Started

### Backend Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file in /backend with your MongoDB URI
echo "MONGODB_URI=your_mongodb_atlas_connection_string" > backend/.env

# 4. Start the API server
cd backend
uvicorn main:app --reload
```

Once running, visit:
- **http://localhost:8000/docs** — interactive Swagger UI to test every endpoint
- **http://localhost:8000/redoc** — clean read-only API documentation

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# App runs at http://localhost:5173
```

### Dataset Setup

Due to size constraints, the full image dataset is not included. To reproduce training:

1. Place images under `backend/ml/dataset/train/<plant_name>/` and `.../test/<plant_name>/`
2. Ensure images have a **white background** and consistent framing
3. Each class needs a minimum of 25 training images

---

## 🐳 Docker Deployment

```bash
# Build and run the backend container
cd backend
docker build -t vedavision-api .
docker run -p 8000:8000 --env-file .env vedavision-api
```

The React frontend is deployed to **Vercel / Netlify** as static files. Update the Axios base URL in `frontend/src/api/` to point at the live Render API URL before building for production.

---

## 🗃️ Sample MongoDB Plant Document

```json
{
  "sinhala_name": "අරලිය",
  "english_name": "Temple Flower",
  "scientific_name": "Plumeria obtusa",
  "other_names": ["Araliya", "Frangipani"],
  "diseases_treated": ["anxiety", "insomnia", "fever"],
  "which_part_used": "flowers",
  "how_to_prepare": "Boil petals in water for 10 minutes.",
  "warnings": "Avoid during pregnancy"
}
```

---

## 📷 Image Capture Guidelines

For best prediction accuracy, images should follow these conditions:

- 📐 **White background** — plain white paper or poster board directly behind the leaf/flower
- 📷 **Camera angle** — 0° face-on; lens at the same height as the flower/leaf center
- 💡 **Lighting** — diffused natural light (north window, 7–9:30 AM) or two 5000K LED lamps at 45° on either side
- 📏 **Distance** — 15–20 cm for medium subjects; flower/leaf should fill 60–70% of frame
- 🔒 **Stability** — phone on tripod with 2-second self-timer; no hand-holding

---

## 🔬 Research Context

This project is being conducted as a Final Year Research Project at the **Faculty of Information Technology, University of Moratuwa** (2022–2023), supervised by **Dr. Lochandaka Ranathunga**.

---

## 📚 Key References

- Padao & Maravillas — *Naïve Bayesian Method for Plant Leaf Classification* (IEEE, 2015)
- Guru, Sharath & Manjunath — *Texture Features and KNN in Flower Classification* (2010)
- Bandara & Ranathunga — *Texture Dominant Approach for Identifying Ayurveda Herbal Species using Flowers* (MERCon IEEE, 2019)
- Gannour et al. — *Performance Evaluation of Transfer Learning for COVID-19 Detection* (IEEE, 2020)

---

## 👩‍💻 Authors

| Name | Index |
|------|-------|
| Kavya J.S. | 214105V |
| Perera M.S.S. | 214150D |
| Wijesinghe S.A. | 214233K |

---

## 📄 License

This project is for academic research purposes. Please cite appropriately if you build upon this work.

---

<p align="center">
  <i>Preserving Sri Lanka's Ayurvedic heritage through technology 🌿</i>
</p>
