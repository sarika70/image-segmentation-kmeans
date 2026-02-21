#  Image Segmentation using K-Means Clustering

*A Flask-based Machine Learning Web Application*

##  Project Overview

This project performs **image segmentation** using the **K-Means clustering algorithm** to divide an image into meaningful color-based regions.
It includes a **Flask REST API** for uploading images and getting segmented outputs, along with **MLflow tracking** for experiment management.

The goal is to demonstrate a **complete ML pipeline** — from image processing to deployment.

---

##  Features

* Upload any image for segmentation
* K-Means clustering for pixel grouping
* REST API built with Flask
* MLflow integration for experiment tracking
* Returns segmented image as output
* Easy to run locally

---

##  How It Works

1. The user uploads an image
2. Image pixels are converted into numerical feature vectors
3. K-Means clusters similar pixels
4. Each pixel is reassigned based on its cluster
5. A segmented image is generated and returned

---

##  Tech Stack

* Python
* OpenCV
* NumPy
* Scikit-learn
* Flask
* MLflow

---

##  Project Structure

```
image-segmentation-kmeans/
│
├── main.py
├── app.py
├── segmentation_engine/
│   ├── __init__.py
│   ├── kmeans.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

##  Installation

### 1️ Clone the Repository

```bash
git clone https://github.com/sarika70/image-segmentation-kmeans.git
cd image-segmentation-kmeans
```

### 2️ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Run the Application

```bash
python app.py
```

The server will start at:

```
http://127.0.0.1:5000
```

---

##  API Usage

### Endpoint

```
POST /segment
```

### Request

Send an image file using Postman or any REST client.

### Response

Returns the segmented image.

---

##  MLflow Tracking

This project logs:

* Number of clusters
* Processing time
* Output image details

To view MLflow UI:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

##  Why This Project Matters

This project demonstrates:

* Real-world **computer vision**
* **Machine learning model usage**
* **REST API deployment**
* **Experiment tracking with MLflow**
* **Clean modular code structure**

---

##  Author

**Tikare Gnana Sarika Bai**
AI & Machine Learning Enthusiast
GitHub: [https://github.com/sarika70](https://github.com/sarika70)

---



