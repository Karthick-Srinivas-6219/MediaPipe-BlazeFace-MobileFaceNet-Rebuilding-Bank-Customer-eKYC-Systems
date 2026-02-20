# 🏦🤖MediaPipe-BlazeFace-MobileFaceNet-Rebuilding-Bank-Customer-eKYC-Systems
This repo presents my approach at building a customer eKYC system used by banks etc. from scratch. These Systems have 2 components - Liveliness Detection (Blinking &amp; Head Rotations - Used MediaPipe Keypoints) &amp; Face Matching (Current face capture with face image on an official document like Aadhar etc. - Used BlazeFace + MobileFaceNet). 

# Demo 👇
<video src="demo.mp4" controls width="640"></video>
[[Link to Demo]](https://youtu.be/Z6WPcFX8OxI "Click to watch")

# Overview of the pipeline
![Alt text](complete_ekyc_pipeline.png)

## 🚀 Features

✅ **Mediapipe**: A library using which we can estimate **478 3D facial keypoints** in real time. These keypoints can be used to perform simple geometric calculations like **Euclidean Distance, Pitch, Yaw & Roll** which are robust yet **lightweight** measures for **detecting blinks & head turns** used in the **liveliness detection** module .

✅ **BlazeFace**: A **mobile-native, lightweight face detector** that is used to **capture** the customers face when he initiates the eKYC process, after **passing** the liveliness detection phase.

✅ **MobileFaceNet**: A **mobile-native, lightweight face embedderr** that extracts facial features and represents them as an **embedding vector of 128-dimensions**. Embeddings are generated for the customer's **captured face** and their **face image in the submitted official document (aadhar, passport etc.)**. To verify identity, **euclidean distance based similarity score** is computed between the 2 embeddings if the distance is **under 0.45** then identity is verified else eKYC has failed. 

## 📂 Project Structure

```bash
.
├── ArcFace/              # Util files for MobileFaceNet to generate captured face and document face embeddings of 128-dim.
├── assets/          # The cache where the captured face and document face images sit before they are consumed by the embedding generating function.
├── anchors.npy           # NumPy file containing pre-set anchors for the BlazeFace face detector.
├── requirements.txt      # Python dependencies.
├── app.py     # Complete eKYC pipeline: liveliness detection --> face matching.
├── crop_recog_persistant_inf.ipynb  # Runs the entire inference pipeline i.e. supply test image --> faces get detected and cropped --> Embeddings get generated and matched with the cached gallery embeddings.
├── demo.py            # A Streamlit demo of the entire project.
├── my_dlib_funcs.py   # Some utility functions for embeddings generation and caching.
├── gallery_embeddings.pkl   # Embedding cache represented as a pickle file.
├── requirements.txt   # Project dependencies
├── dlib-19.24.99-cp313-cp313-win_amd64.whl  # dlib wheel for python3.13
```

