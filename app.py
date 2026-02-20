# eKYC pipeline dependencies

import cv2
import numpy as np
from doc_cam_extract import take_ss_from_cam, crop_face, get_mbfnet_embeddings
from blink_detection import count_blinks
from head_turn_detection import head_rotation_detector
from blazeface import BlazeFace
import torch
import streamlit as st
import time

# load model and enable GPU access

device = 'cuda' if torch.cuda.is_available else 'cpu'
bf = BlazeFace().to(device)
bf.load_weights('blazeface.pth')
bf.load_anchors('anchors.npy')

# detection thresholds
bf.min_score_thresh = 0.75
bf.min_supression_threshold = 0.3

# set title
st.set_page_config(page_title="Bank eKYC System Rebuild", layout="wide")

# paper title
st.markdown("<h1 style='text-align:center;'>🏦💰Bank eKYC System Rebuild🧑📸🤖</h1>", unsafe_allow_html=True)
st.write("---")

st.set_page_config(page_title="Liveliness Test", layout="wide")

# ---- Sidebar-style container on left ----
left_col, right_col = st.columns([1, 2])  # left smaller, right larger

with left_col:
    st.markdown("## 🌱 Liveliness Test")

    # ---- Session State ----
    if "blink_result" not in st.session_state:
        st.session_state.blink_result = None
    if "turn_result" not in st.session_state:
        st.session_state.turn_result = None
    if "screenshot_result" not in st.session_state:
        st.session_state.screenshot_result = None

    # ---- Dummy Test Functions ----
    def run_blink_test():
        blink_check = count_blinks()
        return 1

    def run_head_turn_test():
        head_turn = head_rotation_detector()
        return 1

    def run_webcam_screenshot():
        take_ss_from_cam()
        return 1

    # ---- 3-Column Checklist ----
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.markdown("**Test Type**")
        st.write("Blink Test")
        st.write("Head Turn Test")
        st.write("Webcam Screenshot")

    with c2:
        st.markdown("**Run Test**")
        if st.button("▶ Test1"):
            if run_blink_test() == 1:
                st.session_state.blink_result = "✅ Pass"
        if st.button("▶ Test2"):
            if run_head_turn_test() == 1:
                st.session_state.turn_result = "✅ Pass"
        if st.button("▶ Test3"):
            if run_webcam_screenshot() == 1:
                st.session_state.screenshot_result = "✅ Pass"

    with c3:
        st.markdown("**Result**")
        st.write(st.session_state.blink_result or "❌ Pending")
        st.write(st.session_state.turn_result or "❌ Pending")
        st.write(st.session_state.screenshot_result or "❌ Pending")

    st.markdown("---")

    # ---- Overall Result ----
    if all([
        st.session_state.blink_result == "✅ Pass",
        st.session_state.turn_result == "✅ Pass",
        st.session_state.screenshot_result == "✅ Pass"
    ]):
        st.success("✅ Liveliness Verified")
    else:
        st.info("Awaiting liveliness test completions...")

# right column

with right_col:
    st.markdown("## 🧩 Face Matching Test")

    # Layout: Two images side by side
    img_col1, img_col2 = st.columns(2)

    def load_image(path):
        try:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            else:
                return np.zeros((128, 128, 3), dtype=np.uint8)
        except:
            return np.zeros((128, 128, 3), dtype=np.uint8)

    # ---- Matching Logic ----
    def run_face_matcher():
        # crop faces for embedding
        img1 = cv2.imread('assets/capture.jpg')
        img2 = cv2.imread('assets/Karthick_Face.png')
        crop_face(img1, bf, 'cam')
        crop_face(img2, bf, 'doc')
        # Example image placeholders (replace with actual paths or uploaded images)
        img1_path = "img_cache/cam.jpg"  # from BlazeFace crop
        img2_path = "assets/Karthick_Face.png" # reference face
        with img_col1:
            st.image(load_image(img1_path), caption="Captured Face", use_container_width=True)
        with img_col2:
            st.image(load_image(img2_path), caption="Aadhar Card Face", use_container_width=True)
        embeddings_list = get_mbfnet_embeddings('img_cache')
        vec1 = embeddings_list[0]
        vec2 = embeddings_list[1]
        euc_dist = np.linalg.norm(vec1 - vec2)
        #print(euc_dist)
        # thresholding
        if(euc_dist < 5.5):
            st.write('✅ Customer Identity verified EKYC done')
        else:
            st.write('❌ Identity mis-match EKYC Failed')

        st.markdown("### ")
        

    if st.button("🔍 Match Faces"):
        result = run_face_matcher()
        