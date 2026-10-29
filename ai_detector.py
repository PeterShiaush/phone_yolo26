# ai_detector.py - AI 辨識模組
import streamlit as st
from ultralytics import YOLO
import numpy as np
from config import MODEL_NAME, TARGET_CLASSES

@st.cache_resource
def load_yolo_model():
    """載入 YOLO 模型並快取，避免重複下載耗時"""
    return YOLO(MODEL_NAME)

def detect_objects(model, image_pil):
    """
    輸入 PIL 圖片，輸出畫框後的圖片與計數統計
    """
    image_np = np.array(image_pil)
    results = model(image_np, classes=TARGET_CLASSES)
    
    person_count = 0
    phone_count = 0

    # 計算抓到的物件數量
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        if name == "person":
            person_count += 1
        elif name == "cell phone":
            phone_count += 1

    # 繪製方框並取得標註圖片
    annotated_frame = results[0].plot()

    return annotated_frame, person_count, phone_count