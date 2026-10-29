# streamlit_app.py - 網頁介面主程式
import streamlit as st
from PIL import Image

import config
from ai_detector import load_yolo_model, detect_objects

# 1. 頁面配置
st.set_page_config(
    page_title=f"{config.MY_NAME} 的 AI 專題",
    page_icon="📱",
    layout="centered"
)

# 2. 側邊欄：個人資訊
with st.sidebar:
    st.header("👤 學生個人資訊")
    st.image("https://cdn-icons-png.flaticon.com/512/3242/3242257.png", width=90)
    st.write(f"**姓名：** {config.MY_NAME}")
    st.write(f"**班級：** {config.MY_CLASS}")
    st.write(f"**學號：** {config.MY_ID}")
    st.write(f"**組別：** {config.MY_GROUP}")
    st.divider()
    st.subheader("🛠️ 使用技術")
    for skill in config.MY_SKILLS:
        st.write(f"- {skill}")

# 3. 主畫面標題
st.title("📱 手機端即時物件偵測系統")
st.write(f"你好！我是 **{config.MY_NAME}**。請拍照或上傳照片，AI 將自動辨識人與手機！")
st.divider()

# 4. 載入模型
model = load_yolo_model()

# 5. 影像來源選擇
input_mode = st.radio(
    "選擇影像來源：",
    options=["📷 即時相機拍照 (手機適用)", "📁 上傳照片檔案 (電腦教室適用)"],
    horizontal=True,
)

image_input = None
if "即時相機拍照" in input_mode:
    camera_file = st.camera_input("請拍照")
    st.info("💡 **提示**：若使用手機，可點擊相機畫面中的翻轉按鈕使用後置鏡頭。")
    if camera_file is not None:
        image_input = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader("請選擇照片檔案...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)

# 6. 執行 AI 辨識並顯示結果
if image_input is not None:
    annotated_frame, person_count, phone_count = detect_objects(model, image_input)

    st.image(annotated_frame, caption="AI 辨識結果", use_container_width=True)

    # 顯示計數數據卡片
    col1, col2 = st.columns(2)
    col1.metric("👤 偵測人數", f"{person_count} 人")
    col2.metric("📱 偵測手機數", f"{phone_count} 支")

    # 成果回饋
    if person_count > 0 and phone_count > 0:
        st.balloons()
        st.success("🎯 太棒了！抓到你手上正拿著手機囉！")
    elif person_count > 0:
        st.warning("⚠️ 看到人了，但沒看到手機喔！")
    elif phone_count > 0:
        st.warning("⚠️ 有拍到手機，但沒拍到人捏！")
    else:
        st.info("ℹ️ 什麼都沒抓到，請把人和手機對準鏡頭再試一次～")

st.divider()
st.caption(f"© 2026 {config.MY_NAME} | 高一自主學習成果展示")