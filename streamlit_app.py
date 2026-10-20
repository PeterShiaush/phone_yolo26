import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image


# ==========================================
# 1. 學生基本資料
# ==========================================
my_name = "王?小?明"        # 你自己的姓名
my_class = "高一 999 班"    # 你自己的班級
my_id = "1130123456789"    # 你自己的學號
my_group = "高一非常不想學習組"  # 組別 改為"高一自主想學組"

# 檢查並修改以下資料 
my_skills = [
    "Python 基礎??還好",
    "VS Code 操作??尚可",
    "YOLO??99 物件偵測??完成",
    "Streamlit 網頁應用??佈署",
    "GitHub 雲端倉庫??儲存",
]


# ==========================================
# 2. 側邊欄：個人名片
# ==========================================
st.set_page_config(
    page_title=f"{my_name} 的 AI 專題", page_icon="📱", layout="centered"
)

with st.sidebar:
    st.header("👤 學生個人資訊")
    st.image("https://cdn-icons-png.flaticon.com/512/3242/3242257.png", width=90)
    st.write(f"**姓名：** {my_name}")
    st.write(f"**班級：** {my_class}")
    st.write(f"**學號：** {my_id}")
    st.write(f"**組別：** {my_group}")

    st.divider()

    st.divider()
    st.subheader("🛠️ 使用技術")
    for skill in my_skills:
        st.write(f"- {skill}")

# ==========================================
# 3. 主畫面與模型載入
# ==========================================
st.title("📱 手機端即時物件偵測系統")
st.write(f"你好！我是 **{my_name}**。請拍照或上傳照片，AI 將自動辨識畫面上的人與手機！")
st.divider()


# 使用快取載入模型，避免重複下載耗時
@st.cache_resource
def load_model():
    return YOLO("yolo26s.pt")


model = load_model()

# ==========================================
# 4. 影像輸入：支援拍照與檔案上傳
# ==========================================
input_mode = st.radio(
    "選擇影像來源：",
    options=["📷 即時相機拍照 (手機適用)", "📁 上傳照片檔案 (電腦教室適用)"],
    horizontal=True,
)

image_input = None

if "即時相機拍照" in input_mode:
    camera_file = st.camera_input("請拍照")
    st.info("💡 **提示**：請點擊相機畫面中的「🔄 鏡頭切換圖示」以使用後置鏡頭拍攝物件！")
    if camera_file is not None:
        image_input = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader("請選擇照片檔案...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_input = Image.open(uploaded_file)

# ==========================================
# 5. 執行 AI 辨識與結果呈現
# ==========================================
if image_input is not None:
    # 轉為 NumPy 格式供 YOLO 處理
    image_np = np.array(image_input)

    # 執行 YOLO 預測（僅限定類別 0: person, 67: cell phone）

    results = model(image_np, classes=[0, 67])

    # 初始化計數器
    person_count = 0
    phone_count = 0

    # 逐一檢查 AI 抓到的每一個物件
    for box in results[0].boxes:
        cls_id = int(box.cls[0])  # 物件編號
        name = model.names[cls_id]  # 英文名稱（查表換成文字）
        if name == "person":
            person_count += 1
        if name == "cell phone":
            phone_count += 1

    # 繪製辨識方框並維持 RGB 色彩空間
    annotated_frame = results[0].plot()
    # results[0].plot() 預設輸出 RGB 格式，直接繪製即可避免二次色彩轉換造成的失真
    st.image(annotated_frame, caption="AI 辨識結果", width="stretch")

    # 顯示統計數據卡片
    col1, col2 = st.columns(2)
    col1.metric("👤 偵測人數", f"{person_count} 人")
    col2.metric("📱 偵測手機數", f"{phone_count} 支")

    # 依偵測結果給出提示訊息
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
st.caption(f"© 2026 {my_name} | 高一自主學習成果展示")
