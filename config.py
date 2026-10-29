# config.py - 學生個人資訊與設定

# 1. 學生個人資訊基本資料
MY_NAME = "王?小?明"        # 你自己的姓名
MY_CLASS = "高一 ??? 班"    # 你自己的班級
MY_ID = "1130????"      # 你自己的學號
MY_GROUP = "高一??非常不想學習組"  # 組別 改為"高一自主想學組"

# 2. 個人專長，檢查並修改以下資料 
MY_SKILLS = [
    "Python 基礎??還好",
    "VS Code 操作??尚可",
    "YOLO2699 物件偵測??完成",
    "Streamlit 網頁應用??佈署",
    "GitHub 雲端倉庫??儲存",
]

# 3.  模型"yolo26s.pt"，與類別設定 (0: person, 67: cell phone)
MODEL_NAME = "yolo26s.pt"
TARGET_CLASSES = [0, 67]