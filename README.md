# 🍄 มินิโปรเจกต์: จำแนกเห็ดกินได้ / มีพิษ ด้วย Decision Tree

โปรเจกต์นี้ทำตามโจทย์ที่อาจารย์ให้ — เลือก dataset ใหม่ (ไม่ซ้ำกับ 6 โมเดลก่อนหน้า) และเลือกอัลกอริทึม
**Decision Tree** มาสร้างโมเดลทำนาย พร้อมอธิบายทฤษฎี เปรียบเทียบผล และทำเป็นเว็บแอป Streamlit

## ไฟล์ในโปรเจกต์

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | เว็บแอป Streamlit หลัก (5 แท็บ ตรงตามโจทย์ 5 ข้อ) |
| `requirements.txt` | รายการไลบรารีที่ต้องติดตั้งบน Streamlit Cloud |
| `dt_model.pkl` | โมเดล Decision Tree ที่เทรนแล้ว |
| `feature_columns.json` | รายชื่อคอลัมน์หลัง One-Hot Encoding (116 คอลัมน์) ต้องตรงลำดับกับตอนเทรน |
| `category_options.json` | ตัวเลือกของแต่ละตัวแปร ใช้สร้าง dropdown ในหน้าเว็บ |
| `model_comparison.json` | ผลเปรียบเทียบ Decision Tree / Random Forest / KNN |
| `dt_confusion_matrix.npy` | Confusion matrix ของ Decision Tree บนชุดทดสอบ |
| `dt_feature_importance.json`, `feature_importance.png` | ความสำคัญของตัวแปร (top 10) |
| `comparison_chart.png` | กราฟเปรียบเทียบ Accuracy/Precision/Recall/F1 |
| `dt_rules_preview.txt` | กฎการตัดสินใจของ Decision Tree (3 ระดับแรก) |
| `profile.jpg` | รูปโปรไฟล์ที่แสดงบนหัวเว็บ |
| `agaricus-lepiota.data.txt`, `descriptors.txt` | ข้อมูลดิบต้นฉบับจาก UCI + คำอธิบายตัวย่อ |
| `train.py` | สคริปต์เทรนโมเดลทั้งหมด (โหลดข้อมูล → preprocessing → เทรน → บันทึกผล) |
| `make_charts.py` | สคริปต์สร้างกราฟสองรูป (เรียกหลัง `train.py`) |

## รันทดสอบในเครื่องตัวเอง

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ขึ้น Streamlit Cloud (เหมือน 6 โมเดลก่อนหน้า)

1. สร้าง repository ใหม่บน GitHub เช่น `mushroom-classifier`
2. อัปโหลดไฟล์ทั้งหมดในโฟลเดอร์นี้ (ยกเว้น `train.py`, `make_charts.py` ก็ได้ถ้าไม่อยากรวมสคริปต์เทรน)
3. ไปที่ [share.streamlit.io](https://share.streamlit.io/) → New app → เลือก repo → ไฟล์หลักคือ `app.py`
4. รอสักครู่ จะได้ลิงก์เว็บแอปสำหรับส่งอาจารย์

## หากต้องการเทรนโมเดลใหม่ / ปรับพารามิเตอร์

```bash
python train.py         # เทรนโมเดลใหม่ทั้งหมด สร้างไฟล์ .pkl / .json ใหม่
python make_charts.py   # สร้างกราฟใหม่จากผลล่าสุด
```

## เกี่ยวกับ Dataset

**UCI Machine Learning Repository — Mushroom Data Set**
8,124 ตัวอย่าง, 22 ตัวแปร (ทั้งหมดเป็น categorical), เป้าหมาย: edible (กินได้) / poisonous (มีพิษ)
