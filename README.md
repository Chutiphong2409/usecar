# ระบบทำนายราคารถยนต์มือสองด้วย Machine Learning

แอป Streamlit ภาษาไทยสำหรับสำรวจชุดข้อมูลรถยนต์มือสอง ฝึกและเปรียบเทียบโมเดล Regression และทดลองทำนายราคา

## Dataset
- แหล่งข้อมูล: [Used Car Price Prediction Dataset (GitHub)](https://github.com/abadeanlou/Used-Car-Price-Prediction)
- ไฟล์ข้อมูลที่แอปโหลด: `used_cars.csv` จาก repository ข้างต้น
- ชุดข้อมูลมีข้อมูลรถยนต์มือสองประมาณ 4,009 รายการตาม README ของแหล่งข้อมูล
- ข้อมูลราคาเป็นราคาจากตลาดต่างประเทศและหน่วยเงินตามชุดข้อมูลต้นทาง ไม่ควรตีความเป็นราคาตลาดไทยโดยตรง

## วิธีรันในเครื่อง
```bash
pip install -r requirements.txt
streamlit run app.py
```

## แท็บในแอป
1. หน้าแรก — วัตถุประสงค์และขั้นตอนการทำงาน
2. สำรวจข้อมูล — ตารางข้อมูล กราฟราคา และดาวน์โหลด CSV
3. ทดลองทำนายราคา — กรอกคุณลักษณะรถและรับค่าประมาณราคา
4. เปรียบเทียบโมเดล — Random Forest, Extra Trees และ Ridge Regression พร้อม MAE, RMSE, R²
5. ไฟล์ประกอบ — ลิงก์ Google Colab, YouTube และ PowerPoint

## เพิ่มลิงก์ไฟล์ประกอบ
เปิด `app.py` และเติม URL ที่ต้นไฟล์:
```python
COLAB_URL = "https://colab.research.google.com/..."
YOUTUBE_URL = "https://www.youtube.com/..."
POWERPOINT_URL = "https://..."
```
บันทึกและอัปโหลด `app.py` ที่แก้ไขแล้วขึ้น GitHub

## Deploy ผ่าน Streamlit Community Cloud
1. สร้าง GitHub repository และอัปโหลด `app.py`, `requirements.txt`, `README.md`
2. ไปที่ https://share.streamlit.io/
3. เลือก repository, branch `main` และไฟล์หลัก `app.py`
4. กด Deploy

ผู้พัฒนา: ชุติพงศ์ ทองคำ | รหัสนักศึกษา 664245053 | กลุ่ม 66/44
