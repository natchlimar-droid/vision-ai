# Industrial Quality Inspection: Label Alignment & OCR Validator (PoC)

ระบบตรวจสอบคุณภาพฉลากบรรจุภัณฑ์และวันที่ผลิต/หมดอายุอัตโนมัติ (Automated Visual Inspection) รองรับการรับภาพสดความเร็วสูงจากกล้องสมาร์ตโฟนผ่าน Wi-Fi (IP Camera), กล้อง USB และโหมดออฟไลน์ พร้อมแดชบอร์ดควบคุมและวิเคราะห์ผลแบบ Real-time ด้วย Streamlit และจัดเก็บข้อมูลลง SQLite

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
vision ai/
├── requirements.txt      # รายการ Library ทั้งหมดที่ต้องใช้
├── inspector.py          # Vision Engine: Canny+Hough วัดมุมเอียง, Morphological+OCR ตรวจสอบวันที่
├── video_stream.py       # Threaded Video Streamer (ลด Latency จากกล้อง IP ไม่ให้ภาพดีเลย์)
├── database.py           # SQLite Manager สำหรับบันทึกประวัติการตรวจสอบและสถิติ
├── generate_sample.py    # สคริปต์จำลองภาพฉลากอุตสาหกรรม (Pass / Fail Skew / Fail OCR)
├── app.py                # Streamlit Dashboard หน้าจอควบคุมและแสดงผลแบบ 2 คอลัมน์
└── samples/              # โฟลเดอร์เก็บภาพตัวอย่างสำหรับทดสอบ PoC
```

---

## 🚀 ขั้นตอนการติดตั้งและเริ่มใช้งาน (Quick Start)

### 1. ติดตั้ง Dependencies
เปิด Terminal หรือ Command Prompt ในโฟลเดอร์นี้ แล้วรันคำสั่ง:
```bash
pip install -r requirements.txt
```

### 2. รันระบบแดชบอร์ด
```bash
streamlit run app.py
```
ระบบจะเปิดหน้าจอเว็บเบราว์เซอร์อัตโนมัติที่ `http://localhost:8501`

---

## 📱 คำแนะนำการติดตั้งและเชื่อมต่อกล้องสมาร์ตโฟน (IP Camera Setup)

คุณสามารถใช้สมาร์ตโฟน (Android หรือ iOS) ทำหน้าที่เป็น Industrial Area Scan Camera ได้ง่ายๆ โดยมี 2 วิธีหลักในการเชื่อมต่อ:

### วิธีที่ 1: ใช้ Virtual Private Network เสมือน (Tailscale) — ⭐ แนะนำมากที่สุด
วิธีนี้ปลอดภัย ไม่ต้องแก้การตั้งค่า Router/Firewall และได้ IP ส่วนตัวถาวรที่เชื่อมอุปกรณ์สองเครื่องเข้าด้วยกันเหมือนอยู่ใน Wi-Fi วงเดียวกันเสมอ แม้มือถือจะต่อเน็ต 4G/5G จากที่อื่น หรือ Wi-Fi โรงงานมีระบบ AP Client Isolation

1. **ติดตั้ง Tailscale:**
   - โหลดแอป **Tailscale** ลงบนมือถือ (มีทั้ง iOS และ Android)
   - โหลดโปรแกรม **Tailscale** ลงบนคอมพิวเตอร์
2. **Login บัญชีเดียวกัน:** (เช่น ใช้ Google/Microsoft Account เดียวกัน)
3. **การทำงาน:**
   - ทั้งมือถือและคอมพิวเตอร์จะได้ IP ประจำตัวของวง Tailscale (เช่น `100.x.y.z`)
   - ในมือถือ: เปิดแอปกล้อง (เช่น IP Webcam) แล้วกด **Start server**
   - ในคอมพิวเตอร์ / หน้า Streamlit: นำ Tailscale IP ของมือถือมาต่อท้ายด้วยพอร์ต เช่น:
     `http://100.99.148.64:8080/video`
   - ระบบจะสตรีมภาพความเร็วสูงเข้ามาที่ OpenCV / Streamlit ทันที

---

### วิธีที่ 2: เชื่อมต่อผ่าน Local Wi-Fi เดียวกัน

#### สำหรับ Android (แอป **IP Webcam** โดย Pavel Khlebovich)
1. ดาวน์โหลดและติดตั้งแอป **IP Webcam** จาก Google Play Store
2. ต่อมือถือเข้ากับ **Wi-Fi วงเดียวกันกับคอมพิวเตอร์**
3. เปิดแอป เลื่อนลงมาล่างสุด กด **"Start server"**
4. หน้าจอมือถือจะแสดง URL เช่น `http://192.168.1.105:8080`
5. ในหน้าแดชบอร์ด Streamlit:
   - เลือก Input Mode: **IP Camera / Smartphone Stream**
   - ใส่ URL สตรีม: `http://192.168.1.105:8080/video`
   - *(ทางเลือกเสริม)* กรณี Wi-Fi ช้า สามารถใส่ URL โหมด Snapshot: `http://192.168.1.105:8080/shot.jpg`

#### สำหรับ iOS (iPhone / iPad)
1. ติดตั้งแอป เช่น **IPCams** หรือ **EpocCam** หรือ **DroidCam Webcam & OBS** จาก App Store
2. ตรวจสอบว่า iPhone ต่อ Wi-Fi เดียวกันกับคอมพิวเตอร์
3. เปิดแอปและเริ่มสตรีม RTSP/HTTP นำ URL เช่น `http://192.168.1.105:4747/video` หรือ `rtsp://...` มากรอกในช่อง Smartphone Camera URL บน Sidebar

---

## 🔬 หลักการทำงานของอัลกอริทึมและการอัปเกรดระดับอุตสาหกรรม (Engineering Specifications)

### A. Optical & Vision Enhancements (การจัดแสงและ Image Pipeline ซองฟอยล์)
1. **การแก้ปัญหาแสงสะท้อนบนซองฟอยล์ (Anti-Glare Optical Setup):**
   - **Hardware/Lighting:** ติดตั้งแผ่นกรองแสงโพลาไรซ์แบบตัดแสงสะท้อน (**Linear หรือ Circular Polarizer - CPL**) หน้าเลนส์กล้อง และใช้แสงสว่างชนิด **Diffused Ring Light** หรือ **Coaxial Lighting** ส่องตรงในแนวแกนเลนส์เพื่อกำจัดจุดสะท้อนแสงจ้า (Specular Hotspots) บนฟอยล์อะลูมิเนียม
   - **CLAHE Preprocessing:** ก่อนสกัดเส้นขอบ ระบบทำ **Contrast-Limited Adaptive Histogram Equalization (CLAHE)** เพื่อปรับสมดุลแสงเฉพาะจุด ตัดแสงสะท้อน และเพิ่มความเปรียบต่างของขอบรอยซีล Crimp และตัวหนังสือ Dot-matrix
2. **Robust Alignment (การวัดมุมเอียงแบบผสมผสาน):**
   - **Contour Analysis (`cv2.minAreaRect`):** ค้นหาโพลีกอนขอบนอกของตัวบรรจุภัณฑ์ (`Largest Bounding Polygon`) เพื่อวัดมุมเอียงจริงของรูปทรงซองยา ซึ่งไม่ขึ้นกับรอยยับบนฟอยล์หรือแถบพิมพ์ด้านใน
   - **Hough Line Transform (`cv2.HoughLinesP`):** วัดมุมระนาบรอยซีล (Crimp seal) เพื่อยืนยันความถูกต้องซ้ำสอง
3. **Multi-Orientation OCR Engine:**
   - หมุนสแกนอัตโนมัติ 0°, 90° CW, 270° CW รองรับแผงยาที่พิมพ์ตัวหนังสือแนวตั้ง และสแกนทั้งภาษาไทยและอังกฤษ (`eng+tha`)

---

### B. Data Validation & GMP Sanity Checks (การตรวจสอบข้อมูลยาและป้องกันฉลากสลับรุ่น)
1. **Date Object Sanity Verification:**
   - แปลง String วันที่สู่ `datetime.date` เพื่อคำนวณตามกฎเกณฑ์สากล:
     - $MFG \le Current\_Date$: วันผลิตต้องไม่อยู่ในอนาคต
     - $EXP > MFG$: วันหมดอายุต้องอยู่หลังวันผลิต
     - **Shelf-Life Verification:** คำนวณอายุยาจริงเทียบกับ Master Data (เช่น ยานี้ต้องมี Shelf-life 24 เดือน หากสแกนได้ 36 หรือ 60 เดือน จะถูกตัดสินเป็น `SHELF_LIFE_MISMATCH`)
2. **Master Data Comparison & Label Mix-up Prevention:**
   - นำข้อมูลที่สแกนได้เทียบกับตาราง `product_masters` ในฐานข้อมูล SQLite:
     - ตรวจสอบความสอดคล้องระหว่าง **เลขทะเบียนยา (Reg. No.)**, **ชื่อตัวยา (Drug Name)**, และ **ขนาดความแรง (Dosage)**
     - หากพบความผิดปกติ เช่น เลขทะเบียนยาเป็นของยา Paracetamol แต่ข้อความบนซองระบุเป็น Amoxicillin ระบบจะติดแท็ก **`LABEL_MIXUP_DEFECT`** และตัดสินสั่ง Eject เป็น **REJECT** ทันที (ป้องกัน Defect ร้ายแรงในอุตสาหกรรมยาตามมาตรฐาน GMP / PIC/S)

---

### C. Architecture & Performance (สถาปัตยกรรมแบบแยกส่วนระดับไลน์ผลิต)
1. **Decoupled Architecture (แยก Core Inspection ออกจาก UI):**
   - **`core_engine.py`:** สถาปัตยกรรม Background Worker อิสระ ดึงภาพสดจากกล้อง ประมวลผลแบบ Real-time และสั่งงาน Actuator (ไฟเตือน Andon / กระบอกสูบลมคัดแยก) โดยตรง
   - **`api_server.py`:** ให้บริการ **FastAPI & WebSockets** สำหรับเชื่อมต่อกับระบบ SCADA, PLC หรือ Native UI (PyQt/C++) ภายนอก
   - **`app.py`:** หน้าจอ Streamlit Web Dashboard สำหรับ Operator และ QA Supervisor
2. **SQLite WAL (Write-Ahead Logging) Optimization:**
   - เปิดใช้งาน `PRAGMA journal_mode=WAL;` และ `PRAGMA synchronous=NORMAL;` ใน `database.py` เพื่อให้งานประมวลผลกล้อง Real-time สามารถบันทึกผลได้พร้อมกับที่แดชบอร์ดอ่านข้อมูล โดยไม่มีปัญหา Database Locked (Zero Lock Contention)

---

## 🚦 การตัดสินผลรวม (Overall Verdict Synthesis)
- **PASS**: ผ่านทั้งเกณฑ์มุมเอียง (Skew Angle), เกณฑ์ฟอร์แมตวันที่ (OCR), กฎ Date Sanity ($EXP > MFG$, $MFG \le Today$, Shelf-life ถูกต้อง), และผ่านการตรวจสอบเทียบกับ Master Data
- **REJECT**: ตกข้อใดข้อหนึ่ง พร้อมส่งเสียงสัญญาณไซเรนเตือน (Industrial Audio Buzzer) และสั่งงาน Andon Tower ไฟสีแดงทันที

