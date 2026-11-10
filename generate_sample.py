"""
generate_sample.py - Generates realistic synthetic industrial package labels
for immediate offline testing of Label Skew and OCR inspection.
"""

import cv2
import numpy as np
import os

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "samples")


def create_package_image(
    skew_angle_deg: float = 0.0,
    exp_date: str = "24/11/2026",
    mfg_date: str = "24/11/2024",
    is_dot_matrix: bool = True,
    corrupt_date: bool = False
) -> np.ndarray:
    """
    Creates a synthetic industrial package label on a cardboard box background.
    """
    width, height = 720, 480
    # Background: Cardboard brown / Industrial conveyor grey
    bg = np.full((height, width, 3), (210, 215, 220), dtype=np.uint8)

    # Label dimension
    lbl_w, lbl_h = 440, 260
    label = np.full((lbl_h, lbl_w, 3), (250, 250, 250), dtype=np.uint8)

    # Draw label border / distinct horizontal stripes (for Hough transform detection)
    cv2.rectangle(label, (5, 5), (lbl_w - 5, lbl_h - 5), (40, 40, 40), 4)
    # Distinct horizontal dividing edge
    cv2.line(label, (5, 60), (lbl_w - 5, 60), (30, 30, 30), 4)
    cv2.line(label, (5, lbl_h - 60), (lbl_w - 5, lbl_h - 60), (30, 30, 30), 4)

    # Product Brand & Details
    cv2.putText(label, "PREMIUM PHARMA LAB", (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 50, 150), 2)
    cv2.putText(label, "BATCH: B7042-QC", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

    # Barcode simulation stripes
    bx = 20
    for w_bar in [3, 1, 4, 2, 5, 2, 1, 4, 3, 2, 4, 1, 2, 5, 2]:
        cv2.rectangle(label, (bx, 115), (bx + w_bar, 160), (20, 20, 20), -1)
        bx += w_bar + 3

    # Date code block (EXP / MFG)
    display_exp = "EXP: 99-INVALID-XX" if corrupt_date else f"EXP: {exp_date}"
    display_mfg = f"MFG: {mfg_date}"

    # Draw date code
    cv2.putText(label, display_mfg, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(label, display_exp, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (10, 10, 10), 2)

    # Optional dot-matrix simulation: apply small noise / grid filter to the date area
    if is_dot_matrix:
        date_roi = label[190:255, 15:350]
        mask = np.random.choice([0, 1], size=date_roi.shape[:2], p=[0.15, 0.85]).astype(np.uint8)
        mask = np.expand_dims(mask, axis=-1)
        label[190:255, 15:350] = np.where(mask == 0, 240, date_roi)

    # Rotate label by skew_angle_deg
    center = (lbl_w // 2, lbl_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, skew_angle_deg, 1.0)
    rotated_label = cv2.warpAffine(label, rot_mat, (lbl_w, lbl_h), borderValue=(210, 215, 220))

    # Place rotated label in the center of background
    start_y = (height - lbl_h) // 2
    start_x = (width - lbl_w) // 2
    bg[start_y:start_y + lbl_h, start_x:start_x + lbl_w] = rotated_label

    return bg


def create_medicine_sachet_image(
    skew_angle_deg: float = 0.0,
    drug_name: str = "PARACETAMOL",
    dosage: str = "500 mg",
    lot_no: str = "B240801",
    mfg_date: str = "12/03/2026",
    exp_date: str = "11/03/2029",
    corrupt_exp: bool = False
) -> np.ndarray:
    """
    Creates a synthetic pharmaceutical foil medicine sachet (ซองยา) 
    with heat-seal crimp borders, tear notch, and regulatory details.
    """
    width, height = 720, 480
    # Background: Industrial conveyor belt (dark metallic grey)
    bg = np.full((height, width, 3), (70, 75, 80), dtype=np.uint8)

    # Sachet dimension (Foil pouch)
    s_w, s_h = 340, 420
    # Metallic Silver / White foil pouch gradient
    sachet = np.full((s_h, s_w, 3), (245, 245, 248), dtype=np.uint8)

    # 1. Heat-Seal Crimp Ribs at Top & Bottom (ลายซีลขอบซองยา)
    # Top seal band
    cv2.rectangle(sachet, (0, 0), (s_w, 45), (210, 215, 220), -1)
    for ry in range(5, 45, 4):
        cv2.line(sachet, (0, ry), (s_w, ry), (170, 175, 180), 1)

    # Bottom seal band
    cv2.rectangle(sachet, (0, s_h - 45), (s_w, s_h), (210, 215, 220), -1)
    for ry in range(s_h - 45, s_h, 4):
        cv2.line(sachet, (0, ry), (s_w, ry), (170, 175, 180), 1)

    # Left & Right edge seal
    cv2.line(sachet, (8, 0), (8, s_h), (200, 205, 210), 2)
    cv2.line(sachet, (s_w - 8, 0), (s_w - 8, s_h), (200, 205, 210), 2)

    # Outer border outline
    cv2.rectangle(sachet, (0, 0), (s_w - 1, s_h - 1), (130, 135, 140), 2)

    # 2. Tear Notch (รอยบากสำหรับฉีกซอง)
    notch_y = 60
    notch_pts = np.array([[0, notch_y], [12, notch_y + 6], [0, notch_y + 12]], np.int32)
    cv2.fillPoly(sachet, [notch_pts], (70, 75, 80))

    # 3. Medical Cross Icon (สัญลักษณ์ทางการแพทย์)
    cv2.circle(sachet, (s_w // 2, 75), 18, (230, 235, 240), -1)
    cv2.rectangle(sachet, (s_w // 2 - 3, 63), (s_w // 2 + 3, 87), (20, 120, 40), -1)
    cv2.rectangle(sachet, (s_w // 2 - 12, 72), (s_w // 2 + 12, 78), (20, 120, 40), -1)

    # 4. Pharmaceutical Details Text
    # Drug Name Header
    cv2.putText(sachet, drug_name, (25, 125), cv2.FONT_HERSHEY_DUPLEX, 0.82, (150, 20, 20), 2)
    # Dosage / Strength
    cv2.putText(sachet, dosage, (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(sachet, "TABLETS / ORAL SACHET", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)

    # Indication / Usage Instructions
    cv2.line(sachet, (20, 200), (s_w - 20, 200), (200, 205, 210), 1)
    cv2.putText(sachet, "Pain Reliever & Fever Reducer", (25, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    cv2.putText(sachet, "Dosage: Take 1 tab every 4-6 hrs", (25, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
    cv2.putText(sachet, "Reg. No. 1A 88/65", (25, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 50, 20), 1)

    # 5. Inkjet / Thermal Dot-Matrix Print Box (Lot, MFG, EXP)
    cv2.rectangle(sachet, (18, 290), (s_w - 18, 365), (255, 255, 255), -1)
    cv2.rectangle(sachet, (18, 290), (s_w - 18, 365), (180, 185, 190), 1)

    disp_exp = "EXP: XX/99/UNREADABLE" if corrupt_exp else f"EXP: {exp_date}"
    cv2.putText(sachet, f"LOT: {lot_no}", (26, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1)
    cv2.putText(sachet, f"MFG: {mfg_date}", (26, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 1)
    cv2.putText(sachet, disp_exp, (26, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)

    # Rotate sachet by skew_angle_deg
    center = (s_w // 2, s_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, skew_angle_deg, 1.0)
    rotated_sachet = cv2.warpAffine(sachet, rot_mat, (s_w, s_h), borderValue=(70, 75, 80))

    # Place in conveyor background
    start_y = (height - s_h) // 2
    start_x = (width - s_w) // 2
    bg[start_y:start_y + s_h, start_x:start_x + s_w] = rotated_sachet
    return bg


def generate_all():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # 1. PASS Sample Box (Skew = 0.5 deg, Valid EXP = 24/11/2026)
    pass_img = create_package_image(skew_angle_deg=0.5, exp_date="24/11/2026")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_pass.jpg"), pass_img)

    # 2. FAIL Skew Sample Box (Skew = 6.5 deg > 3.0 threshold, Valid Date)
    fail_skew_img = create_package_image(skew_angle_deg=6.5, exp_date="15/08/2027")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_fail_skew.jpg"), fail_skew_img)

    # 3. FAIL OCR Sample Box (Skew = 0.0 deg, Corrupt / Unparseable Date)
    fail_ocr_img = create_package_image(skew_angle_deg=0.0, corrupt_date=True)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_fail_ocr.jpg"), fail_ocr_img)

    # 4. Medicine Sachet - PASS (ซองยาพาราเซตามอล ขอบตรง +0.4°, EXP 11/03/2029)
    sachet_pass = create_medicine_sachet_image(skew_angle_deg=0.4, exp_date="11/03/2029")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sachet_pass.jpg"), sachet_pass)

    # 5. Medicine Sachet - FAIL SKEW (ซองยาเอียง +5.8° > 3.0°)
    sachet_fail_skew = create_medicine_sachet_image(skew_angle_deg=5.8, exp_date="11/03/2029")
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sachet_fail_skew.jpg"), sachet_fail_skew)

    # 6. Medicine Sachet - FAIL OCR (ซองยาวันหมดอายุชำรุด/อ่านไม่ออก)
    sachet_fail_exp = create_medicine_sachet_image(skew_angle_deg=0.2, corrupt_exp=True)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sachet_fail_exp.jpg"), sachet_fail_exp)

    print(f"[OK] Generated 6 test images in: {SAMPLE_DIR}")


if __name__ == "__main__":
    generate_all()

