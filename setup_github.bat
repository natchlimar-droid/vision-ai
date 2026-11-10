@echo off
chcp 65001 >nul
echo ========================================================
echo   Industrial Vision QA - GitHub Repository Setup Helper
echo ========================================================
echo.

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] ตรวจไม่พบโปรแกรม Git ในเครื่องของคุณ
    echo     กรุณาติดตั้ง Git ก่อนโดยรันคำสั่ง:
    echo     winget install --id Git.Git -e --source winget
    echo     หรือดาวน์โหลดจาก https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo [1/4] ตรวจสอบความพร้อมของไฟล์...
if not exist ".git" (
    echo [*] ทำการ Initialize Git Repository...
    git init
    git branch -M main
) else (
    echo [*] Git repository มีอยู่แล้ว
)

echo [2/4] สเตจไฟล์ทั้งหมด (อิงตาม .gitignore)...
git add .

echo [3/4] ทำการบันทึก Commit แรก...
git commit -m "feat: Industrial Quality Inspection - Label Skew & Pharmaceutical OCR Validator PoC"

echo.
echo ========================================================
echo   สร้าง Repository เปล่าบน GitHub ของคุณแล้วหรือยัง?
echo ========================================================
echo   1. ไปที่ https://github.com/new
echo   2. ตั้งชื่อ Repository (เช่น: vision-ai-qa-inspector)
echo   3. เลือกเป็น Public หรือ Private (ไม่ต้องติ๊ก Add README)
echo   4. กด Create repository
echo.
set /p REPO_URL="กรุณาวาง GitHub Repository URL ของคุณที่นี่ (เช่น https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo [!] ไม่ได้ระบุ URL ยกเลิกการ Push
    echo     คุณสามารถ Push ภายหลังได้ด้วยคำสั่ง:
    echo     git remote add origin ^<URL^>
    echo     git push -u origin main
    echo.
    pause
    exit /b 0
)

echo [4/4] กำลัง Push โค้ดขึ้น GitHub...
git remote remove origin >nul 2>nul
git remote add origin %REPO_URL%
git push -u origin main

echo.
echo ========================================================
echo   [SUCCESS] อัปโหลดโปรเจกต์ขึ้น GitHub เรียบร้อยแล้ว!
echo ========================================================
pause

