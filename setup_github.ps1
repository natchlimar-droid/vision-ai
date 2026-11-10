# setup_github.ps1 - PowerShell script to initialize and push project to GitHub
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Industrial Vision QA - GitHub Setup Helper (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$gitInstalled = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitInstalled) {
    Write-Host "[!] ตรวจไม่พบโปรแกรม Git ในเครื่องของคุณ" -ForegroundColor Red
    Write-Host "    กรุณาติดตั้ง Git โดยรันคำสั่ง:" -ForegroundColor Yellow
    Write-Host "    winget install --id Git.Git -e --source winget" -ForegroundColor Green
    Write-Host "    หรือดาวน์โหลดจาก: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n[1/4] ตรวจสอบและสร้าง Git Repository..." -ForegroundColor Green
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
} else {
    Write-Host "Git repository ได้รับการ Initialized ไว้แล้ว" -ForegroundColor Gray
}

Write-Host "`n[2/4] สเตจไฟล์เข้าสู่ Git Index..." -ForegroundColor Green
git add .

Write-Host "`n[3/4] บันทึก Git Commit..." -ForegroundColor Green
git commit -m "feat: Industrial Quality Inspection - Label Skew & Pharmaceutical OCR Validator PoC"

Write-Host "`n========================================================" -ForegroundColor Yellow
Write-Host "  ตั้งค่า Remote Repository บน GitHub" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "1. ไปที่ https://github.com/new และสร้าง New Repository"
Write-Host "2. คัดลอก URL ของ Repository มาวางด้านล่างนี้"

$repoUrl = Read-Host "`nกรุณาวาง GitHub Repository URL (เช่น https://github.com/username/repo.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "`n[!] ไม่ได้ระบุ URL คุณสามารถรันคำสั่งด้วยตนเองในภายหลังได้ดังนี้:" -ForegroundColor Yellow
    Write-Host "    git remote add origin <URL>"
    Write-Host "    git push -u origin main"
    exit 0
}

Write-Host "`n[4/4] ทำการ Push โค้ดขึ้น GitHub..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin $repoUrl
git push -u origin main

Write-Host "`n[SUCCESS] อัปโหลดโปรเจกต์ขึ้นสู่ GitHub เรียบร้อยแล้ว!" -ForegroundColor Green

