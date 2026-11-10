"""
api_server.py - Production Decoupled FastAPI & WebSocket Inspection Service.
Allows distributed edge cameras, PLC controllers, and browser/native clients
to query QA results, stream real-time telemetry, and inspect images asynchronously.
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import cv2
import numpy as np
import asyncio
import os

from inspector import IndustrialInspector
import database as db

app = FastAPI(
    title="Industrial Quality Inspection API",
    description="Decoupled Vision Inspection & Pharmaceutical GMP Sanity Verification Service",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Core Inspector instance
inspector = IndustrialInspector()

# Connected WebSocket clients for real-time telemetry streaming
active_connections: List[WebSocket] = []


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/api/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "Industrial Vision QA Engine",
        "wal_mode": True,
        "database": "SQLite (WAL)"
    }


@app.get("/api/masters")
def list_masters():
    """Returns all registered pharmaceutical products from Master Data."""
    return db.get_product_masters()


@app.get("/api/history")
def get_history(limit: int = 50):
    """Retrieves recent inspection logs."""
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inspection_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


@app.post("/api/inspect")
async def inspect_uploaded_image(file: UploadFile = File(...)):
    """Accepts an uploaded image and performs full Skew & Pharmaceutical QA."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image payload"}

    result = inspector.inspect_frame(image)
    s_inf = result.get("sachet_info", {})

    # Save to SQLite WAL
    log_id = db.log_inspection(
        skew_angle=result["skew_angle"],
        angle_status=result["angle_status"],
        date_found=result["date_found"],
        ocr_status=result["ocr_status"],
        verdict=result["verdict"],
        raw_text=result["raw_text"],
        cycle_time_ms=result["cycle_time_ms"],
        drug_name=s_inf.get("drug_name", "-"),
        dosage=s_inf.get("dosage", "-"),
        lot_no=s_inf.get("lot_no", "-"),
        mfg_date=s_inf.get("mfg_date", "-"),
        reg_no=s_inf.get("reg_no", "-"),
        sanity_status=result.get("sanity_status", "PASS"),
        defect_type=result.get("defect_type", "NONE"),
        master_match=result.get("master_match", "MATCHED")
    )

    # Return pure JSON payload (exclude numpy images)
    return {
        "log_id": log_id,
        "verdict": result["verdict"],
        "skew_angle": result["skew_angle"],
        "angle_status": result["angle_status"],
        "date_found": result["date_found"],
        "ocr_status": result["ocr_status"],
        "sanity_status": result["sanity_status"],
        "defect_type": result["defect_type"],
        "calculated_shelf_life": result.get("calculated_shelf_life"),
        "expected_shelf_life": result.get("expected_shelf_life"),
        "master_match": result["master_match"],
        "sachet_info": s_inf,
        "cycle_time_ms": result["cycle_time_ms"]
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint streaming inspection telemetry to SCADA / UI clients."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep-alive ping
            await asyncio.sleep(1.0)
            await websocket.send_json({"type": "PING", "status": "CONNECTED"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("Starting Industrial Quality Inspection FastAPI Server on http://0.0.0.0:8000")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)

