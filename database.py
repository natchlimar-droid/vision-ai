"""
database.py - SQLite storage and telemetry logging for Industrial QA Inspector.
Handles thread-safe persistence and querying of inspection metrics and verdicts.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_FILE = os.path.join(os.path.dirname(__file__), "inspection_history.db")


def get_connection() -> sqlite3.Connection:
    """Creates and returns a connection to SQLite configured with WAL mode for concurrency."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # High-concurrency Write-Ahead Logging (WAL)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    """Initializes the database schema with WAL mode and pharmaceutical master data."""
    conn = get_connection()
    cursor = conn.cursor()

    # Primary QA Inspection Telemetry Log
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            skew_angle REAL NOT NULL,
            angle_status TEXT NOT NULL,
            date_found TEXT,
            ocr_status TEXT NOT NULL,
            verdict TEXT NOT NULL,
            raw_text TEXT,
            cycle_time_ms REAL DEFAULT 0.0,
            drug_name TEXT DEFAULT '-',
            dosage TEXT DEFAULT '-',
            lot_no TEXT DEFAULT '-',
            mfg_date TEXT DEFAULT '-',
            reg_no TEXT DEFAULT '-',
            batch_id TEXT DEFAULT 'BATCH-2026-01',
            sanity_status TEXT DEFAULT 'PASS',
            defect_type TEXT DEFAULT 'NONE',
            master_match TEXT DEFAULT 'MATCHED'
        )
        """
    )

    # Pharmaceutical Master Data (Product Catalog for Label Mix-up Detection)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS product_masters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reg_no TEXT UNIQUE NOT NULL,
            drug_name TEXT NOT NULL,
            dosage TEXT NOT NULL,
            shelf_life_months INTEGER NOT NULL,
            allowed_lot_prefixes TEXT DEFAULT '',
            description TEXT
        )
        """
    )

    # 21 CFR Part 11 / GMP Audit Trail Table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            operator TEXT NOT NULL,
            details TEXT NOT NULL
        )
        """
    )
    conn.commit()

    # Seed Default Pharmaceutical Master Records if table is empty
    cursor.execute("SELECT COUNT(*) FROM product_masters")
    if cursor.fetchone()[0] == 0:
        seed_masters = [
            ("2C 15/60", "PARACETAMOL", "500 mg", 24, "DFH,B,P", "Paracetamol 500mg Blister (24 Months Shelf-life)"),
            ("1A 88/65", "PARACETAMOL", "500 mg", 36, "B,LOT,P", "Paracetamol Oral Sachet (36 Months Shelf-life)"),
            ("1A 20/55", "AMOXICILLIN", "500 mg", 24, "AMX,B", "Amoxicillin Capsules (24 Months Shelf-life)"),
            ("2A 45/61", "CETIRIZINE", "10 mg", 24, "CTZ,B", "Cetirizine 10mg Tablets (24 Months Shelf-life)"),
            ("1C 01/64", "PREMIUM PHARMA", "500 mg", 24, "B,LOT", "General Pharmaceutical Packaging Specification")
        ]
        cursor.executemany(
            """
            INSERT OR IGNORE INTO product_masters 
            (reg_no, drug_name, dosage, shelf_life_months, allowed_lot_prefixes, description)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            seed_masters
        )
        conn.commit()

    # Migration check for existing SQLite tables created before this update
    cursor.execute("PRAGMA table_info(inspection_logs)")
    columns = [row["name"] for row in cursor.fetchall()]
    for col in ["drug_name", "dosage", "lot_no", "mfg_date", "reg_no", "batch_id", "sanity_status", "defect_type", "master_match"]:
        if col not in columns:
            cursor.execute(f"ALTER TABLE inspection_logs ADD COLUMN {col} TEXT DEFAULT '-'")
    conn.commit()
    conn.close()


def get_product_masters() -> List[Dict[str, Any]]:
    """Retrieves all registered pharmaceutical products from Master Data."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM product_masters ORDER BY id ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def find_matching_master(reg_no: str = "-", drug_name: str = "-") -> Optional[Dict[str, Any]]:
    """Looks up product master data by Registration Number or Drug Name."""
    conn = get_connection()
    cursor = conn.cursor()
    clean_reg = reg_no.strip().upper().replace(" ", "").replace(".", "")
    clean_name = drug_name.strip().upper()

    cursor.execute("SELECT * FROM product_masters")
    masters = [dict(r) for r in cursor.fetchall()]
    conn.close()

    # Priority 1: Match by Registration Number
    if clean_reg and clean_reg != "-":
        for m in masters:
            m_reg = m["reg_no"].strip().upper().replace(" ", "").replace(".", "")
            if clean_reg == m_reg or clean_reg in m_reg or m_reg in clean_reg:
                return m

    # Priority 2: Match by Drug Name
    if clean_name and clean_name not in ["-", "NOT SPECIFIED", "UNKNOWN"]:
        for m in masters:
            if clean_name in m["drug_name"].upper() or m["drug_name"].upper() in clean_name:
                return m

    return None


def log_inspection(
    skew_angle: float,
    angle_status: str,
    date_found: Optional[str],
    ocr_status: str,
    verdict: str,
    raw_text: str = "",
    cycle_time_ms: float = 0.0,
    timestamp: Optional[str] = None,
    drug_name: str = "-",
    dosage: str = "-",
    lot_no: str = "-",
    mfg_date: str = "-",
    reg_no: str = "-",
    batch_id: str = "BATCH-2026-01",
    sanity_status: str = "PASS",
    defect_type: str = "NONE",
    master_match: str = "MATCHED",
    **kwargs
) -> int:
    """
    Inserts a single inspection record into the database with pharmaceutical metadata.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inspection_logs (
            timestamp, skew_angle, angle_status, date_found, 
            ocr_status, verdict, raw_text, cycle_time_ms,
            drug_name, dosage, lot_no, mfg_date, reg_no, batch_id,
            sanity_status, defect_type, master_match
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            round(skew_angle, 2),
            angle_status,
            date_found if date_found else "N/A",
            ocr_status,
            verdict,
            raw_text,
            round(cycle_time_ms, 1),
            drug_name,
            dosage,
            lot_no,
            mfg_date,
            reg_no,
            batch_id,
            sanity_status,
            defect_type,
            master_match
        ),
    )
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def log_audit_event(event_type: str, details: str, operator: str = "QA_Supervisor") -> None:
    """Records a GMP / 21 CFR Part 11 compliant audit trail log."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_events (timestamp, event_type, operator, details)
        VALUES (?, ?, ?, ?)
        """,
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, operator, details)
    )
    conn.commit()
    conn.close()


def get_audit_events_df(limit: int = 25) -> pd.DataFrame:
    """Returns the latest audit trail logs."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT id, timestamp, event_type, operator, details
        FROM audit_events
        ORDER BY id DESC
        LIMIT ?
        """,
        conn,
        params=(limit,)
    )
    conn.close()
    return df


def get_recent_logs(limit: int = 10) -> List[Dict[str, Any]]:
    """Fetches the latest N inspection records."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, timestamp, batch_id, drug_name, dosage, lot_no, skew_angle, 
               angle_status, date_found AS exp_date, mfg_date, 
               ocr_status, verdict, cycle_time_ms, raw_text
        FROM inspection_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_all_logs_df() -> pd.DataFrame:
    """Returns the entire inspection history as a pandas DataFrame."""
    init_db()
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT id, timestamp, batch_id, drug_name, dosage, lot_no, skew_angle, 
               angle_status, date_found AS exp_date, mfg_date, 
               ocr_status, verdict, cycle_time_ms, raw_text
        FROM inspection_logs
        ORDER BY id DESC
        """,
        conn
    )
    conn.close()
    return df


def get_statistics() -> Dict[str, Any]:
    """Computes overall statistics: Total, Pass, Reject, and Yield (Pass Rate %)."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END) FROM inspection_logs")
    total, passed = cursor.fetchone()
    conn.close()

    total = total or 0
    passed = passed or 0
    rejected = total - passed
    yield_rate = (passed / total * 100.0) if total > 0 else 0.0

    return {
        "total": total,
        "passed": passed,
        "rejected": rejected,
        "yield_rate": round(yield_rate, 1)
    }


def clear_history() -> None:
    """Clears all records in the database table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM inspection_logs")
    conn.commit()
    conn.close()


# Ensure DB is initialized on import
init_db()

