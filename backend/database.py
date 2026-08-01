"""SQLite database for CurrencyGuard ₹ — Indian Rupee scan history."""

import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "currency_detector.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            model_used TEXT NOT NULL,
            currency TEXT NOT NULL DEFAULT 'INR',
            features_json TEXT,
            timestamp TEXT NOT NULL,
            latitude REAL,
            longitude REAL
        )
    """)
    # Migrations for existing DB
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN latitude REAL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE scans ADD COLUMN longitude REAL")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def insert_scan(scan_id, result, confidence, model_used, features_json, timestamp, lat=None, lon=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO scans (scan_id, result, confidence, model_used, currency, features_json, timestamp, latitude, longitude) VALUES (?,?,?,?,?,?,?,?,?)",
        (scan_id, result, confidence, model_used, "INR", features_json, timestamp, lat, lon),
    )
    conn.commit()
    conn.close()

def seed_demo_locations():
    conn = get_connection()
    cities = [
        (19.0760, 72.8777), # Mumbai
        (28.6139, 77.2090), # Delhi
        (12.9716, 77.5946), # Bangalore
        (22.5726, 88.3639), # Kolkata
        (18.5204, 73.8567), # Pune
        (17.3850, 78.4867), # Hyderabad
        (23.0225, 72.5714), # Ahmedabad
        (13.0827, 80.2707)  # Chennai
    ]
    rows = conn.execute("SELECT id FROM scans WHERE latitude IS NULL OR longitude IS NULL").fetchall()
    for r in rows:
        row_id = r["id"]
        lat, lon = cities[row_id % 8]
        conn.execute("UPDATE scans SET latitude=?, longitude=? WHERE id=?", (lat, lon, row_id))
    conn.commit()
    conn.close()

def get_map_data():
    import json
    conn = get_connection()
    rows = conn.execute("SELECT scan_id, result, confidence, latitude, longitude, timestamp, features_json FROM scans WHERE latitude IS NOT NULL AND longitude IS NOT NULL").fetchall()
    conn.close()
    
    out = []
    for r in rows:
        denom = "500"
        try:
            if r["features_json"]:
                fj = json.loads(r["features_json"])
                if "denomination" in fj:
                    denom = str(fj["denomination"])
        except:
            pass
        out.append({
            "lat": r["latitude"],
            "lon": r["longitude"],
            "result": r["result"],
            "confidence": round(r["confidence"] * 100, 1),
            "denomination": denom,
            "timestamp": r["timestamp"].replace("T", " ")[:19]
        })
    return out

def get_history(limit=50):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_stats():
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    genuine = conn.execute("SELECT COUNT(*) FROM scans WHERE result='Genuine'").fetchone()[0]
    counterfeit = conn.execute("SELECT COUNT(*) FROM scans WHERE result='Counterfeit'").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM scans").fetchone()[0] or 0.0
    conn.close()
    return {
        "total_scans": total,
        "genuine_count": genuine,
        "counterfeit_count": counterfeit,
        "detection_rate": round(counterfeit / total, 4) if total > 0 else 0.0,
        "avg_confidence": round(avg_conf, 4),
        "currency": "INR",
    }
