import json
import os
from collections import Counter
from src import db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

def generate_statistics(save_to_file=True):
    results = db.fetch_all_results()
    if not results:
        print("[!] No data in database yet.")
        return

    total_scans = len(set([r[0] for r in results]))
    total_alerts = len([r for r in results if r[4]])
    risk_levels = Counter([r[5] for r in results if r[5]])
    alerts = Counter([r[4] for r in results if r[4]])

    stats = {
        "total_scans": total_scans,
        "total_alerts": total_alerts,
        "risk_distribution": dict(risk_levels),
        "top_alerts": alerts.most_common(5),
        "latest_scan": {
            "scan_id": results[0][0],
            "target": results[0][1],
            "started_at": results[0][2],
            "finished_at": results[0][3],
        }
    }

    print("\n📊 Vulnerability Analytics Report")
    print("=" * 40)
    print(f"Total Scans: {stats['total_scans']}")
    print(f"Total Alerts: {stats['total_alerts']}")
    print("\nRisk Level Distribution:")
    for risk, count in stats["risk_distribution"].items():
        print(f"  {risk}: {count}")

    print("\nMost Common Alerts:")
    for alert, count in stats["top_alerts"]:
        print(f"  {alert}: {count}")

    print("\nLast Scan Summary:")
    print(f"  Target: {stats['latest_scan']['target']}")
    print(f"  Started: {stats['latest_scan']['started_at']}")
    print(f"  Finished: {stats['latest_scan']['finished_at']}")
    print("=" * 40)

    if save_to_file:
        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
        path = os.path.join(REPORTS_DIR, "stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        print(f"[+] Analytics saved to {path}")

    return stats
