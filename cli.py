import argparse
import json
import os

from src import db, scanner, report_generator, analytics
from src.models.risk import train_risk
from src.models.anomaly import train_anomaly


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def export_json():
    """Export all scan records into JSON."""
    results = db.fetch_all_results()
    if not results:
        print("[!] No scans available to export")
        return

    scans = {}
    for row in results:
        scan_id, target, started, finished, alert, risk, created_at = row
        if scan_id not in scans:
            scans[scan_id] = {
                "scan_id": scan_id,
                "target": target,
                "started_at": started,
                "finished_at": finished,
                "status": "completed" if finished else "running",
                "alerts": []
            }
        if alert:
            scans[scan_id]["alerts"].append({
                "alert_name": alert,
                "risk": risk,
                "created_at": created_at
            })

    os.makedirs(REPORTS_DIR, exist_ok=True)
    json_path = os.path.join(REPORTS_DIR, "scan_results.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(scans.values()), f, indent=4)

    print(f"[+] JSON export saved at {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Vuln Scanner CLI")

    parser.add_argument("--target", type=str, help="Target URL to scan")
    parser.add_argument("--list-scans", action="store_true", help="List previous scans")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report")
    parser.add_argument("--export-json", action="store_true", help="Export scans to JSON")

    # ML Models
    parser.add_argument("--train-risk", action="store_true", help="Train the risk prediction model")
    parser.add_argument("--train-anomaly", action="store_true", help="Train the LSTM anomaly detection model")

    args = parser.parse_args()

    # Init DB every time
    db.init_db()

    if args.target:
        scanner.run_scan(args.target)

    elif args.list_scans:
        results = db.fetch_all_results()
        if not results:
            print("[!] No scans found")
        else:
            print("Previous scans:\n")
            for row in results:
                scan_id, target, started, finished, alert, risk, created_at = row
                print(
                    f"Scan {scan_id} | Target={target} | Started={started} | "
                    f"Finished={finished} | Alert={alert or '-'} | Risk={risk or '-'}"
                )

    elif args.generate_report:
        report_generator.generate_report()

    elif args.export_json:
        export_json()

    elif args.train_risk:
        train_risk.train_and_save()

    elif args.train_anomaly:
        train_anomaly.train_and_save()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
