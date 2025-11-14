import os
from src import db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

def generate_report():
    results = db.fetch_all_results()
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    html_content = """
    <html>
    <head>
        <title>Scan Report</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h2>Vulnerability Scan Report</h2>
        <table>
            <tr>
                <th>Scan ID</th>
                <th>Target</th>
                <th>Started</th>
                <th>Finished</th>
                <th>Alert</th>
                <th>Risk</th>
                <th>Alert Timestamp</th>
            </tr>
    """

    for row in results:
        scan_id, target, started, finished, alert, risk, created_at = row
        html_content += f"<tr><td>{scan_id}</td><td>{target}</td><td>{started}</td><td>{finished}</td><td>{alert or ''}</td><td>{risk or ''}</td><td>{created_at or ''}</td></tr>"

    html_content += """
        </table>
    </body>
    </html>
    """

    report_path = os.path.join(REPORTS_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[+] Report generated: {report_path}")
