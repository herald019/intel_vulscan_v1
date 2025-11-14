# src/scanner.py

import time
from zapv2 import ZAPv2
from src import db
from src.traffic.traffic_logger import log_event, log_request

ZAP_PROXY = "http://localhost:8090"
zap = ZAPv2(proxies={'http': ZAP_PROXY, 'https': ZAP_PROXY})


def run_scan(target):
    db.init_db()
    scan_id = db.create_scan(target)

    print(f"Starting scan {scan_id} on {target}...")

    # Log scan start
    log_event(scan_id, {"event": "scan_start", "target": target})

    try:
        # Request #1 – initial open (log it)
        zap.urlopen(target)
        log_request(
            scan_id=scan_id,
            url=target,
            method="GET",
            status_code=200,              # ZAP doesn't expose, so best-guess defaults
            response_time_ms=0,
            response_length=0
        )
        time.sleep(2)

        # -----------------------------
        # SPIDER PHASE
        # -----------------------------
        spider_id = zap.spider.scan(target)

        while int(zap.spider.status(spider_id)) < 100:
            progress = int(zap.spider.status(spider_id))
            print("Spider progress:", progress, "%")

            # Log each discovered URL
            for url in zap.spider.results(spider_id):
                log_request(
                    scan_id,
                    url,
                    method="GET",
                    status_code=200,
                    response_time_ms=0,
                    response_length=0
                )

            time.sleep(2)

        print("Spider completed.")

        # -----------------------------
        # ACTIVE SCAN PHASE
        # -----------------------------
        ascan_id = zap.ascan.scan(target)

        while int(zap.ascan.status(ascan_id)) < 100:
            progress = int(zap.ascan.status(ascan_id))
            print("Active scan progress:", progress, "%")

            attacked = zap.ascan.messages_ids(ascan_id)
            for msg_id in attacked:
                try:
                    msg = zap.core.message(msg_id)
                    log_request(
                        scan_id=scan_id,
                        url=msg.get("requestHeader", "").split(" ")[1] if "requestHeader" in msg else target,
                        method=msg.get("requestHeader", "").split(" ")[0] if "requestHeader" in msg else "GET",
                        status_code=msg.get("statusCode", 0),
                        response_time_ms=msg.get("responseTimeInMs", 0),
                        response_length=msg.get("responseBodyLength", 0)
                    )
                except:
                    pass

            time.sleep(5)

        print("Active scan completed.")

        # -----------------------------
        # ALERTS
        # -----------------------------
        alerts = zap.core.alerts(baseurl=target)
        print(f"Found {len(alerts)} alerts.")

        for alert in alerts:
            db.insert_alert(scan_id, alert["alert"], alert["risk"])

            log_event(scan_id, {
                "event": "alert",
                "alert_name": alert.get("alert"),
                "risk": alert.get("risk"),
                "url": alert.get("url", target),
                "is_alert": True
            })

            print(f"{alert['alert']} - {alert['risk']}")

        db.finish_scan(scan_id, status="completed")
        log_event(scan_id, {"event": "scan_finish", "target": target})

        return scan_id, alerts

    except Exception as e:
        print(f"[!] Error during scan: {e}")
        db.finish_scan(scan_id, status="failed")
        log_event(scan_id, {"event": "scan_failed", "error": str(e)})
        return scan_id, []
