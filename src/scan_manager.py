from src import scanner, db, report_generator

def main():
    db.init_db()
    target = "http://dvwa:80"
    scanner.run_scan(target)
    report_generator.generate_report()

if __name__ == "__main__":
    main()
