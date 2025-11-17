# Intel-VulScan
AI-Augmented Vulnerability Scanner with ZAP Integration, ML Models & Smart Crawler (DQN)

Intel-VulScan is a lightweight, intelligent vulnerability scanner built on top of OWASP ZAP, enhanced with:
- Machine-learning powered Risk Prediction Model
- LSTM-based Anomaly Detection Model
- Prototype Reinforcement Learning Smart Crawler (DQN)
- Traffic logging system for ML training
- HTML reports, JSON exports, and CLI utilities
- Fully offline, zero-cost, Python-based solution
This project is aimed at research, experimentation, and proof-of-concept development in the direction of AI-augmented cyber-security automation.

## Features

### Vulnerability Scanning (OWASP ZAP)
- Automated spidering
- Active scan execution
- Full logging of HTTP traffic
- Vulnerability alert extraction

### Risk Prediction Model (ML)
- Predicts risk severity of vulnerabilities using:
- Alert metadata
- Target domain
- Scan duration
- Alert frequency
- Uses TF-IDF + LightGBM classifier.

### Anomaly Detection (LSTM Autoencoder)
- Detects abnormal traffic patterns using:
- Status codes
- Response times
- Response size
- Alert flags
- Training uses .jsonl traffic logs collected from scans.

### Smart Crawler (Deep Reinforcement Learning – DQN)
- A prototype crawler that:
- Learns traversal behavior
- Rewards for discovering new URLs
- Bonus reward when finding alert-heavy areas
- Uses basic state representation (index-based fallback)
⚠️ This component is experimental and still under refinement.
The current version includes:

### Training (DQN)
- Running the crawler using the trained model
- Basic environment + spider integration

### Output Options
- HTML Report
- JSON export of all scans
- SQLite DB storage
- Traffic .jsonl logs

## Install Requirements
just install all the packages on a virtual env using the requirements.txt
start the docker files using docker-compose up -d command
(zap and the dummy webpage should start running after that)

also make sure to run owaspzap in daemon/headless mode. use the following command
```bash
  zap.sh -daemon -config api.disablekey=true -port 8090 
  ```
or
```bash
  zap.bat -daemon -config api.disablekey=true -port 8090
```

### CLI
just read through the cli file i mean..... its literally written in the cli.py file. the ai crawler is very experimental though.... might not work proper (i should work proper tho)


