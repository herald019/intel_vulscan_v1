from zapv2 import ZAPv2

zap = ZAPv2(apikey='', proxies={'http': 'http://localhost:8090', 'https': 'http://localhost:8090'})
print("ZAP Version:", zap.core.version)
    