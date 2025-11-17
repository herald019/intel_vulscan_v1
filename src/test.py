from zapv2 import ZAPv2

zap = ZAPv2(proxies={'http': 'http://localhost:8090', 'https': 'http://localhost:8090'})
print(zap.core.version)
