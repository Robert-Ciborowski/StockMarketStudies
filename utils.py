import json

def getKeys(path="keys.json"):
    with open(path) as f:
      data = json.load(f)
      return data
