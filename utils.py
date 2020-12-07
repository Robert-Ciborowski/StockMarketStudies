import json

def getKeys(path="keys.json"):
    try:
        with open(path) as f:
          data = json.load(f)
          return data
    except:
        raise Exception("Failed to find " + path + " when getting keys!")
