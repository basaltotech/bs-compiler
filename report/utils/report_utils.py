import json

def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

def load_results(filename):
    with open(filename, "r") as f:
        return json.load(f)