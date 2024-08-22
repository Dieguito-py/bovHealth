import subprocess

subprocess.run(["python", "scripts/preprocess_data.py"])

subprocess.run(["python", "scripts/train_model.py"])

subprocess.run(["python", "scripts/calibrate_model.py"])

subprocess.run(["python", "scripts/evaluate_model.py"])
