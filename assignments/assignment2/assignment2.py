import subprocess

if __name__ == "__main__":
    # fetch data from https://github.com/openMVG/SfM_quality_evaluation
    subprocess.run(["python", "assignments\\assignment2\\feat_match.py"])

    subprocess.run(["python", "assignments\\assignment2\\sfm.py"])