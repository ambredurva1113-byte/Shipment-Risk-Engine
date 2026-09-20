"""One command to rebuild everything:  python setup.py"""
import subprocess, sys
for step in ["generate_data.py", "trust_engine.py", "train_model.py"]:
    print(f"\n>>> running {step}")
    subprocess.run([sys.executable, step], check=True)
print("\nDone. Try:  python risk_predictor.py   then   streamlit run app.py")
