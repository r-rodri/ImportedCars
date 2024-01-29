import subprocess
import threading

# List of scripts to run in sequence
scripts_to_run = ["Extrair AutoScout24.py", "Extrair StandVirtual.py", "Cleaning AutoScout24.py", "MergedFiles.py", "RAW_Analise Diferencas Preco.py"]

# Function to run a script
def run_script(script):
    print(f"\nRunning script: {script}")
    result = subprocess.run(["python", script])
    if result.returncode != 0:
        print(f"Error: Failed to run {script}")
    else:
        print(f"Script {script} completed successfully.")

# Create threads for script1 and script2
threads = []
for script in scripts_to_run[:2]:
    thread = threading.Thread(target=run_script, args=(script,))
    threads.append(thread)
    thread.start()

# Wait for script1 and script2 to complete
for thread in threads:
    thread.join()

# Run the remaining scripts one by one
for script in scripts_to_run[2:]:
    run_script(script)
