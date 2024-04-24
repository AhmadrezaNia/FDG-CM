import threading
import subprocess

def execute_client_script(script_name):
    subprocess.call(['python', script_name])
client_scripts =[]
# List of client script names
for i in range(9):
    client_scripts.append (f"Client_Material0{i}.py")
for j in range(6):
    client_scripts.append (f"Client_Surface0{j}.py")



# Create and start a thread for each client script
threads = []
for script in client_scripts:
    thread = threading.Thread(target=execute_client_script, args=(script,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()