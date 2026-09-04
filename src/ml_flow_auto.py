import socket
import subprocess
import time
import sys
import os
import platform

# --- CONFIGURAZIONE ---
MLFLOW_PORT = 5000  # Porta standard (su Windows di solito è libera)
MLFLOW_HOST = "127.0.0.1"


def is_port_in_use(host: str, port: int) -> bool:
    """Verifica se la porta specificata è già occupata."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def get_mlflow_executable() -> str:
    """Trova il percorso assoluto di mlflow nel venv attivo per evitare errori su Windows."""
    if platform.system() == "Windows":
        # Cerca mlflow.exe dentro la cartella Scripts del venv corrente
        venv_mlflow = os.path.join(sys.prefix, "Scripts", "mlflow.exe")
        if os.path.exists(venv_mlflow):
            return venv_mlflow
    else:
        venv_mlflow = os.path.join(sys.prefix, "bin", "mlflow")
        if os.path.exists(venv_mlflow):
            return venv_mlflow
    return "mlflow"  # Fallback globale


def launch_mlflow_server(host=MLFLOW_HOST, port=MLFLOW_PORT):
    """Avvia in sicurezza il server MLflow locale in background."""
    print(f"[*] [Telemetry] Controllo porta {port}...")
    if is_port_in_use(host, port):
        print(f"[*] [Telemetry] Server già attivo su http://{host}:{port}.")
        return None

    print(f"[*] [Telemetry] Avvio del server MLflow locale in background...")
    mlflow_bin = get_mlflow_executable()

    cmd = [
        mlflow_bin,
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--default-artifact-root",
        "./mlruns",
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Salviamo i log del server in un file per il debugging
    log_file = open("mlflow_server_boot.log", "w")

    # Windows necessita di shell=True per gestire correttamente i comandi dell'ambiente virtuale
    use_shell = platform.system() == "Windows"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            # close_fds deve essere False su Windows per non interrompere gli handle di sistema
            close_fds=(platform.system() != "Windows"),
            shell=use_shell,
        )
    except Exception as e:
        print(f"Errore critico durante il boot del server: {e}")
        print("Assicurati di aver attivato l'ambiente virtuale con 'venv\\Scripts\\activate'!")
        sys.exit(1)

    # Attesa attiva del boot del server
    max_retries = 15
    for i in range(max_retries):
        time.sleep(1.0)
        if is_port_in_use(host, port):
            print(f"[✔] [Telemetry] Server avviato con successo! (PID: {process.pid})")
            return process
        print(f"    - In attesa che il server risponda... ({i+1}/{max_retries})")

    print("Errore critico: Il server MLflow non è partito nei tempi previsti.")
    print("Controlla il file 'mlflow_server_boot.log' per analizzare il problema.")
    process.terminate()
    sys.exit(1)
