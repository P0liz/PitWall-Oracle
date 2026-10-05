import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import socket
import subprocess
import time
import sys

# --- CONFIGURAZIONE ---
MLFLOW_PORT = 5000  # Porta standard (su Windows di solito è libera)
MLFLOW_HOST = "127.0.0.1"


def is_port_in_use(host: str, port: int) -> bool:
    """Verifica se la porta specificata è già occupata."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def launch_mlflow_server(host=MLFLOW_HOST, port=MLFLOW_PORT):
    """Avvia in sicurezza il server MLflow locale in background."""
    print(f"[*] [Telemetry] Controllo porta {port}...")
    if is_port_in_use(host, port):
        print(f"[*] [Telemetry] Server già attivo su http://{host}:{port}.")
        return None

    print(f"[*] [Telemetry] Avvio del server MLflow locale in background...")
    # Invoca MLflow attraverso l'interprete corrente invece dell'entrypoint
    # venv\Scripts\mlflow.exe. Su Windows Device Guard può bloccare il
    # wrapper .exe generato dall'ambiente, mentre python.exe è già autorizzato.
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
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
    log_file = open("mlflow_server_boot.log", "w", encoding="utf-8")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            close_fds=True,
            shell=False,
        )
    except Exception as e:
        print(f"Errore critico durante il boot del server: {e}")
        print("Assicurati di aver attivato l'ambiente virtuale con 'venv\\Scripts\\activate'!")
        sys.exit(1)

    try:
        # Attesa attiva del boot del server
        max_retries = 15
        for i in range(max_retries):
            time.sleep(1.0)
            if is_port_in_use(host, port):
                print(f"[OK] [Telemetry] Server avviato con successo! (PID: {process.pid})")
                return process
            if process.poll() is not None:
                print(f"Errore critico: MLflow si è chiuso con codice {process.returncode}.")
                print("Controlla il file 'mlflow_server_boot.log' per analizzare il problema.")
                sys.exit(1)
            print(f"    - In attesa che il server risponda... ({i+1}/{max_retries})")

        print("Errore critico: Il server MLflow non è partito nei tempi previsti.")
        print("Controlla il file 'mlflow_server_boot.log' per analizzare il problema.")
        process.terminate()
        sys.exit(1)
    finally:
        log_file.close()
