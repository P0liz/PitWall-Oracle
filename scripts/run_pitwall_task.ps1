<#
.SYNOPSIS
    Wrapper eseguito da Task Scheduler per PitWall Oracle.
    Sostituisce i due workflow GitHub Actions (publish-prediction.yml,
    post-race-pipeline.yml), preservando:
      - la mutua esclusione che prima dava `concurrency: group: pitwall-publication`
      - la distinzione tra "deferred/no-op" (exit 0, normale) e vero fallimento
      - un log persistente equivalente al job summary di Actions

.PARAMETER Task
    "prediction" oppure "postrace".

.NOTES
    Percorsi da adattare una sola volta ai valori reali del tuo ambiente,
    vedi la sezione CONFIGURAZIONE qui sotto.
#>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("prediction", "postrace")]
    [string]$Task
)

# ============================== CONFIGURAZIONE ==============================
$RepoRoot   = "C:\Users\ASUS\Github\PitWall-Oracle"      
$VenvPython = "C:\Users\ASUS\Github\PitWall-Oracle\venv\Scripts\python.exe"
$LogDir     = "C:\Users\ASUS\Github\PitWall-Oracle\logs"
$LockDir    = "C:\Users\ASUS\Github\PitWall-Oracle\.locks"
# ==============================================================================

New-Item -ItemType Directory -Force -Path $LogDir  | Out-Null
New-Item -ItemType Directory -Force -Path $LockDir | Out-Null

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logFile   = Join-Path $LogDir "$Task`_$timestamp.log"
# Lock CONDIVISO tra i due task: replica `concurrency: group: pitwall-publication`
# di GitHub Actions, dove prediction e post-race non giravano mai insieme perché
# entrambi i workflow usavano lo stesso nome di gruppo.
$lockFile    = Join-Path $LockDir "pitwall-publication.lock"
$waitingLock = Join-Path $LockDir "waiting_$Task.lock"

# Inizializzazione Win32 Power Management
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);

    [DllImport("powrprof.dll", SetLastError = true)]
    public static extern bool SetSuspendState(bool hibernate, bool forceCritical, bool disableWakeEvent);
'@ -ErrorAction SilentlyContinue

$ES_CONTINUOUS        = [uint32]::Parse("2147483648") # 0x80000000
$ES_SYSTEM_REQUIRED   = [uint32]1                     # 0x00000001
$ES_AWAYMODE_REQUIRED = [uint32]64                    # 0x00000040

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Add-Content -Path $logFile -Value $line
    Write-Output $line
}

function Send-FailureNotification {
    param([string]$Summary)
    # Notifica minimale via toast di Windows (nessuna dipendenza esterna).
    # Se preferisci una mail, sostituisci questo blocco con Send-MailMessage
    # o un client SMTP dedicato.
    try {
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
        $template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
        $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
        $textNodes = $xml.GetElementsByTagName("text")
        $textNodes.Item(0).AppendChild($xml.CreateTextNode("PitWall Oracle - $Task fallito")) | Out-Null
        $textNodes.Item(1).AppendChild($xml.CreateTextNode($Summary)) | Out-Null
        $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("PitWall Oracle").Show($toast)
    } catch {
        Write-Log "Notifica toast non disponibile: $_"
    }
}

function Test-WokeByTimer {
    try {
        $recentWake = Get-WinEvent -FilterHashtable @{
            LogName      = 'System'
            ProviderName = 'Microsoft-Windows-Power-Troubleshooter'
            Id           = 1
        } -MaxEvents 1 -ErrorAction Stop

        if ($recentWake) {
            $ageMinutes = ((Get-Date) - $recentWake.TimeCreated).TotalMinutes
            if ($ageMinutes -lt 45) {
                if ($recentWake.Properties.Count -ge 3 -and $recentWake.Properties[2].Value -eq 4) {
                    return $true
                }
                $msg = $recentWake.Message
                if ($msg -match "Wake Source:.*(Timer|NT TASK|Task Scheduler)") {
                    return $true
                }
            }
        }
    } catch {
        # Nel dubbio non assume wake da timer
    }
    return $false
}

function Test-UserActive {
    # 1. Se LogonUI.exe non sta girando e c'è explorer attivo, la sessione console è sbloccata
    $logonUI = Get-Process LogonUI -ErrorAction SilentlyContinue
    if (-not $logonUI) {
        $explorer = Get-Process explorer -ErrorAction SilentlyContinue
        if ($explorer) {
            return $true
        }
    }

    # 2. Controllo rapido via quser
    try {
        $quser = quser 2>$null
        foreach ($line in ($quser -split "`r?`n")) {
            if ($line -match "console\s+Active\s+(\.|\d+:\d+|\d+)") {
                if ($matches[1] -eq ".") {
                    return $true
                }
            }
        }
    } catch {}

    return $false
}

function Wait-ForNetwork {
    param([int]$TimeoutSeconds = 90)
    Write-Log "Verifico connettivita' di rete..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $attempts = 0
    while ((Get-Date) -lt $deadline) {
        $attempts++
        $tcpOk = $false
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $asyncResult = $tcp.BeginConnect("github.com", 443, $null, $null)
            $waitHandle = $asyncResult.AsyncWaitHandle.WaitOne(3000, $false)
            if ($waitHandle -and $tcp.Connected) {
                $tcp.EndConnect($asyncResult)
                $tcpOk = $true
            }
            $tcp.Close()
        } catch {
            $tcpOk = $false
        }

        if ($tcpOk) {
            Write-Log "Connettivita' di rete verificata con successo (tentativo $attempts)."
            return $true
        }

        $remaining = [math]::Round(($deadline - (Get-Date)).TotalSeconds)
        Write-Log "In attesa che la rete si riconnetta (tentativo $attempts, timeout tra ${remaining}s)..."
        Start-Sleep -Seconds 5
    }
    Write-Log "ATTENZIONE: Timeout di rete raggiunto ($TimeoutSeconds s). Procedo comunque."
    return $false
}

$wokeByTimer = Test-WokeByTimer
$lockAcquired = $false

# Segnala presenza in coda per evitare che l'altro task iberni mentre questo attende
New-Item -ItemType File -Path $waitingLock -Force | Out-Null

try {
    # Impedisce a Windows di rientrare in sospensione/ibernazione (timeout 2 min unattended)
    # mentre il task è in attesa del lock o in esecuzione.
    try {
        [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED) | Out-Null
    } catch {}

    # --------------------------- Mutua esclusione e coda ---------------------------
    $waitDeadline = (Get-Date).AddMinutes(45)
    while (Test-Path $lockFile) {
        $lockAgeMinutes = (New-TimeSpan -Start (Get-Item $lockFile).LastWriteTime -End (Get-Date)).TotalMinutes
        if ($lockAgeMinutes -gt 90) {
            Write-Log "Lock trovato ma stantio (>90 min): probabile crash precedente, lo rimuovo e procedo."
            Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
            break
        }
        if ((Get-Date) -ge $waitDeadline) {
            Write-Log "Tempo massimo di attesa lock (45 min) superato. Esco senza eseguire."
            exit 0
        }
        Write-Log "Lock attivo da un altro task. In attesa di rilascio (controllo tra 15s)..."
        Start-Sleep -Seconds 15
    }

    New-Item -ItemType File -Path $lockFile -Force | Out-Null
    $lockAcquired = $true
    Remove-Item $waitingLock -Force -ErrorAction SilentlyContinue

    Set-Location $RepoRoot
    Write-Log "=== Avvio task '$Task' ==="
    Write-Log "Rilevato avvio da Wake Timer: $wokeByTimer"

    # Attesa riconnessione Wi-Fi dopo il risveglio da S4
    Wait-ForNetwork -TimeoutSeconds 90

    $summaryPath = Join-Path $RepoRoot "results\publication-summary.json"

    if ($Task -eq "prediction") {
        # Equivalente allo step "Publish automatically due predictions"
        & $VenvPython scripts\generate_web_result.py `
            --operation auto --auto-kind prediction --publish `
            --summary-path $summaryPath 2>&1 | Tee-Object -Append -FilePath $logFile
        $exitCode = $LASTEXITCODE
    }
    else {
        # Equivalente al job "publish" (actual) + "Retrain and publish" di post-race-pipeline.yml
        & $VenvPython scripts\generate_web_result.py `
            --operation auto --auto-kind actual --publish `
            --summary-path $summaryPath 2>&1 | Tee-Object -Append -FilePath $logFile
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            $currentYear = (Get-Date).ToUniversalTime().Year
            $trainingRequired = & $VenvPython scripts\validate_workflow_changes.py training-required `
                --summary-path $summaryPath `
                --current-year $currentYear `
                --enabled "true"
            Write-Log "Training required: $trainingRequired"

            if ($trainingRequired.Trim() -eq "true") {
                & $VenvPython scripts\run_post_race_pipeline.py --year $currentYear 2>&1 | Tee-Object -Append -FilePath $logFile
                $exitCode = $LASTEXITCODE
            }
        }
    }

    if ($exitCode -ne 0) {
        Write-Log "FALLIMENTO reale (exit code $exitCode) - vedi $logFile"
        Send-FailureNotification -Summary "Task '$Task' terminato con errore reale. Log: $logFile"
        exit $exitCode
    }

    # exit 0 qui copre sia "pubblicato" sia "deferred/no-op" (es. dati non
    # ancora pronti, ResultsNotReadyError catturato internamente): e' il
    # comportamento normale, nessuna notifica necessaria.
    Write-Log "=== Task '$Task' completato (published/deferred/no-op) ==="

    # --------------------------- Commit & push ---------------------------
    # Sostituisce lo step "Commit and push" dei workflow: qui gira con le
    # TUE credenziali git locali (assumo tu abbia gia' push access configurato,
    # via credential manager o SSH key - nessun token da gestire).
    git add -A -- results webapp\api\data models data_files
    git diff --cached --quiet; $hasChanges = ($LASTEXITCODE -ne 0)
    if ($hasChanges) {
        git commit -m "chore(publication): $Task via local scheduler ($timestamp)"
        git push origin HEAD:main
        Write-Log "Push effettuato."
    } else {
        Write-Log "Nessuna modifica da pubblicare."
    }
}
catch {
    Write-Log "ECCEZIONE non gestita: $_"
    Send-FailureNotification -Summary "Eccezione non gestita in '$Task': $($_.Exception.Message)"
    exit 1
}
finally {
    if ($lockAcquired) {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $waitingLock -Force -ErrorAction SilentlyContinue

    # Rilascia la richiesta energetica
    try {
        [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
    } catch {}

    # ------------------ Rientro in Hibernation (Opzione A) ------------------
    if ($lockAcquired) {
        $otherWaiting = Get-ChildItem -Path $LockDir -Filter "waiting_*.lock" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne "waiting_$Task.lock" }
        if ($otherWaiting -and $otherWaiting.Count -gt 0) {
            Write-Log "Altro task PitWall in coda ($($otherWaiting.Name -join ', ')): non iberno per consentirne l'esecuzione."
        }
        else {
            $isUserActive = Test-UserActive
            if ($wokeByTimer -and (-not $isUserActive)) {
                Write-Log "Condizioni soddisfatte per rientro in ibernazione (wake via timer, nessun utente attivo sulla console)."
                Write-Log "Rientro in ibernazione (S4) tra 5 secondi..."
                Start-Sleep -Seconds 5
                try {
                    Start-Process -FilePath "shutdown.exe" -ArgumentList "/h" -NoNewWindow
                } catch {
                    [Win32.Power]::SetSuspendState($true, $false, $false)
                }
            } else {
                if ($isUserActive) {
                    Write-Log "Utente attivo rilevato sulla console: il PC resta acceso normalmente."
                } else {
                    Write-Log "Avvio non da Wake Timer (es. interattivo/manuale): il PC resta acceso."
                }
            }
        }
    }
}

<#
Get-ScheduledTaskInfo -TaskName 'PitWall-Oracle-Predictions'
Get-ScheduledTaskInfo -TaskName 'PitWall-Oracle-PostRace'

Start-ScheduledTask -TaskName "PitWall-Oracle-Predictions"
Start-ScheduledTask -TaskName "PitWall-Oracle-PostRace"
#>