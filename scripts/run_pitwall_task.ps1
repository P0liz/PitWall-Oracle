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
$lockFile  = Join-Path $LockDir "pitwall-publication.lock"

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

# --------------------------- Mutua esclusione ---------------------------
# Se l'altro job (prediction/postrace) e' ancora in corso, esci senza errore:
# verra' ritentato al prossimo poll schedulato, e questo NON e' un fallimento
# vero (choose_due_operations() ricontrollera' cosa e' ancora "due").
if (Test-Path $lockFile) {
    $lockAgeMinutes = (New-TimeSpan -Start (Get-Item $lockFile).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($lockAgeMinutes -lt 90) {
        Write-Log "Lock attivo (eta $([math]::Round($lockAgeMinutes,1)) min), un altro job e' in corso: esco senza eseguire."
        exit 0
    } else {
        Write-Log "Lock trovato ma stantio (>90 min): probabile crash precedente, lo rimuovo e procedo."
        Remove-Item $lockFile -Force
    }
}
New-Item -ItemType File -Path $lockFile -Force | Out-Null

try {
    Set-Location $RepoRoot
    Write-Log "=== Avvio task '$Task' ==="

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
    git add -A -- results webapp\api\data models
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
    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
}

<#
Get-ScheduledTaskInfo -TaskName 'PitWall-Oracle-Predictions'
Get-ScheduledTaskInfo -TaskName 'PitWall-Oracle-PostRace'

Start-ScheduledTask -TaskName "PitWall-Oracle-Predictions"
Start-ScheduledTask -TaskName "PitWall-Oracle-PostRace"
#>