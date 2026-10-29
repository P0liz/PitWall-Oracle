<#
.SYNOPSIS
    Da eseguire UNA VOLTA, come Amministratore, per registrare i due task
    schedulati che sostituiscono publish-prediction.yml e post-race-pipeline.yml.

.NOTES
    Orari: la logica interna (scheduler.py::choose_due_operations) accetta
    una finestra di 0-15h prima della sessione per le prediction e da 24h+
    dopo l'inizio sessione per gli actual/retrain. Non serve quindi replicare
    al minuto gli orari UTC dei cron di GitHub Actions: bastano orari "wall
    clock" comodi. Qui uso l'equivalente CEST (UTC+2, valido per la stagione
    F1 che cade quasi interamente in ora legale). Se ti infastidisce lo
    scarto di un'ora in inverno, ignoralo pure: rientra ampiamente nella
    tolleranza delle finestre sopra.
#>

#Requires -RunAsAdministrator

$ScriptPath = "C:\Users\ASUS\Github\PitWall-Oracle\scripts\run_pitwall_task.ps1"
$WorkingDir = "C:\Users\ASUS\Github\PitWall-Oracle"

# --------------------------------------------------------------------------
# Configurazione energetica: abilita Wake Timers per consentire il risveglio
# dall'ibernazione (S4) sia a corrente (AC) sia a batteria (DC).
# In Windows di default è spesso "2 = Solo importanti", che blocca i task utente.
# Impostando a "1 = Abilita", Windows sveglierà il sistema per i task schedulati.
# --------------------------------------------------------------------------
Write-Host "Configurazione Wake Timers nel piano energetico attivo..."
# SUB_SLEEP = 238c9fa8-0aad-41ed-83f4-97be242c8f20 | RTCWAKE = bd3b718a-0680-4d9d-8ab2-e1d2b4ac806d
powercfg /SETACVALUEINDEX SCHEME_CURRENT 238c9fa8-0aad-41ed-83f4-97be242c8f20 bd3b718a-0680-4d9d-8ab2-e1d2b4ac806d 1
powercfg /SETDCVALUEINDEX SCHEME_CURRENT 238c9fa8-0aad-41ed-83f4-97be242c8f20 bd3b718a-0680-4d9d-8ab2-e1d2b4ac806d 1
powercfg /SETACTIVE SCHEME_CURRENT

# --------------------------------------------------------------------------
# Task 1: Predictions -> ogni 6h
# --------------------------------------------------------------------------
$predAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Task prediction" `
    -WorkingDirectory $WorkingDir

$predTriggers = @(
    New-ScheduledTaskTrigger -Daily -At "02:00"
    New-ScheduledTaskTrigger -Daily -At "08:00"
    New-ScheduledTaskTrigger -Daily -At "14:00"
    New-ScheduledTaskTrigger -Daily -At "20:00"
)

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -Priority 4 `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)

# Esegue con i TUOI privilegi utente (necessari per git push con le tue
# credenziali), sia che tu sia loggato o meno (-LogonType S4U), con
# RunLevel Highest per consentire a run_pitwall_task.ps1 di rientrare
# in ibernazione (SeShutdownPrivilege) dopo l'esecuzione unattended.
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Highest

Register-ScheduledTask -TaskName "PitWall-Oracle-Predictions" `
    -Action $predAction -Trigger $predTriggers -Settings $settings -Principal $principal `
    -Description "Equivalente locale di .github/workflows/publish-prediction.yml" -Force

# --------------------------------------------------------------------------
# Task 2: Post-race (actual + retrain) -> ogni 12h
# --------------------------------------------------------------------------
$postAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -Task postrace" `
    -WorkingDirectory $WorkingDir

$postTriggers = @(
    New-ScheduledTaskTrigger -Daily -At "07:00"
    New-ScheduledTaskTrigger -Daily -At "19:00"
)

Register-ScheduledTask -TaskName "PitWall-Oracle-PostRace" `
    -Action $postAction -Trigger $postTriggers -Settings $settings -Principal $principal `
    -Description "Equivalente locale di .github/workflows/post-race-pipeline.yml (actual + retrain)" -Force

Write-Host "Registrati: PitWall-Oracle-Predictions (ogni 6h) e PitWall-Oracle-PostRace (ogni 12h)."
Write-Host "Verifica in Task Scheduler (taskschd.msc) e lancia un test manuale con:"
Write-Host "  Start-ScheduledTask -TaskName 'PitWall-Oracle-Predictions'"
