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
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 10)

# Esegue con i TUOI privilegi utente (necessari per git push con le tue
# credenziali), sia che tu sia loggato o meno (-LogonType S4U).
# In register_pitwall_tasks.ps1, sostituisci la riga del principal con:
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

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
