param(
    [string]$RawDir = (Join-Path $PSScriptRoot "raw"),
    [string]$OutDir = (Join-Path $PSScriptRoot "readable")
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $OutDir | Out-Null

function Format-MessageText {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }

    $Text = $Text -replace "`r`n", "`n"
    $Text = $Text -replace "`r", "`n"
    return $Text.Trim()
}

Get-ChildItem -Path $RawDir -Filter "*.jsonl" | Sort-Object Name | ForEach-Object {
    $source = $_
    $threadName = $source.BaseName
    $sessionId = ""
    $rows = New-Object System.Collections.Generic.List[string]

    foreach ($line in Get-Content -LiteralPath $source.FullName) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }

        try {
            $entry = $line | ConvertFrom-Json
        } catch {
            continue
        }

        if ($entry.type -eq "session_meta") {
            $sessionId = [string]$entry.payload.id
            continue
        }

        if ($entry.type -ne "event_msg") {
            continue
        }

        $payloadType = [string]$entry.payload.type
        if ($payloadType -eq "thread_name_updated") {
            $threadName = [string]$entry.payload.thread_name
            continue
        }

        if ($payloadType -eq "user_message") {
            $message = Format-MessageText ([string]$entry.payload.message)
            if ($message) {
                $rows.Add("## User - $($entry.timestamp)")
                $rows.Add("")
                $rows.Add($message)
                $rows.Add("")
            }
            continue
        }

        if ($payloadType -eq "agent_message") {
            $message = Format-MessageText ([string]$entry.payload.message)
            if ($message) {
                $rows.Add("## Assistant - $($entry.timestamp)")
                $rows.Add("")
                $rows.Add($message)
                $rows.Add("")
            }
        }
    }

    $outPath = Join-Path $OutDir ($source.BaseName + ".md")
    $header = @(
        "# $threadName",
        "",
        "- Source: ``raw/$($source.Name)``",
        "- Session id: ``$sessionId``",
        "",
        "This readable transcript keeps user and assistant messages from the raw Codex JSONL export.",
        ""
    )
    Set-Content -LiteralPath $outPath -Value ($header + $rows) -Encoding UTF8
}
