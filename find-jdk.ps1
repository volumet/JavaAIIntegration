# Dot-sourced by run.ps1 and onnx-genai/chat.ps1. Finds a JDK that is new enough WITHOUT touching the
# system JAVA_HOME (other projects on this machine use older JDKs; IntelliJ manages JDKs per project).
#
# Order: JAVA_HOME if it is new enough, else the newest suitable JDK that IntelliJ downloaded to ~/.jdks.

function Get-JdkMajor([string]$jdkHome) {
    $release = Join-Path $jdkHome 'release'
    if (Test-Path $release) {
        $m = Select-String -Path $release -Pattern '^JAVA_VERSION="(\d+)' | Select-Object -First 1
        if ($m) { return [int]$m.Matches[0].Groups[1].Value }
    }
    return 0
}

function Resolve-JdkHome([int]$MinVersion = 25) {
    if ($env:JAVA_HOME -and (Get-JdkMajor $env:JAVA_HOME) -ge $MinVersion) { return $env:JAVA_HOME }

    $candidates = Get-ChildItem (Join-Path $env:USERPROFILE '.jdks') -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\java.exe') } |
        ForEach-Object { [pscustomobject]@{ Home = $_.FullName; Major = Get-JdkMajor $_.FullName } } |
        Where-Object { $_.Major -ge $MinVersion } |
        Sort-Object Major -Descending
    if (-not $candidates) {
        throw "No JDK $MinVersion+ found: JAVA_HOME is '$env:JAVA_HOME' and ~/.jdks has none. Download one in IntelliJ (Project Structure > SDK)."
    }
    $pick = @($candidates)[0]
    Write-Host "Using $($pick.Home) (JDK $($pick.Major)); JAVA_HOME is left unchanged." -ForegroundColor DarkGray
    return $pick.Home
}
