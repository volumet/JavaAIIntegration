<#
.SYNOPSIS
  Runs one lab with Java's source launcher. No build step is needed, only JDK 25+.
  Uses JAVA_HOME if it is 25+, else the newest 25+ JDK in ~/.jdks (where IntelliJ puts them).
.EXAMPLE
  ./run.ps1 lab1_vector/DotProduct
  ./run.ps1 lab3_tinynn/MatMul 1024
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Lab,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$LabArgs
)

. (Join-Path $PSScriptRoot 'find-jdk.ps1')
$java = Join-Path (Resolve-JdkHome 25) 'bin\java.exe'

$file = Join-Path $PSScriptRoot "labs/src/main/java/playground/$($Lab -replace '\.java$', '').java"
if (-not (Test-Path $file)) {
    Write-Error "No such lab: $file"
    exit 1
}

& $java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED $file @LabArgs
