<#
.SYNOPSIS
  Interactive chat with the local ONNX model (lab 5). Builds first if needed.
.EXAMPLE
  ./onnx-genai/chat.ps1
  ./onnx-genai/chat.ps1 -Prompt "Explain the KV cache in two sentences"
  ./onnx-genai/chat.ps1 -Model D:\models\other-genai-model -MaxLength 4096
#>
param(
    [string]$Prompt = "",
    [string]$Model = "",
    [int]$MaxLength = 2048,
    [string]$System = ""
)
$ErrorActionPreference = 'Stop'
$root    = Split-Path $PSScriptRoot -Parent
$tools   = Join-Path $root 'tools'
$ortLib  = Join-Path $tools 'onnxruntime\onnxruntime-win-x64-1.32.0\lib\onnxruntime.dll'
$genaiLib = Join-Path $tools 'onnxruntime-genai\onnxruntime-genai-0.20.0-win-x64\lib\onnxruntime-genai.dll'
if (-not $Model) { $Model = Join-Path $root 'models\phi3-repo\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4' }

. (Join-Path $root 'find-jdk.ps1')
$jdkHome = Resolve-JdkHome 25
$java = Join-Path $jdkHome 'bin\java.exe'
$classes = Join-Path $PSScriptRoot 'target\classes'

# Interactive stdin doesn't go through the mvnd daemon well, so mvnd only builds, and java runs directly.
if (-not (Test-Path (Join-Path $classes 'playground\onnx\Chat.class'))) {
    Write-Host "Building onnx-genai (first run) ..."
    Push-Location $root
    $savedJavaHome = $env:JAVA_HOME
    $env:JAVA_HOME = $jdkHome   # only for this mvnd call; restored below so the terminal keeps its own
    try { mvnd -q -pl onnx-genai compile; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
    finally { $env:JAVA_HOME = $savedJavaHome; Pop-Location }
}

$javaArgs = @('--enable-native-access=ALL-UNNAMED', '-Dstdout.encoding=UTF-8', '-Dstdin.encoding=UTF-8',
              "-Dort.lib=$ortLib", "-Dgenai.lib=$genaiLib", '-cp', $classes,
              'playground.onnx.Chat', $Model, "--max-length=$MaxLength")
if ($Prompt) { $javaArgs += "--prompt=$Prompt" }
if ($System) { $javaArgs += "--system=$System" }
# Forward piped input (e.g. a scripted conversation) to java; otherwise java reads the keyboard.
if ($MyInvocation.ExpectingInput) { $input | & $java @javaArgs } else { & $java @javaArgs }
