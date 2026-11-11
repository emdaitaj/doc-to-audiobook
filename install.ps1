# doc-to-audiobook installer bootstrap for Windows.
# Part of doc-to-audiobook - https://github.com/emdaitaj/doc-to-audiobook
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [options]
#   (or just double-click install.cmd)
#
# Gets uv (a fast Python + package manager, installed for your user only, no
# admin needed) and hands over to install.py. Every option is passed through:
# see `install.ps1 --help`.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Uv {
    if ($env:UV -and (Test-Path $env:UV)) { return $env:UV }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:USERPROFILE\.local\bin\uv.exe", "$env:USERPROFILE\.cargo\bin\uv.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host "uv not found - installing it for this user (https://docs.astral.sh/uv/) ..."
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $uv = Find-Uv
    if (-not $uv) { throw "uv could not be installed. Get it from https://docs.astral.sh/uv/ and rerun." }
}
$env:UV = $uv

# install.py only needs the standard library; uv finds (or downloads) a Python to run it.
& $uv run --no-project --python ">=3.9" "$here\install.py" @args
exit $LASTEXITCODE
