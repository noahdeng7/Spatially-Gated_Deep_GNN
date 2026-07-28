# run_all.ps1 -- full pipeline on Windows without GNU make.
#
# Unlike `make`, this does NOT skip steps whose outputs are already current --
# it runs everything in order. That is fine because every step is idempotent,
# but the raster passes in 03 and 05 are slow, so prefer `make all` if you have
# make available.
#
#   .\scripts\run_all.ps1              # steps 01-08 (assumes raw data present)
#   .\scripts\run_all.ps1 -Download    # fetch raw inputs first
#   .\scripts\run_all.ps1 -From 04     # resume from a given step
#
# NOTE ON ORDER. 08_income.py runs BEFORE 07_assemble.py. The number is its
# position in the source tree, not in the build: income was added after the
# assembler existed, and the assembler consumes interim/income.parquet like
# every other covariate. The Makefile expresses this as a prerequisite; here it
# is expressed by the order of $steps, and -From resolves against that order
# rather than against the numeric label -- so `-From 08` still runs 07 after it.

[CmdletBinding()]
param(
    [switch]$Download,
    [string]$From = "01"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

$steps = @(
    @{ id = "01"; script = "src/01_flows.py";       name = "migration flows (spine)" },
    @{ id = "02"; script = "src/02_population.py";  name = "population" },
    @{ id = "03"; script = "src/03_distance.py";    name = "distance (slow: raster pass)" },
    @{ id = "04"; script = "src/04_gdp.py";         name = "GDP per capita" },
    @{ id = "05"; script = "src/05_climate.py";     name = "climate normals (slow: 24 raster passes; NOT the panel source by default)" },
    @{ id = "05b"; script = "src/climate_loader.py"; name = "climate ERA5-Land via Earth Engine (panel source; needs GEE auth)" },
    @{ id = "06"; script = "src/06_demography.py";  name = "youth share" },
    @{ id = "08"; script = "src/08_income.py";      name = "household income (ICMM)" },
    @{ id = "07"; script = "src/07_assemble.py";    name = "assemble dyadic panel" }
)

# Resolve -From to a POSITION in $steps, not to a number. See the order note above.
$startAt = [array]::FindIndex([object[]]$steps, [Predicate[object]] { $args[0].id -eq $From })
if ($startAt -lt 0) {
    Write-Host "Unknown -From '$From'. Valid: $(($steps | ForEach-Object { $_.id }) -join ', ')" -ForegroundColor Red
    exit 1
}

try {
    if ($Download) {
        Write-Host "`n=== 00  download raw inputs ===" -ForegroundColor Cyan
        python src/00_download.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`nDownload reported missing URLs (exit $LASTEXITCODE)." -ForegroundColor Yellow
            Write-Host "Fill them into config.yaml under downloads:, then re-run." -ForegroundColor Yellow
            exit $LASTEXITCODE
        }
    }

    for ($i = 0; $i -lt $steps.Count; $i++) {
        $step = $steps[$i]
        if ($i -lt $startAt) {
            Write-Host "--- $($step.id)  skipped (before -From $From)" -ForegroundColor DarkGray
            continue
        }
        Write-Host "`n=== $($step.id)  $($step.name) ===" -ForegroundColor Cyan
        python $step.script
        $code = $LASTEXITCODE

        # Exit 2 is DecisionRequired -- a deliberate stop, not a crash.
        if ($code -eq 2) {
            Write-Host "`nStep $($step.id) stopped for a human decision." -ForegroundColor Yellow
            Write-Host "Resolve it, then resume with: .\scripts\run_all.ps1 -From $($step.id)" -ForegroundColor Yellow
            exit 2
        }
        if ($code -ne 0) {
            Write-Host "`nStep $($step.id) FAILED (exit $code). Pipeline halted." -ForegroundColor Red
            Write-Host "Resume after fixing with: .\scripts\run_all.ps1 -From $($step.id)" -ForegroundColor Yellow
            exit $code
        }
    }

    Write-Host "`nPIPELINE COMPLETE" -ForegroundColor Green
    Write-Host "  panel    data/processed/mx_dyadic.parquet"
    Write-Host "  positive data/processed/mx_dyadic_positive.csv   <- the shareable deliverable"
    Write-Host "  codebook data/processed/mx_dyadic_codebook.md"
    Write-Host "  reports  reports/"
    Write-Host "  log      logs/build.log"
}
finally {
    Pop-Location
}
