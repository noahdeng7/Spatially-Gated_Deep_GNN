# =============================================================================
# mx-migration -- one command reproduces everything
#
#   make all      full pipeline: download -> interim tables -> processed panel
#   make test     unit tests
#   make clean    remove interim + processed (raw/ is NEVER touched)
#
# Each step is a file target keyed on its output, so `make all` re-runs only
# what is stale. Every step is independently runnable and idempotent:
#
#   python src/03_distance.py
#
# Windows without GNU make: run scripts/run_all.ps1, or invoke the steps in
# numeric order by hand -- they are ordinary scripts with no make magic in them.
# =============================================================================

PYTHON  ?= python
SRC     := src
RAW     := data/raw
INTERIM := data/interim
PROC    := data/processed

.DEFAULT_GOAL := all
.PHONY: all download flows population distance gdp income climate climate-gee \
        demography assemble test lint clean clean-interim clean-processed \
        distclean dirs status help

# --- step outputs -----------------------------------------------------------
FLOWS      := $(INTERIM)/flows_dyadic.parquet
POPULATION := $(INTERIM)/population.parquet
DISTANCE   := $(INTERIM)/distance.parquet
GDP        := $(INTERIM)/gdp.parquet
INCOME     := $(INTERIM)/income.parquet
CLIMATE     := $(INTERIM)/climate.parquet       # 05_climate.py (WorldClim/CHELSA zonal)
CLIMATE_GEE := $(INTERIM)/climate_gee.parquet   # climate_loader.py (ERA5-Land via GEE)
DEMOGRAPHY := $(INTERIM)/demography.parquet
PANEL      := $(PROC)/mx_dyadic.parquet

# Which climate table feeds the panel. MUST agree with config.climate.panel_source:
#   $(CLIMATE_GEE) <-> era5_gee (default)   |   $(CLIMATE) <-> worldclim_zonal
# Override on the command line, e.g.  make CLIMATE_TABLE=$(CLIMATE) all
CLIMATE_TABLE ?= $(CLIMATE_GEE)

# --- shared code: touching these invalidates every step ---------------------
COMMON := $(SRC)/common.py $(SRC)/geo.py $(SRC)/zonal.py config.yaml

all: $(PANEL) ## build the full dyadic panel

dirs:
	@mkdir -p $(RAW) $(INTERIM) $(PROC) logs reports

# --- 00 : download ----------------------------------------------------------
# Not a prerequisite of the build targets on purpose. Raw downloads are large,
# slow, and several have no verified URL yet -- so `make all` must not silently
# retrigger them. Run it explicitly.
download: dirs ## fetch raw inputs (prints TODOs for unverified URLs)
	$(PYTHON) $(SRC)/00_download.py

download-status: ## show which raw inputs are present, missing, or lack a URL
	$(PYTHON) $(SRC)/00_download.py --list

# --- 01 : flows (the spine) -------------------------------------------------
$(FLOWS): $(SRC)/01_flows.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/01_flows.py
flows: $(FLOWS) ## build O-D migration flows from census microdata

# --- 02 : population --------------------------------------------------------
$(POPULATION): $(SRC)/02_population.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/02_population.py
population: $(POPULATION) ## municipal population, 2015 and 2020

# --- 03 : distance ----------------------------------------------------------
$(DISTANCE): $(SRC)/03_distance.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/03_distance.py
distance: $(DISTANCE) ## population-weighted centroids and geodesic distance

# --- 04 : gdp (depends on population for the per-capita denominator) --------
$(GDP): $(SRC)/04_gdp.py $(POPULATION) $(COMMON) | dirs
	$(PYTHON) $(SRC)/04_gdp.py
gdp: $(GDP) ## PPP-adjusted municipal GDP per capita, two sources

# --- 08 : income (INEGI ICMM 2020; reads raw ICMM + ITER, no interim dep) ---
$(INCOME): $(SRC)/08_income.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/08_income.py
income: $(INCOME) ## mean household income per municipality (INEGI ICMM 2020)

# --- 05 : climate (WorldClim/CHELSA zonal normals) --------------------------
$(CLIMATE): $(SRC)/05_climate.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/05_climate.py
climate: $(CLIMATE) ## population-weighted temperature and precipitation (normals)

# --- 05b : climate via ERA5-Land / Google Earth Engine ----------------------
# No local raster download: the reduction runs server-side on GEE and returns
# only the per-municipality numbers. Needs Earth Engine auth and, for the
# land-mask gap fill, the centroids that 03_distance.py writes.
$(CLIMATE_GEE): $(SRC)/climate_loader.py $(DISTANCE) $(COMMON) | dirs
	$(PYTHON) $(SRC)/climate_loader.py
climate-gee: $(CLIMATE_GEE) ## ERA5-Land 2016-2020 pop-weighted temp/precip via Earth Engine

# --- 06 : demography --------------------------------------------------------
$(DEMOGRAPHY): $(SRC)/06_demography.py $(COMMON) | dirs
	$(PYTHON) $(SRC)/06_demography.py
demography: $(DEMOGRAPHY) ## youth share of working-age population

# --- 07 : assemble ----------------------------------------------------------
$(PANEL): $(SRC)/07_assemble.py $(FLOWS) $(POPULATION) $(DISTANCE) \
          $(GDP) $(INCOME) $(CLIMATE_TABLE) $(DEMOGRAPHY) $(COMMON) | dirs
	$(PYTHON) $(SRC)/07_assemble.py
assemble: $(PANEL) ## join everything into the final dyadic panel

# --- quality ----------------------------------------------------------------
test: ## run unit tests
	$(PYTHON) -m pytest tests/ -v

test-quick: ## run unit tests, quiet
	$(PYTHON) -m pytest tests/ -q

lint: ## syntax-check every pipeline script
	$(PYTHON) -m compileall -q $(SRC)

# --- housekeeping -----------------------------------------------------------
status: ## show which pipeline outputs exist
	@echo "--- interim ---"
	@ls -lh $(INTERIM)/*.parquet 2>/dev/null || echo "  (none)"
	@echo "--- processed ---"
	@ls -lh $(PROC)/*.parquet 2>/dev/null || echo "  (none)"
	@echo "--- reports ---"
	@ls -1 reports/ 2>/dev/null || echo "  (none)"

clean-interim: ## delete interim tables
	rm -rf $(INTERIM)/*.parquet $(INTERIM)/raster_cache

clean-processed: ## delete the final panel and codebook
	rm -rf $(PROC)/*.parquet $(PROC)/*.md

clean: clean-interim clean-processed ## delete all derived data (raw/ untouched)
	rm -rf logs/*.log reports/*

# NOTE: there is deliberately no target that deletes data/raw/. Those files are
# slow to fetch, some have no verified URL, and `never edited` is the contract.
distclean:
	@echo "Refusing: data/raw/ is never deleted by the build."
	@echo "Some inputs are multi-GB and several have no verified download URL."
	@echo "Remove it by hand if you really mean to."
	@exit 1

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
