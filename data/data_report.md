# Data Imputation Report

- Generated: 2026-07-20 10:53:19
- Per-muni master: 5565 munis × 50 feature columns
- Imputation step: ENABLED

## Summary

- Numeric feature columns: 50
- Columns needing imputation: 5
- Columns fully populated (no imputation): 45
- Muni-values imputed: 9 (0.003% of all muni×feature cells)

## Per-column imputation

Only columns that required imputation are listed, most-imputed first. *Method breakdown* shows how the missing cells were actually filled (a climate column may be part KNN, part median residual).

| Column | Primary strategy | Imputed | % of munis | Method breakdown |
|---|---|---:|---:|---|
| wind_mean | idw_knn | 2 | 0.04% | idw_knn=2 |
| wet_bulb_F | idw_knn | 2 | 0.04% | idw_knn=2 |
| num_degreedays | idw_knn | 2 | 0.04% | idw_knn=2 |
| degreedays_streak | idw_knn | 2 | 0.04% | idw_knn=2 |
| uv_log_mean | idw_knn | 1 | 0.02% | idw_knn=1 |

## Strategy legend

- **idw_knn** — IDW-KNN haversine (k=8, inverse-squared distance) from nearest munis

## Fully populated columns (no imputation needed)

GDP_per_capita, accessibility, agri_gdp, agri_hhi, agri_nat_pct, agri_value_cagr_pct, coffee_pct, common_bean_pct, corn_pct, cotton_pct, deforestation_pct, dependency_ratio, disaster_pct, fire_pct, hh_mean_size, indigenous_pct, informal_worker_pct, male_pct_w, mean_income, mining_pct, ndvi, no_electricity_%, no_piped_water_%, no_sewage_%, pasture_pct, pct_18_35, pct_bolsa_familia, pct_pension, pct_urban_w, population_density, precip, primary_or_less_%, railway_density, road_density, secondary_%, semiarid_pct, soybean_pct, sugarcane_pct, temp, tertiary_%, total_pop, unemp_pct, urban_area_pct, z_from_baseline_precip, z_from_baseline_temp

---

*Note:* this report covers the central `dyad.impute()` step on the per-muni master. Feature loaders also fill their own gaps before the merge (e.g. land-use/transport/vulnerability → 0, climate temp/precip → IDW-KNN in-loader), and the derived columns added after imputation (accessibility, population_density, GDP_per_capita) are not covered here.
