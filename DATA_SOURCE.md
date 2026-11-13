# Dataset provenance and license notice

## Active dataset: car_data.csv

Source: **Vehicle dataset**, Nehal Birla and collaborators, CarDekho v3.

- Dataset page: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
- Downloaded public CSV mirror: https://raw.githubusercontent.com/HarshKapadia2/car-details/main/data/car_details_v3.csv
- Mirror project and analysis: https://harshkapadia2.github.io/car-details/
- Retrieved during this update on 14 November 2026 (Pakistan local date).

The dataset page identifies the database license as Open Database and the contents license as Database Contents. This derived CSV is made available under the same **Open Database License (ODbL) v1.0** and **Database Contents License (DbCL) v1.0** terms. Preserve this attribution and notice when redistributing the derived dataset.

- ODbL: https://opendatacommons.org/licenses/odbl/1-0/
- DbCL: https://opendatacommons.org/licenses/dbcl/1-0/

This notice applies to the sourced/derived dataset, not an invented license for the author's code or original course CSV.

## Transformation

The mirror has 8,128 rows. Only the following fields were retained:

| Source field | Output field | Treatment |
| --- | --- | --- |
| year | Year | Manufacturing year |
| km_driven | Mileage | Odometer distance in km, NOT the source fuel-economy field named mileage |
| engine | EngineSize | Extract numeric CC value and divide by 1,000 to obtain litres |
| selling_price | Price | Listed price in Indian rupees (INR); no currency conversion |

Rows missing required values or with invalid year, negative distance, nonpositive engine or price were discarded. Exact duplicate rows across these four selected columns were removed, leaving **6,559 rows**. Different source listings with identical selected values collapse to one row; no independent vehicle identifiers are available in this derived table. No fabricated, resampled or synthetic rows were added. The date range is 1994–2020; new model years are not observed training examples.

