# agrc/palletjack

![Build Status](https://github.com/agrc/python/workflows/Build%20and%20Test/badge.svg)
[![codecov](https://codecov.io/gh/agrc/python/branch/main/graph/badge.svg)](https://codecov.io/gh/agrc/python)

A library for updating AGOL feature services with data from SFTP shares.

Pallet jack: [forklift's](https://www.github.com/agrc/forklift) little brother.

## Installation

1. Activate your application's environment
1. `pip install ugrc-palletjack`

## Usage


## Development

1. Create a conda environment with arcpy, arcgis
   - `conda create -n palletjack`
   - `activate palletjack`
   - `conda install arcgis arcpy -c esri`
1. Clone the repo
1. Install in dev mode
   - `pip install -e .[tests]`

1. Navigate to [codecov.io](https://codecov.io/gh/agrc/python) and create a `CODECOV_TOKEN` [project secret](https://github.com/agrc/python/settings/secrets)
