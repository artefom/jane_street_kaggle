Jane Street Kaggle competition
============================

ML model for predicting returns on market trades given general market features. Part of Jane Street Market Prediction Kaggle competition. www.kaggle.com/c/jane-street-market-prediction

Usage
============================

To install project in editable mode (can be edited after installation locally),
run cd `jane_street_kaggle && pip install -e .`

After that, Jane Street Kaggle competition can be imported inside any python code as
```python
import jane_street_kaggle
# Jane Street Kaggle competition creates configuration file jane_street_kaggle.cfg from template (default_jane_street_kaggle.cfg
# inside current working directory
# You can now edit the jane_street_kaggle.cfg to change the configuration and rerun script

# Import main entrypoint
from jane_street_kaggle.train import train

# Run
train()
```

Project configuration
============================

Project uses jane_street_kaggle.cfg file for configuration, which can be overridden using environment variables.
The file is divied into several sections, grouping configuration parameters for convenience.

Each section can be acessed inside code as

```python
from jane_street_kaggle.configuration import conf

# Get value from file or environment variable
conf.get('section_name', 'variable name')
```

Values in configuration must be specified without quotes

To override default variables in .cfg file one can specify environment variables as
JANE__SECTION__VALUE=value

On startup Jane Street Kaggle competition will automatically parse those and use them instead of values from .cfg file 

See jane_street_kaggle/config_templates/default_jane_street_kaggle.cfg for detailed info about configuration

Docker execution
============================

Project comes with pre-configured docker file which will copy source code, install all libraries in cache-friendly fashion and run help message: `jane_street_kaggle --help`
