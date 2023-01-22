# Create VENV (in repo directory)
python -m venv ./databricks-ml-associate

# Activate VENV (in repo directory)
source ./databricks-ml-associate/bin/activate

# Install requirements (make sure to source+activate)
pip install -r requirements.txt

# Upgrade pip (make sure to source activate)
python -m pip install --upgrade pip

# Uninstall unused library
pip uninstall library_name