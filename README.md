# WasteBins
This is a prototype for the microprocessors and microelectronics lab.

## Setup notes (Windows)

- Use a virtual environment. From the repo root:
	- Create: `python -m venv venv`
	- Activate: `venv\\Scripts\\activate`
- Install dependencies: `pip install -r waste_manager/requirements.txt`
- scikit-learn: install the package named `scikit-learn` (not `sklearn`). If you see an error like "ModuleNotFoundError: No module named 'sklearn'" or pip fails on `sklearn`, use:
	- `pip install scikit-learn`
- Python 3.13: Recent versions of scikit-learn support Python 3.13. If you hit build errors, first upgrade build tools:
	- `python -m pip install --upgrade pip setuptools wheel`
	- Then retry `pip install scikit-learn`

When running any project scripts, ensure your venv is active so imports resolve correctly.
