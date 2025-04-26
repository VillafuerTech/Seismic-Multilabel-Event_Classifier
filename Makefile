.PHONY: env run-notebooks

# Crear entorno Conda
env:
	conda env create -f environment.yml

# Ejecutar todos los notebooks con papermill\ nrun-notebooks:
	mkdir -p outputs
	for NB in notebooks/*.ipynb; do \
	  papermill $$NB outputs/$$(basename $${NB%%.ipynb})_output.ipynb; \
	done
