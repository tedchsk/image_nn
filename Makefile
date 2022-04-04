envsave: 
	# conda env update --file environment.yml --prune --from-history
	conda env export > environment.yml --from-history
envload:
	conda env create --name python39 --file environment.yml
test:
	pytest -m "not slow"
test_all:
	pytest
