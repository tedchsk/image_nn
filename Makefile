envsave: 
	conda env update --file environment.yml --prune
	# conda env export > environment.yml
envload:
	conda env create --name python39 --file environment.yml
