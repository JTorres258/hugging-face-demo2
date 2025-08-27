# We can create commands to be used with `make` in a terminal

# Installing libraries
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# Applying some tests
test:
	python -m pytest -vvv --cov=hello --cov=greeting \
		--cov=smath --cov=web tests
	python -m pytest --nbval notebook.ipynb #tests our jupyter notebook
	#python -m pytest -v tests/test_web.py #if you just want to test web

debug:
	python -m pytest -vv --pdb #Debugger is invoked

one-test:
	python -m pytest -vv tests/test_greeting.py::test_my_name4

debugthree:
	python -m pytest -vv --pdb --maxfail=4 #drop to pdb for the frist three failures

format:
	black *.py

lint:
	pylint --disable=R,C *.py

# Do everything at once
all: install lint test format