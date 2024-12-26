sprng_interface.so : sprng_interface.cpp
	g++ -shared -fPIC -o sprng_interface.so sprng_interface.cpp -I/home/jgkim/sprng5/include -L/home/jgkim/sprng5/lib -lsprng
clean :
	rm -f *.so
	rm -f *.csv
	rm -f *.png
	rm -rf __pycache__
	rm -rf venv
venv :
	test -d venv || python3 -m venv venv
install: venv
	source venv/bin/activate && pip install -r requirements.txt
run1 : install
	mpirun -np 10 python potential.py &
run2 : install
	python3 error.py &
