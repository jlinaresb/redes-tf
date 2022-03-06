
Funciona con Python 3.7

	- Instala esa versión de Python:

		sudo apt update
		sudo apt install software-properties-common
		sudo add-apt-repository ppa:deadsnakes/ppa
		sudo apt install python3.7

		
	- Instala los paquetes en las versiones que por requirements.txt (hay más de los que necesitas, pero no te hace daño instalarlos todos, es el requirements de mi tesis):
		
		python3.7 -m pip install -r requirements.txt


Tarda un poco, si lo quieres lanzar en Cesga dímelo que te hago un .sh que funcione :)


Este ejemplo es una caca porque realmente es un problema de binary_cross_validation, pero lo he adaptado xddd