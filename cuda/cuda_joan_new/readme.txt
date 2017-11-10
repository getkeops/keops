
Le fichier test.cu est destiné à donner un exécutable pour tester un appel de convolution. Pour le compiler : 
nvcc -std=c++11 -o test test.cu

Le fichier GpuConv2D.cu permet de produire des fichiers ".so". Voir le fichier compile_dlls pour des exemples de compilation

2 points importants :
- pour l'instant il n'y a pas de paramètre d'échelle sigma dans ces codes. C'est parce que dans mes codes matlab, je divise par sigma les coordonnées avant de faire les convolutions. 
- l'ordre des variables est différent par rapport aux fichiers ".so" précédents. Je détaillerai ça plus tard...



	
	



