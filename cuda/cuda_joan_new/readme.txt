
Le fichier test.cu est destiné à donner un exécutable pour tester un appel de convolution. Pour le compiler : 
nvcc -std=c++11 -o test test.cu
Les différents paramètres (type de noyau, de fonction scalaire, dimensions) peuvent être modifiés via les #define au début du fichier.

2 points importants :
- pour l'instant il n'y a pas de paramètre d'échelle sigma dans ces codes. C'est parce que dans mes codes matlab, je divise par sigma les coordonnées avant de faire les convolutions. 
- l'ordre des variables d'entrée est différent des autres codes Cuda. 



	
	



