This part is still completely under development.. 
To be read and used by authors only !!!

If you are using a standard Ubuntu 16.04 configuration (nvcc7.5, etc.),
please consider adding the flag "-D_FORCE_INLINES"
to your compilation command (see e.g. the "compile" script).

La commande "compile" permet de compiler une formule arbitraire ; 
le fichier ".so" correspondant est placé dans le répertoire build

Pour vérifier que ça fonctionne, j'ai fait deux programmes test_link.cu 
et test_link_grad.cu qui testent une fonction et son gradient 

J'ai rajouté (12 décembre) :
- Un code python cudaconv.py adapté de celui du dossier pykp qui appelle un noyau 
compilé avec le module
- des noyaux matriciels Curl Free, Div Free et TRI pour dans le fichier kernel_library.h
- la possibilité d'utiliser des floats ou des doubles. Les fichiers compilés se
terminent maintenant en "_float.so" ou "_double.so". Par défaut la compilation 
produit une version float, sinon il faut faire "./compile FORMULE double"

Remarques :
- Je n'ai pas eu le temps de vérifier grand chose sur les sorties ; 
  je le ferai quand je pourrai comparer facilement avec les codes existants.
