This part is still completely under development.. 
To be read and used by authors only !!!

------------------------- Compilation options------------------------------------

If you are using a standard Ubuntu 16.04 configuration (nvcc7.5, etc.),
please consider adding the flag "-D_FORCE_INLINES" and/or "-D_MWAITXINTRIN_H_INCLUDED "
to your compilation command (see e.g. the various "compile" and "compile_XX" scripts)
as suggested in https://github.com/tensorflow/tensorflow/issues/1066#issuecomment-225937814

On top of it, specifying explicitly a version of GCC that is compatible with your CUDA
version may make a difference : try adding the "-ccbin gcc-4.9" option to your nvcc commands.


---------------------------Nouvelle arborescence----------------------------------

/ : les 'makeFile' et ce Readme
|
---- core : contient les fichiers autodiff, Gpu, Pack et formulas math et noyaux
|
---- build : repertoire qui contient les .o, .so et binaire
|
---- test : les fichiers d'exemples et test
|
---- Python : les fichiers contenant les bindings python. TODO: a mettre dans
                pykp quand cela sera stable.

----------------------------MakeFiles-----------------------------------------
Du nouveau!

on a maintenant un fichier cmake qui permet de gérer les compilations: 

pour tout compiler :

cd /build
cmake .. -DFORMULA="Scal<Square<Scalprod<_X<0,4>,_Y<3,4>>>,GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<4,3>>>" -D__TYPE__=float
make 

pour ne compiler que le mex file :

cd /build
cmake .. -DFORMULA="Scal<Square<Scalprod<_X<0,4>,_Y<3,4>>>,GaussKernel<_P<0>,_X<0,3>,_Y<1,3>,_Y<4,3>>>" -D__TYPE__=double
make mex_cuda


Je pense qu'on pourra s'affranchir des scripts qui passent par la sortie standard quand tout ca sera stabilisé...

#La commande "compile" permet de compiler une formule arbitraire ; 
#le fichier ".so" correspondant est placé dans le répertoire build
#
#La possibilité d'utiliser des floats ou des doubles. Les fichiers compilés se
#terminent maintenant en "_float.so" ou "_double.so". Par défaut la compilation 
#produit une version float, sinon il faut faire "./compile FORMULE double"
#
#Pour vérifier que ça fonctionne, j'ai fait deux programmes test_link.cu 
#et test_link_grad.cu qui testent une fonction et son gradient. Les commandes 
#"compile_XX" permettent de compiler les fichiers de tests correspondant 
#qui sont maintenant dans le sous-repertoire ./test/
#


------------------------------------ Python -----------------------------------
- Un code python cudaconv.py adapté de celui du dossier pykp qui appelle un noyau 
compilé avec le module



 
Remarques :
- des noyaux matriciels Curl Free, Div Free et TRI pour dans le fichier kernel_library.h
- Je n'ai pas eu le temps de vérifier grand chose sur les sorties ; 
  je le ferai quand je pourrai comparer facilement avec les codes existants.
