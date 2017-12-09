This part is still completely under development.. To be read and used by authors only !!!

Remarques :
- J'ai corrigé un problème (9 décembre) avec les ordres des variables, jusque là ça fonctionnait mais ça mélangeait les variables, donc ça calculait n'importe quoi.
- Je n'ai pas eu le temps de vérifier grand chose sur les sorties ; je le ferai quand je pourrai comparer facilement avec les codes existants.
- Je poursuis mon idée de lier ce code générique avec l'autodiff de pytorch. J'ai rajouté une commande "compile" qui permet de compiler n'importe quelle formule :
Par exemple :
./compile "Scal<Square<Scalprod<X<2,4>,Y<3,4>>>,GaussKernel<P<0>,X<0,3>,Y<1,3>,Y<4,3>>>"
