This part is still completely under development.. 
To be read and used by authors only !!!

La commande "compile" permet de compiler une formule arbitraire ; 
le fichier ".so" correspondant est placé dans le répertoire build
Pour vérifier que ça fonctionne, j'ai fait deux programmes test_link.cu 
et test_link_grad.cu qui testent une fonction et son gradient 

Remarques :
- J'ai corrigé un problème (9 décembre) avec les ordres des variables, 
  jusque là ça fonctionnait mais ça mélangeait les variables, donc ça calculait n'importe quoi.
- Je n'ai pas eu le temps de vérifier grand chose sur les sorties ; 
  je le ferai quand je pourrai comparer facilement avec les codes existants.
