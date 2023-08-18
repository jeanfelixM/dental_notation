#!/bin/bash

# Lancement du script init.sh
chmod +x init.sh
./init.sh

ROOT_PATH="."
for BIG_LAUNCH in $(find $ROOT_PATH -type f -name 'launch_simulation.sh' -not -path "*/input/*")
do

  # Sauvegarde du répertoire courant
  current_dir=$(pwd)

  # Changement de répertoire
  cd $(dirname $BIG_LAUNCH)

  # Lancement du gros launch_simulation.sh
  chmod +x ./launch_simulation.sh
  ./launch_simulation.sh
  
  # Retour au répertoire précédent
  cd $current_dir
  
  if [ $? -eq 0 ]
  then
    # Checkpoint : Ecriture du nom du script qui vient de s'exécuter dans un fichier de log
    echo "Terminé : $BIG_LAUNCH" >> launch.log
  else
    # Sinon, écrire un message d'erreur dans le fichier de log
    echo "Erreur : $BIG_LAUNCH n'a pas pu être exécuté correctement" >> launch.log
    continue
  fi
  
  
done

# Un message de fin d'exécution pour confirmer que tout s'est bien passé
echo "Tous les scripts ont été exécutés avec succès"
