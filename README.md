Pour lancer le software, il faut d'abord installer anaconda, puis lancer les commandes suivantes:
```
conda env create -f dependance.yml
conda activate notationdent
pip install -e .
python main/mainui.py
```
Qui lancera donc l'interface graphique.
