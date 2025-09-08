Progetto per l'esame di sviluppo e ciclo vitale di software di intelligenza artificiale
Benedetto Tessarini, matricola VR485893

dataset usato: cifar-10: https://www.cs.toronto.edu/~kriz/cifar.html
nel progetto è disponibile un file eda che produce un istogramma e 10 immagini associate alla relativa classe

una volta clonata le repository:
1) make install per installare le dipendenze
2) python src/cifar_train.py per eseguire il training del modello, che sarà salvato in results/model.pt

tramite docker
1) docker build -t trainer .  per costruire l'immagine
2) docker run trainer         per eseguire il container

comandi make
make install -> installa dipendenze

make lint -> esegue linting 

make test -> esegue i test (nella cartella tests)

make build -> crea pacchetto python usando il file pyproject.toml

make clean -> pulisce cartelle temporanee

(make lint e make clean non funzionano su windows ma vanno su linux)
