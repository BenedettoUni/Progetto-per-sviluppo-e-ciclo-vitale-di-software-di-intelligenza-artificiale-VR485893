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

Pipeline github actions CI/CD
1) Clona il repository
2) Setup python
3) Installa dipendenze
4) Linting
5) Unit test
6) Build package Python
7) Build Docker image
8) Run the docker container to train
9) Extract model.pt file from the container
10) Upload final tensor.pt as artifact
11) Clean up container
