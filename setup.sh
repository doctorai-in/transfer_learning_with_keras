sudo cp -rf transfer_learning_with_keras/docker-compose.yml .
sudo cp -rf transfer_learning_with_keras/Dockerfile .
sudo chmod -R 777 .

#install Docker for Debian: https://docs.docker.com/install/linux/docker-ce/debian/
sudo apt-get update
sudo apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg2 \
    software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/debian \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get -y --allow-unauthenticated install docker-ce docker-ce-cli containerd.io

#fix Docker sock issue: https://stackoverflow.com/questions/47854463/got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket-at-uni
sudo usermod -a -G docker $USER
#have to restart after this
