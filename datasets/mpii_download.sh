sudo apt install unzip

wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
tar -xvzf mpii_human_pose_v1.tar.gz
unzip mpii_human_pose_v1_u12_2.zip
rm mpii_human_pose_v1.tar.gz
rm mpii_human_pose_v1_u12_2.zip

echo "Start installing OpenCV"
pip install opencv-contrib-python
sudo apt-get install libgl1-mesa-glx

