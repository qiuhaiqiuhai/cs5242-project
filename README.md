# cs5242-project

1. 在本地建一个../preprocessed_data/ 的dir
2. run data_preprocess.py
3. run main.py



# google cloud commands

0. connect to instance
https://console.cloud.google.com/compute/instances?project=cs5242-project-219611
# login and select instance{cs5242} and zone{us-east1-c}, first time login only)
gcloud init --zone us-east1-c

# list all the VMs
gcloud compute instances list

# connect to ssh
gcloud compute ssh cs5242

1. upload/download file
# go to the local directory with the data to upload
gcloud compute scp {local_file_dir} cs5242:/home/xiaogouman/... --zone us-east1-c
gcloud compute scp cs5242:/home/xiaogouman/... {local_dir} --zone us-east1-c

2. jupyter notebook
# on local, connect to remote jupyter(make sure the ~/.ssh/google_compute_engine exists)
ssh -i ~/.ssh/google_compute_engine -L 8899:localhost:8888 xiaogouman@104.196.170.68

# on remote, open jupyter notebook
jupyter notebook

# on local
localhost:8899
