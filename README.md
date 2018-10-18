# cs5242-project

1. 在本地建一个../preprocessed_data/ 的dir
2. run data_preprocess.py
3. run main.py

# update with branch shuman-1
1. go to cs5242-project
2. activate environment: $ env
3. if need to preprocess data, specify the parameters and: $ process
4. $ train


__google cloud commands__  
1. connect to instance  
https://console.cloud.google.com/compute/instances?project=cs5242-project-219611  
-> login and select instance{cs5242} and zone{us-east1-c}, first time login only)  
__gcloud init --zone us-east1-c__  

-> list all the VMs  
__gcloud compute instances list__  

-> connect to ssh  
__gcloud compute ssh cs5242__  

2. upload/download file  
-> go to the local directory with the data to upload  
__gcloud compute scp {local_file_dir} cs5242:/home/xiaogouman/... --zone us-east1-c__  
__gcloud compute scp cs5242:/home/xiaogouman/... {local_dir} --zone us-east1-c__  

3. jupyter notebook  
-> on local, connect to remote jupyter(make sure the ~/.ssh/google_compute_engine exists)  
__ssh -i ~/.ssh/google_compute_engine -L 8899:localhost:8888 xiaogouman@35.190.181.16__  

-> on remote, open jupyter notebook  
__jupyter notebook__  

-> on local  
__localhost:8899__
