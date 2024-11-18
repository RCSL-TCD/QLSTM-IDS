This repository contains the training code and the backend HLS implementation for the Quantised LSTM based Intrusion Detection System for Automotive Controller Area Network.

1. The training script comprises of model training code used to train the model on the open Car hacking dataset and the Survival analysis dataset for intrusion detection.
2. The backend implementation comprises of the hardware implementation of the final LSTM + Dense model using hardware building blocks available in the finn-hlslib library.
3. The deployment folder comprises of the bit file for the IDS IP and the jupyter notebook comprises of the deployment script to monitor latency and energy utilisation of the IP.
