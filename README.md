# Quantised LSTM-Based Intrusion Detection System for Automotive Controller Area Network  

This repository contains the training code and the backend HLS implementation for the **Quantised LSTM-Based Intrusion Detection System (QLSTM-IDS)** for Automotive Controller Area Network (CAN).  

## Repository Structure  
1. **Training Script**:  
   - Includes the model training code used to train the QLSTM-IDS on the **Open Car Hacking Dataset** and the **Survival Analysis Dataset** for intrusion detection.  

2. **Backend Implementation**:  
   - Features the hardware implementation of the final **LSTM + Dense model** using hardware building blocks available in the **FINN-HLSLib** library.  

3. **Deployment**:  
   - Contains the **bitstream file** for the IDS IP.  
   - Includes a **Jupyter notebook** to monitor latency and energy utilization of the deployed IP.  

---
