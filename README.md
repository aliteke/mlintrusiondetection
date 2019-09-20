# mlintrusiondetection
Python Scripts for Detecting Anomalies in Cloud Hypervisor using machine learning techniques. Dataset used by permission from University of Victoria, Information Security and Object Technology Research Lab (https://www.uvic.ca/engineering/ece/isot/)

### iostat_to_json_convertor.py
converts the *iostat*, *vmstat -a* and *vmstat -d* command logs, into JSON format. (These commands were run on a cloud hypervisor hosting vms under attack)

### anomalyscore.py
Takes JSON formatted data as input and detects anomalies in the CPU/Memory/Disk Usage.
This code makes use of Kullback-Leibler Divergence for calculating probabilistic distance between sequential windows.
Results are published in 2019 IEEE Conference on Communications and Network Security (CNS), 5th Workshop on Security and Privacy in the Cloud (SPC'19), June 2019. Washington D.C. (https://cns2019.ieee-cns.org/workshop/spc-5th-ieee-workshop-security-and-privacy-cloud-2019/program)


### anomalyscore_HankelMatrix.py
We are using Hankel Matrices for each window, and calculate the singular value distribution for each Hankel matrix compared to previous windows and come up with a anomaly score in this code. Results are submitted as an academic paper to The 2020 American
Control Conference, pending review. (http://acc2020.a2c2.org/)



*Code in this repository contains shared work of Xinfinity Research Group (https://www.cs.sunyit.edu/~chiangc/xinfinity/) at SUNY Polytechnic Institute*
