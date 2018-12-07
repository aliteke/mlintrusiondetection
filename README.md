# mlintrusiondetection
Python Scripts for Detecting Anomalies in Cloud Hypervisor using machine learning techniques. Dataset used by permission from University of Victoria, Information Security and Object Technology Research Lab (https://www.uvic.ca/engineering/ece/isot/)

### iostat_to_json_convertor.py
converts the *iostat*, *vmstat -a* and *vmstat -d* command logs, into JSON format. (These commands were run on a cloud hypervisor hosting vms under attack)

### anomalyscore.py
Takes JSON formatted data as input and detects anomalies in the CPU/Memory/Disk Usage.

(This repository is shared work of Xinfinity Research Group (https://www.cs.sunyit.edu/~chiangc/xinfinity/) at SUNY Polytechnic Institute)
