FYI: Files in repo still not changed

## [SwissLog: Robust Anomaly Detection and Localization for Interleaved Unstructured Logs](https://yuxiaoba.github.io/publication/swisslog22/swisslog22.pdf)

**Abstract**—Modern distributed systems generate interleaved logs when running in parallel. Identifiers (ID) are always attached to them
to trace running instances or entities in logs. Therefore, log messages can be grouped by the same IDs to help anomaly detection and
localization. The existing approaches to achieve this still fall short meeting these challenges: 1) Log is solely processed in single
components without mining log dependencies. 2) Log formats are continually changing in modern software systems. 3) It is challenging
to detect latent performance issues non-intrusively by trivial monitoring tools. To remedy the above shortcomings, we propose
SwissLog, a robust anomaly detection and localization tool for interleaved unstructured logs. SwissLog focuses on log sequential
anomalies and tries to dig out possible performance issues. SwissLog constructs ID relation graphs across distributed components and
groups log messages by IDs. Moreover, we propose an online data-driven log parser without parameter tuning. The grouped log
messages are parsed via the novel log parser and transformed with semantic and temporal embedding. Finally, SwissLog utilizes an
attention-based Bi-LSTM model and a heuristic searching algorithm to detect and localize anomalies in instance-granularity,
respectively. The experiments on real-world and synthetic datasets confirm the effectiveness, efficiency, and robustness of SwissLog.
**Index Terms**—deep learning; log parsing; anomaly detection; anomaly localization; log correlation

## Datasets
This demo adopts logpai benchmark. [Logpai](https://github.com/logpai/logparser) adopts 16 real-world log datasets ranging from distributed systems, supercomputers, operating systems, mobile systems, server applications, to standalone software including HDFS, Hadoop, Spark, Zookeeper, BGL, HPC, Thunderbird, Windows, Linux, Android, HealthApp, Apache, Proxifier, OpenSSH, OpenStack, and Mac. The above log datasets are provided by [LogHub](https://github.com/logpai/loghub). Each dataset contains 2,000 log samples with its ground truth tagged by a rule-based log parser.


## Models 
Original model SwissLog from the article. 
It's based on attention-based Bi-LSTM model and a heuristic searching algorithm

## Pipeline

![image](https://github.com/Regylirovshik/mlops_project/assets/99333239/9c569831-cb8f-40df-8189-fb0cc15ead62)
