# Improving Detection Accuracy for Imbalanced Network Intrusion Classification using Cluster-based Under-sampling with Random Forests
##### Authors: Md. Ochiuddin Miah, Sakib Shahriar Khan, Swakkhar Shatabda, and Dewan Md. Farid

#### Abstract
Network intrusion classification in the imbalanced big data environment becomes a significant and important issue in information and communications technology (ICT) in this digital era. Presently, intrusion detection systems (IDSs) are commonly using tool to detect and prevent internal and external network attacks/ intrusions. IDSs are majorly bifurcated into host-based and network-based systems, and use pattern matching techniques to detect intrusions that known as misuse-based intrusion detection system. Machine learning (ML) and data mining (DM) algorithms are widely using for classifying intrusions in IDS over the last few decades. One of the major challenges for building IDS employing machine learning and data mining algorithms is to improve the intrusion classification accuracy and also reducing the false-positive rate. In this paper, we have introduced a new method for improving detection rate to classify minority-class network attacks/ intrusions using cluster-based under-sampling with Random Forest classifier. The proposed method is a multi-layer classification approach, which can process the highly imbalanced big data to correctly identify the minority/ rare class-intrusions. Initially, the proposed method classify a data point/ incoming data is attack/ intrusion or not (like normal behaviour), if it’s an attack then the proposed method try to classify attack type and later sub-attack type. We have used cluster-based under-sampling technique to deal with class-imbalanced problem and popular ensemble classifier Random Forest for addressing overfitting problem. We have used KDD99 intrusion detection benchmark dataset for experimental analysis and tested the performance of proposed method with existing machine learning algorithms like: Artificial Neural Network (ANN), na ̈ıve Bayes (NB) classifier, Random Forest, and Bagging techniques.

The source codes—used to overcome this problem—are publicly available at https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification.

&nbsp;

### 1. Download Package
#### 1.1. Direct Download
We can directly [download](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification) by clicking the link.

> **Note:** The package will download in zip format `(.zip)` named `Network-Intrusion-Classification.zip`.


#### 1.2. Clone a GitHub Repository (Optional)

Cloning a repository syncs it to our local machine (Example for Linux-based OS). After clone, we can add and edit files and then push and pull updates.
- Clone over HTTPS: `user@machine:~$ git clone https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification`
- Clone over SSH: `user@machine:~$ git clone git@github.com:MdOchiuddinMiah/Network-Intrusion-Classification.git `

&nbsp;


### 2. How does it works (Machine Learning Perspective)?

Normal or attack detection
|----------------------|
|<img align="center" src="https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification/blob/main/ids_block_1.png" width="200" height="300" /> |

&nbsp;

Main attack types detection
|----------------------|
|<img align="center" src="https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification/blob/main/ids_block_2.png" width="200" height="320" /> |

&nbsp;

Final attack/ intrusion detection
|----------------------|
|<img align="center" src="https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification/blob/main/ids_block_3.png" width="250" height="400" /> |

&nbsp;

### 3. Pre‑processed Dataset:

The datasets are available on the open-source repository. Please [click](https://drive.google.com/drive/folders/1GSW0KDK9r_XNaa49ZD8k7eS63z3Ebk-5) for the download.

&nbsp;

### 4. Machine Learning:
The source code of the Machine Learning model are available on the open-source repository. Please [click](https://github.com/MdOchiuddinMiah/Network-Intrusion-Classification/tree/main/Network-Intrusion-Classifier) for the download.
