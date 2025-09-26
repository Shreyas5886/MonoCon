# MonoCon
A general deep metric learning framework for learning ultra-compact high-fidelity embeddings using monotonicity constraints

<details>
<summary><b>View Paper Abstract</b></summary>
Learning high-quality, robust, efficient, and disentangled representations is a central challenge in artificial intelligence (AI). Deep metric learning frameworks tackle this challenge primarily using architectural and optimization constraints. Here, we introduce a third approach that instead relies on functional constraints. Specifically, we present MonoCon, a simple framework that uses a small monotonic multi-layer perceptron (MLP) head attached to any pre-trained encoder. Due to co-adaptation between encoder and head guided by contrastive loss and monotonicity constraints, MonoCon learns robust, disentangled, and highly compact embeddings at a practically negligible performance cost. On the CIFAR-100 image classification task, MonoCon yields representations that are nearly 9x more compact and 1.5x more robust than the fine-tuned encoder baseline, while retaining 99\% of the baseline's 5-NN classification accuracy. We also report a 3.4x more compact and 1.4x more robust representation on an SNLI sentence similarity task for a marginal reduction in the STSb score, establishing MonoCon as a general domain-agnostic framework. Crucially, these robust, ultra-compact representations learned via functional constraints offer a unified solution to critical challenges in disparate contexts ranging from edge computing to cloud-scale retrieval. 

</details>

### Setting up the conda environment
Use the environment.yml file to create a conda environment by running the following command:

`conda env create -f environment.yml -n my-project-env`

Replace `my-project-env` with a name of your choice.

### Description of files
1.  All code is provided in python script as well as jupyter notebook format, in correspondingly named folders.
2.  The source code consists of four training scripts and four analysis scripts corresponding to the four models studied: CIFAR-10 MonoCon, CIFAR-100 MonoCon,     CIFAR-100 standard MLP head ablation study, and SNLI MonoCon. An additional script to analyze training dynamics of CIFAR-100 is also included.

### Note on file usage
The training scripts save model checkpoints periodically, as well as the overall best performing model. The four main analysis scripts analyze the saved best       model, whereas the CIFAR-100 training dynamics script analyzes saved checkpoints. Please ensure that the checkpoint loop in the training dynamics analysis script is consistent with checkpoint saving frequency in the main CIFAR-100 training script.
