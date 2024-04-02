# Distributed dynamic channel selection

## About

This Python project is the PyTorch implementation of the distributed dynamic channel selection method proposed [1]. The purpose is to derive an optimal, instance-wise subset of features to employ for inference, with constraints being put on how often each feature is allowed to be selected on average. Its distributed implementation allows it to be employed for bandwith reduction in a wireless sensor network, by dynamicaly assigning only a subset of sensor to transmit their data to a fusion center. To boost performance, the model is also extended with the Dynamic Spatial Filtering module of Banville et al [2]. As an additional benchmark, an implementation of the greedy conditional mutual information (CMI) method of Covert et al. is also provided [3]. The dataset used int this work is the High Gamma Dataset [4], which can be accessed at https://github.com/robintibor/high-gamma-dataset. For the purposes of our application, this data has been preprocessed such that it emulates the data that would be measured in a wireless EEG sensor network, as described in [1]. The code in this repository operates on this preprocessed data, which can be found at https://zenodo.org/records/10907610. To run the code, first download this data into the Data folder.

## Usage

To run the code, first install the conda environment from the *environment.yml* file.
```
conda create --name myenv --file environment.yml
```
Training the distributed dynamic channel selection model can be done with *train_dynamic.py* and the greedy CMI method with *train_cmi.py*. An example for doing this with the default setting is included in the *run.sh* file.

 ## References
 
[1] Strypsteen, Thomas, and Alexander Bertrand. "A distributed neural network architecture for dynamic sensor selection with application to bandwidth-constrained body-sensor networks." arXiv preprint arXiv:2308.08379 (2023). <br />
[2] Banville, Hubert, et al. "Robust learning from corrupted EEG with dynamic spatial filtering." NeuroImage 251 (2022): 118994. <br />
[3] Covert, Ian Connick, et al. "Learning to maximize mutual information for dynamic feature selection." International Conference on Machine Learning. PMLR, 2023. <br />
[4] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, “Deep learning with convolutional neural networks for EEG decoding and visualization,” Human brain mapping, vol. 38, no. 11, pp. 5391– 5420, 2017.
