#  Nonnegative matrix factorization from binary data via majorization-minimization

This repository contains the code for reproducing the experiments in our paper entitled [A majorization-minimization algorithm for nonnegative binary matrix factorization](https://arxiv.org/abs/2204.09741), published in the IEEE Signal Processing Letters in 2022.

After cloning or downloading this repository, you will have the .rda data for several datasets.

### Requirements

In order to run the experiments (which uses the R package logistic PCA), you need to install R. Detailed instructions [here](https://linuxize.com/post/how-to-install-r-on-ubuntu-20-04/), but minimally:

```
sudo apt install dirmngr gnupg apt-transport-https ca-certificates software-properties-common
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt install r-base
```

Then run R (e.g., in command line simply using the `R`), and install the corresponding pacakges:

```
install.packages('logisticPCA')
install.packages('rARPACK')
```

Finally, you also need to install the Python package `rpy2` for interfacing both.


### Reproducing the experiments

Now that you're all set, simply run the following scripts:

- `training.py` will train the WMF model (whether with content or not) and perform a grid search over the hyper-parameters to maximize the NDCG on the validation subset.

- `evaluation.py` will predict the ratings with all the models and compute and display the corresponding NDCG.

### Reference

<details><summary>If you use any of this code for your research, please cite our paper:</summary>
  
```latex
@inproceedings{Magron2022nbmf,  
  author={P. Magron and C. F{\'e}votte},  
  title={A majorization-minimization algorithm for nonnegative binary matrix factorization},  
  booktitle={IEEE ISignal Processing Letters},  
  year={2022}
}
```
</details>
