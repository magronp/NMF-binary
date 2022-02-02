#  Nonnegative matrix factorization from binary data via majorization-minimization

This repository contains the code for reproducing the experiments in our paper entitled [Nonnegative matrix factorization from binary data via majorization-minimization](https://arxiv.org/abs/2010.00392), published in the IEEE Signal Processing Letters in 2022.

### Getting the data

After cloning or downloading this repository, you will have the .rda data for several datasets.
To conduct music recommendation experiments, you need to also get the data from the [Million Song Dataset](http://millionsongdataset.com/).
Its available to download directly [here](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip), the txt file should be placed  in the `data/` folder.
Note that you can change the folder structure, as long as you change the path accordingly in the code.

Once you're set, simply execute the `prepare_data.py` script to produce a handful of files for each dataset (notably splitting the dataset into training, validation and test subsets).

### Requirements

In order to run the experiments (which uses the R package logistic PCA), you need to install R. Detailed instructions here, but minimally:

```

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
@inproceedings{Magron2021,  
  author={P. Magron and C. F{\'e}votte},  
  title={Leveraging the structure of musical preference in content-aware music recommendation},  
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  year={2021},
  month={June}
}
```

Also note that part of this code is taken from the  [content_wmf](https://github.com/dawenl/content_wmf) repository.
Please consider citing the corresponding paper:

  
```latex
@inproceedings{Liang2015,
    author = {Liang, D. and Zhan, M. and Ellis, D.},
    title = {Content-aware collaborative music recommendation using pre-trained neural networks},
    booktitle = {Proc. International Society for Music Information Retrieval Conference (ISMIR)},
    year = {2015},
    month = {October}
}
```

</p>
</details>
