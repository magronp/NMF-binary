#  Content-aware music recommendation using the AVD model of musical preference

This repository contains the code for reproducing the experiments in our paper entitled [Leveraging the structure of musical preference in content-aware music recommendation](https://arxiv.org/abs/2010.00392), published at the IEEE International Conference on Audio, Speech and Signal Processing (ICASSP) I2021.

### Getting the data

After cloning or downloading this repository, you will need to get the data from the [Million Song Dataset](http://millionsongdataset.com/) to reproduce the results.

* The [meta-data file](http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db) for the whole set.
* The list of [unique tracks](http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt).
* The [playcounts](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip) from the Taste Profile set.

You will also need the pre-computed ESSENTIA data to extract the features, which are available [here](https://zenodo.org/record/3860557#.X5BuHJ1fg5m) (download the file `msd_played_songs_essentia.csv.gz`).

All the files should be unziped (if needed) and placed in the `data/` folder.
Note that you can change the folder structure, as long as you change the path accordingly in the code.

### Preprocessing the playcounts and features

Once you're set, simply execute the `prepare_data.py` script to produce a handful of files (notably splitting the dataset into training, validation and test subsets).
This will also extract the AVD features and create the corresponding split. 

Note that by runing the script `helpers/extrac_features.py`, you can also obtain the factor loadings (Table 1 in the paper) and obtain the songs with maximum / minimum AVD values.


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
