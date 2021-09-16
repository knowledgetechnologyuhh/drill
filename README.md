# DRILL: Dynamic Representations for Imbalanced Lifelong Learning

This is the official code for our paper, presented at 
[ICANN 2021](https://e-nns.org/icann2021/). If you find this work useful, please cite our [paper](https://www2alt.informatik.uni-hamburg.de/wtm/publications/2021/AAW21/index.php):

```
@InProceedings{AAW21,
  author       = "Ahrens, Kyra and Abawi, Fares and Wermter, Stefan",
  title        = "DRILL: Dynamic Representations for Imbalanced Lifelong Learning",
  booktitle    = "Proceedings of the 30th International Conference on Artificial Neural Networks (ICANN 2021)",
  pages        = "409–-420",
  month        = "September",
  year         = "2021",
  publisher    = "Springer",
  series       = "LCNS"
  editor       = "Igor Farkaš, Paolo Masulli, Sebastian Otte, Stefan Wermter",
  key          = "ahrens2021drill",
  doi          = "10.1007/978-3-030-86340-1_33",
  url          = "https://www2.informatik.uni-hamburg.de/wtm/publications/2021/AAW21/DRILL_Paper.pdf"
}
```


## Setup / Requirements

* python >= 3.5
* pytorch >= 1.15

Navigate in your terminal to the project root, then install all required packages:

```
pip3 install -r requirements.txt
```

Next, download all text classification datasets:

```
cd data
source download_data.sh
```

**Note:** Mac OSX users that get the error message "command not found: wget" will have to install wget first, e.g. via homebrew:

```
brew install wget
```

## Logging using [Comet](https://www.comet.ml/) 

To log the results with comet.ml, export the following environment variables replacing `<>` with the corresponding account
properties:

```
export COMET_KEY=<YOUR COMET API KEY>
export COMET_WORKSPACE=<YOUR WORSKPACE NAME>
export COMET_PROJ_NAME=cosmell_benchmarks
```

## Training

To train DRILL on the `YelpShort` dataset for debugging and testing:

```
python3 train.py --cfg_file configs/train.config --cfg_tag drill_soinn --updates 5 --batch_size 4 --max_len 20 --replay_rate 1 --replay_every 20 --max_age 10 --gpu 0
```

To train DRILL on the full dataset with the orders specified in the paper:

```
python3 train.py --cfg_file configs/train.config --cfg_tag drill_soinn --del_freq 1000 --order 1 --gpu 0
```

Optionally, you can specify the datasets used in custom orders, for example for the UCI Sentiment datasets:

```
# drill
python3 train.py --cfg_file configs/train.config --cfg_tag drill_soinn --meta_lr 0.00003 --inner_lr 0.001 --replay_every 100 --replay_rate 1 --updates 5 --del_freq 1000 --datasets imdbshort amazonshort yelpshort --gpu 0
# oml
python3 train.py --cfg_file configs/train.config --cfg_tag oml_soinn --meta_lr 0.00003 --inner_lr 0.001 --replay_every 100 --replay_rate 1 --updates 5 --datasets imdbshort amazonshort yelpshort --gpu 0
```

To run the hyper-parameter optimizer using Comet ML (Parzen-Rozenblatt Estimator in this case), specify the
`hyperparam_cfg_file` location ensuring that the `--learner` and `--memory` are compatible with the configuration file.
Note that the optimizer runs on comet.ml, therefore, make sure to export `COMET_KEY`, `COMET_WORKSPACE`,
and `COMET_PROJ_NAME`
in your environment. An example of using the optimzer:

```
python3 train.py --hyperparam_cfg_file configs/hyperopt.config --cfg_file configs/train.config --cfg_tag drill_soinn --meta_lr 0.00003 --inner_lr 0.001 --batch_size 4 --max_len 20 --replay_rate 1 --updates 5 --replay_every 20 --max_age 10 --gpu 0
```

## Notes

* To run OML instead of DRILL, either change `--cfg_tag oml_default` or specify `--learner oml`
* When specifying `--learner drill`, this defaults to DRILL_C (concatenation).
* To run a training session on the CPU, simply remove
  `--gpu 0` from the arguments. You can specify the gpu to be used by changing the `--gpu` id. Parallel GPU training is
  not supported due to the **higher** library for meta-learning. 