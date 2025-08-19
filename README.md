# Randomized PCA Forest for Outlier Detection

This project implements Randomized PCA Forest algorithm for outlier detection. The script allows you to perform outlier detection based on RPCA forest.

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage

To run the full algorithm, use the following command:
```bash
python rpcaforest-od.py [options]
```

## Arguments

The script accepts the following arguments:

- `-d`, `--dataset`: Path to the dataset CSV file. Default is `./data.csv`.
- `-p`, `--principalcomponents`: Number of principal components to use. Default is `1` for full algorithm and `5` for the fast implementation.
- `-l`, `--leafsize`: Maximum size of a node to be considered a leaf. Default is `10`.
- `-f`, `--forestsize`: Number of trees in the forest. Default is `40`.
- `-t`, `--threads`: Number of threads to use. Default is `4`.
- `-r`, `--recursionlimit`: Maximum number of recursions allowed. Default is `1000`.
- `-v`, `--verbos`: Set it to `1` to enable verbosity, `0` to disable it. Default is `1`.
- `-i`, `--load`: Filename to load the results from, `0` to disable loading.
- `-o`, `--load`: Filename to save the results to, `0` to disable saving.

## Example

Here is an example of how to run the script:
```bash
python rpcaforest.py -d ./ -p 5 -l 20 -f 50 -t 8 -r 2000 -v 1 -o output.odr
```

The output.odr file will containt the fitted forest, calculated probabilities, scores, and auc results for all of the datasets in the directory.

## Citation


If you use our method in your research, please cite the original paper:


```
@misc{rajabinasab2025randomizedpcaforestoutlier,
      title={Randomized PCA Forest for Outlier Detection}, 
      author={Muhammad Rajabinasab and Farhad Pakdaman and Moncef Gabbouj and Peter Schneider-Kamp and Arthur Zimek},
      year={2025},
      eprint={2508.12776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.12776}, 
}
```
