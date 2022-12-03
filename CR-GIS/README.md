
# Code for CR-GIS

> The implementation of [**CR-GIS: Improving Conversational Recommendation via Goal-aware Interest Sequence Modeling**](https://aclanthology.org/2022.coling-1.32/)

<img src="https://img.shields.io/badge/Venue-COLING--22-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/> <img src="https://img.shields.io/badge/Last%20Updated-2022--12--03-2D333B" alt="update"/>

## Usage

### Requirements

Install the required libraries as follows:

- `python==3.6.12`
- `torch==1.3.0+cu100`
- `torch-geometric==1.3.2`
- `nltk==3.4.5`
- `fuzzywuzzy==0.18.2`

### Dataset 

The preprocessed dataset can be available from [Google Drive](https://drive.google.com/drive/folders/1Y1E84U1lllt2wQ7DDf9pbkN16_Jja-I9?usp=share_link), and put into the `data/` dir.

### Training & Testing

- Run `bash train_on_opendialkg.sh` for training CR-GIS on **OpenDialKG** dataset.

- Run `bash train_on_tgredial.sh` for training CR-GIS on **TGReDial** dataset.

- More details can be found in the two scripts.

## Citation

If you find our work useful for your research, please kindly cite our paper as follows:

```
@inproceedings{zhou-etal-2022-cr,
    title = "{CR}-{GIS}: Improving Conversational Recommendation via Goal-aware Interest Sequence Modeling",
    author = "Zhou, Jinfeng  and
      Wang, Bo  and
      Yang, Zhitong  and
      Zhao, Dongming  and
      Huang, Kun  and
      He, Ruifang  and
      Hou, Yuexian",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.32",
    pages = "400--411"
```

