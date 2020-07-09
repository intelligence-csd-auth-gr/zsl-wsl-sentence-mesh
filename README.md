# ZSL using sentence-based bioBERT embeddings

This repository contains the code for our [SETN 2020](http://www.eetn.gr/index.php/setn-2020-home) paper with title: *Zero-Shot Classification of Biomedical Articles with EmergingMeSH Descriptors*

In brief, we develop a Zero-shot learning (**ZSL**) mechanism ( *ZSLbioSentMax* ) that tries to perform on-the-fly classification examining text-raw data that stem from [PubMed](https://www.ncbi.nlm.nih.gov/pubmed) abstracts searching for emerging [MeSH](https://www.ncbi.nlm.nih.gov/mesh) terms that appear each upcoming year. Therefore, bioBERT embeddings are exploited computed on sentence-level so as to exploit better the semantic relationships that occur into the corresponding abstracts, avoiding misclassification that may occur in case of word-guided approaches. Later, we use the same mechanism for providing weak labels into gathered data from previous years, and then apply two different Weakly supervised learning (**WSL**) approaches for examining their performance on the same tasks ( *WSLbioSentMax(bioBERT)* and *WSLbioSentMax(tfidf)* ). Three different MeSH terms were investigated here, under single-label mode. Appropriate baselines and a state-of-the-art approach have been developed as they are described into the original paper. An expansion of our work towards multi-label classification tasks covering more complex cases and larger number of labels is the next step in the context of a larger project, called AMULET (more info on Additional resources section). 

*ZSL_and_WSL_implementations_setn2020.py* file contains a proper menu for selecting any of the implemented approaches, regarding the datasets inside **raw data** folder. In case of additional desired MeSH terms, you can add .txt files where each instance is in the next format: b'X labels: #y1#y2#...#yn' (X: abstract raw-text, y1,y2,...,yn the corresponding labels). The results are placed in folder with compatible names, while additional plots or code scripts for executing optimization of the existing hyper-parameters are placed accordingly. 

*Hint*: Modes 6 and 7 have been added so as to save the appropriate embeddings per MeSH term, for applying the optimization based on separate threshold values. Otherwise, each run would produce results only for one specific threshold value, leading to much spent comoutational resources.


If you find the code useful for your research, please cite our paper: (**citation still not announced**)

        @inproceedings{,
              title={Zero-Shot Classification of Biomedical Articles with EmergingMeSH Descriptors},
              booktitle={Proceedings of the 11th Hellenic Conference on Artificial Intelligence,
               {SETN} 2020, Athens, Greece, September 02-04, 2020)},
              author={N. Mylonas, S. Karlos and G. Tsoumakas},
              pages={},
              year={2020}
        }
        

## Requirements/Dependencies

Our code has been tested on Windows10 using python 3.7.6. The mentioned time responses correspond to a working station embedded with Intel Core i7-9700 (3GHz) processor and 32 GB RAM. The next libaries are necessary:

- Numpy
- bioBERT
- Spacy
- Pandas
- Seaborn and Matplotlib (for graphing)


## Developed by: 

|           Name  (English/Greek)            |      e-mail          |
| -------------------------------------------| ---------------------|
| Nikolaos Mylonas    (Νικόλαος Μυλωνάς)     | myloniko@csd.auth.gr |
| Stamatis Karlos     (Σταμάτης Κάρλος)      | stkarlos@csd.auth.gr |
| Grigorios Tsoumakas (Γρηγόριος Τσουμάκας)  | greg@csd.auth.gr     |

## Funded by

The research work was supported by the Hellenic Foundation forResearch and Innovation (H.F.R.I.) under the “First Call for H.F.R.I.Research Projects to support Faculty members and Researchers andthe procurement of high-cost research equipment grant” (ProjectNumber: 514).

## Additional resources

- [AMULET project](https://www.linkedin.com/showcase/amulet-project/about/)
- [Academic Team's page] (https://intelligence.csd.auth.gr/#)
