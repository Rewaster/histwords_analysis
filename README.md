# Scripts for analyzing HistWord embeddings
This repository contains files designed specifically to work with HistWords embeddings. It currently supports both decade analysis and century analysis (although century analysis produces a lot more noise right now).
If you plan to work with Chinese and want to train embeddings yourself, don't forget to change folder name with Chinese sgn embeddings from chi-sim-all-sgns to cmn-all-sgns, scripts will not work otherwise!
Here's a short algorithm to work using these scripts:
## Decade embeddings
### Preparation
If you want to align embeddings yourself, download appropriate embeddings from [HistWords website](https://nlp.stanford.edu/projects/histwords), appropriate bilingual dictionary from [MUSE](https://github.com/facebookresearch/MUSE) and frequency dictionaries from [Leeds](http://corpus.leeds.ac.uk/list.html), and then align using [VecMap](https://github.com/artetxem/vecmap). To correctly align everything using these scripts, make sure the scripts such are pointing at appropriate folders on your computer. The *vecmap_master* folder is not included here by default.
All bilingual dictionaries that I used are available in the *all_dicts* folder - very limited small seed dictionary of semantically stable word pairs (about 60 pairs for each language pair, seed_dict_xxx_xxx.txt), extended dictionary of semantically stable word pairs (about 300-400 word pairs for each language pair, seed_dict_xxx_xxx-filtered.txt) and MUSE bilingual dictionary in two forms - for century scripts (seed_dict_xxx_xxx-muse.txt) and decade scripts (xxx-xxx-muse.txt).

For faster alignment, full_cycle would do all the work for you with appropriate folders linked to it, but the embeddings used for student results were aligned by hand rather than using this script and it would require the folder structure produced either by hand or by *data_formatting.py* script, so it's not tested in terms of precision.

If you choose to use already aligned embeddings for English - German and English - French available [here](https://uni-bielefeld.sciebo.de/s/gtIjITSM0Fvjciu), decade analysis is very simple:
1. Open the decade_cosine_sim_identifier and adjust languages in lang_list variable and paths to all needed data accordingly
2. Run the script
3. The data will be available in .csv and .xslx formats

Note: if you would like to adjust the sensitivity of word pair selection, you can also change *similarity_marker* and *threshhold_change* variables. Both decade embeddings and this script were provided by https://github.com/Kathrin227 and slightly modified by me. It also uses HistWords **embedding** and **sequentialembedding** modules, which are already included here.
## Century embeddings
### Preparation
The alignment process is very similar to decade, but requires more work beforehand. First, the appropriate embeddings must be converted into .txt files and combined with averaged vectors. This is achieved using *data_formatting.py* script, just give it the paths to all the HistWords sgns and it will produce results with an appropriate folder structure which is then used by other scripts.
After combining, use *full_cycle.py* with *dec_analysis = False* and all the appropriate paths and languages changed to what you want to train.
### Some of the other flags available in the commands inside the code:
1. *words*. Whether to use a dictionary with bilingual data (True) or not (False)
2. *common_numbers*. Whether to use numbers present in both embeddings for alignment (True) or not (False)
3. *tagging*. Whether to add tags with the appropriate language, e.g. _oldENG or _newGER (True) or not (False) - will probably make decade alignment look worse if True, not tested. Will break century alignment if turned off.
4. --cuda is present inside of the *embtools.py* script, if your GPU is slow or does not support CUDA (or it's not set up), remove the flag so it works using a CPU for alignment. This will take a very long time, though (for me it was several hours with 5950x and 64GBs of RAM). You can also change --semi-supervised to --supervised there if you're using a MUSE bilingual dictionary or any other dictionary that is big enough, it will be much faster.
5. **word_1** and **word_2** can be used to immediately draw a plot after you're done with the alignment - if you forget to change the words, don't worry, it will just break after the alignment and all the aligned data will be fine.


After running the *full_cycle.py*, you should have three folders named *Aligned*, *Pickled* and *Split* in your century folder - default name embeddded in a lot of variables is *AllDecadesPrePost*, which is the folder produced by *data_formatting.py*.
The *Aligned* folder contains final embedding files after alignment - language 1 (L1) in one embedding space, language 2 (L2) in one embedding space and a combined embedding of both languages.
The *Pickled* folder contains 4 embedding spaces, separated by epoch - 19th century L1 and L2, 20th century L1 and L2.
The *Split* folder contains the same files as *Pickled*, but in .txt - this is not needed in theory, but still used in some scripts.
After this is done, the closest words in the new and old embedding space are collected using TensorFlow and CPU (also takes a while and consumes **A LOT** of RAM, around 27 GBs peak for me).
For this, the *corr_checker.py* script is used - it produces two new embedding spaces (L1-L2 combined, 19th century and L1-L2 combined, 20th century), saved in *Split* folder and two dictionaries of new closest words (*old-close.pkl*) and 20th century closest word (*new-close.pkl*), saved in the *Pickled* folder. Those two dictionaries are then used to find closest words using two thresholds - *threshold* for convergence/divergence, *abs_threshold = 0.1* for parallel change.
Using those values and data, three lists are producted in the *Pickled* folder - *conv2.txt* for convergence, *div2.txt* for divergence and *par2.txt* for parallel change.

From here on, there are several things that can be checked using the scripts available:
1. If you want to check how much do our arbitrary thresholds influence the amount of word pairs recorded in the lists, the *threshold_graph.py* does this. I used Excel for plotting, since it only needs to be done once, but in theory, it should be easy to add plotting right there.
2. If you want to filter convergence/divergence/parallel change lists using Leeds frequency dictionary, *new_freq_filter.py* is the way to go. 
3. To check how accurate the similarity is compared to FastText corpora, a *fasttext-checker.py* can be run. It uses FastText embeddings, which should be downloaded separately from [here](https://fasttext.cc/docs/en/aligned-vectors.html) and *fasttext.py* from [fastText_multilingual](https://github.com/babylonhealth/fastText_multilingual), which is already included.
4. To test correlations between our semantic change gold standard or data for cosine similarities for human-rated data in *gold_standard* folder, uncomment *#corr = ans.corr_check(..* in *corr_checker.py* and select appropriate flags: *scored = True* for human-rated data and *scored = False* for gold standard category comparison. Both gold standards and appropriate human-rated data lists for English - French and English - German are available in *gold_standard* folder.

If you have any questions, do not hesitate to reach out to me at bergdanfinmir@gmail.com, I will try to help as much as I can!
