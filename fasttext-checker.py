from anstools import analysis
from embtools import refine
import os
from gensim.models import KeyedVectors
from fasttext import FastVector
import pandas as pd
import numpy as np

rf = refine()
ans = analysis()

lang_list = ['eng','fre']

hrd_path = 'D:\\DL4NLP\\gold_standard\\' + lang_list[0].upper() + '-' + lang_list[1].upper() + '_scored.csv'
embl1 = 'D:\\DL4NLP\\ft\\' + lang_list[0] + '\\wiki.' + lang_list[0] + '.align.vec'
embl2 = 'D:\\DL4NLP\\ft\\' + lang_list[1] + '\\wiki.' + lang_list[1] + '.align.vec'
emb2c = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\NEW.txt'
century_dir = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\' + lang_list[0] + '-' + lang_list[1] + '.txt'
pickle_dir = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Pickled\\' + lang_list[0] + '-' + lang_list[1] + '\\' 

out_path = hrd_path[:-4] + '_results.csv'
embl1_loaded = FastVector(vector_file=embl1)
embl2_loaded = FastVector(vector_file=embl2)
emb2c_loaded = KeyedVectors.load_word2vec_format(emb2c, encoding ='utf-8')

if not os.path.isfile(out_path):
        df = pd.read_csv(hrd_path)
        df['Similarity_ft'] = np.nan
        df['Similarity_our'] = np.nan
        for i in range(len(df)):
            try:
                word_1 = df.loc[i][1].casefold()
                word_2 = df.loc[i][2].casefold()
                df.loc[[i],'Similarity_ft'] = FastVector.cosine_similarity(embl1_loaded[word_1], embl2_loaded[word_2])
                df.loc[[i],'Similarity_our'] = emb2c_loaded.similarity(word_1, word_2)
            except:
                pass
        df_nn = df.dropna(axis=0, how='any')
else:
        df_nn = pd.read_csv(out_path)
        
score_ft = df_nn['score'].corr(df_nn['Similarity_ft'], method='spearman')
print('Correlation between human rated similarity and FastText embeddings similarity is ', score_ft)
score_our = df_nn['score'].corr(df_nn['Similarity_our'], method='spearman')
print('Correlation between human rated similarity and our embeddings similarity is ', score_our)

df_nn.to_csv(out_path, index=False)
