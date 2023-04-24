from anstools import analysis
from embtools import refine
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from math import isclose

rf = refine()
ans = analysis()

def get_word_data(word, freq_dict):
    try:
        freq_row = freq_dict[freq_dict['word'] == word].reset_index()
        rank = freq_row.loc[0,'rank']
        freq = freq_row.loc[0,'rel_freq']
    except:
        rank = np.nan
        freq = np.nan 
    return rank, freq

lang_list = ['eng','ger']
dict_type = 'wo-numbers\\full_dict'

hrd_dir = 'D:\\DL4NLP\\all_dicts\\'
freq_path = 'D:\\DL4NLP\\freq-dicts\\'
emb1_path = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\OLD.txt'
emb2_path = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\NEW.txt'
century_dir = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + 'AllDecadesPrePost\\Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\' + lang_list[0] + '-' + lang_list[1] + '.txt'

oldL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'old' + lang_list[0].upper() + '.txt'
newL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'new' + lang_list[0].upper() + '.txt'
oldL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'old' + lang_list[1].upper() + '.txt'
newL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'new' + lang_list[1].upper() + '.txt'
all_path = century_dir

dict_dir = hrd_dir + 'seed_dict_' + lang_list[1] + '_' + lang_list[0] + '-muse.txt'

freq_l1 = freq_path + 'internet-' + lang_list[0] + '-wf.txt'
dict_l1 = pd.read_csv(freq_l1, sep = ' ', names = ['rank', 'rel_freq', 'word'])
dict_l1['word'] = dict_l1['word'].str.casefold()


freq_l2 = freq_path + 'internet-' + lang_list[1] + '-wf.txt'
dict_l2 = pd.read_csv(freq_l2, sep = ' ', names = ['rank', 'rel_freq', 'word'])
dict_l2['word'] = dict_l2['word'].str.casefold()


bldict = pd.read_csv(dict_dir)
bldict['old_sim'] = np.nan
bldict['new_sim'] = np.nan
bldict['shift'] = np.nan
bldict['change_type'] = np.nan
bldict['word_1_distance'] = np.nan
bldict['word_2_distance'] = np.nan
bldict['word_1_rank'] = np.nan
bldict['word_1_rel_freq'] = np.nan
bldict['word_2_rank'] = np.nan
bldict['word_2_rel_freq'] = np.nan
emb1 = KeyedVectors.load_word2vec_format(emb1_path, encoding ='utf-8')
emb2 = KeyedVectors.load_word2vec_format(emb2_path, encoding ='utf-8')

for i in range(len(bldict)):
    try:
        L1_word = bldict[lang_list[0]][i].casefold()
        L2_word = bldict[lang_list[1]][i].casefold()
        old_sim, new_sim, shift, W1_mono_distance, W2_mono_distance = ans.change_calc(emb1[L1_word], emb2[L1_word], emb1[L2_word], emb2[L2_word])
        if old_sim > 0.5 or new_sim > 0.5: 
                bldict.loc[[i],'old_sim'] = old_sim
                bldict.loc[[i],'new_sim'] = new_sim
                bldict.loc[[i],'shift'] = shift
                bldict.loc[[i],'word_1_distance'] = W1_mono_distance
                bldict.loc[[i],'word_2_distance'] = W2_mono_distance
                
                w1_rank, w1_freq = get_word_data(L1_word, dict_l1)
                w2_rank, w2_freq = get_word_data(L2_word, dict_l2)
                
                bldict.loc[[i],'word_1_rank'] = w1_rank
                bldict.loc[[i],'word_1_rel_freq'] = w1_freq
                bldict.loc[[i],'word_2_rank'] = w2_rank
                bldict.loc[[i],'word_2_rel_freq'] = w2_freq
                if isclose(W1_mono_distance, W2_mono_distance, abs_tol = 0.1):
                    result = 'Parallel Change'
                elif shift > 0.15:
                    result = 'Convergence'
                elif shift < -0.15:
                    result = 'Divergence'
                bldict.loc[[i],'change_type'] = result
    except:
        pass

bldict_nn = bldict.dropna(axis=0, how='any')    
bldict_nn.to_csv(hrd_dir + lang_list[1] + '_' + lang_list[0] + '-muse-list.txt' , index=False)