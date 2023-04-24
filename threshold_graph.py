# -*- coding: utf-8 -*-
from anstools import analysis
from embtools import refine
import os
import numpy as np

rf = refine()
ans = analysis()

lang_list = ['eng','fre']
hrd_dir = 'D:\\DL4NLP\\gold_standard\\' + lang_list[0].upper() + '-' + lang_list[1].upper() + '.csv'
emb1_path = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\OLD.txt'
emb2_path = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\NEW.txt'
century_dir = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\' + lang_list[0] + '-' + lang_list[1] + '.txt'
pickle_dir = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\Pickled\\' + lang_list[0] + '-' + lang_list[1] + '\\' 

oldL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'old' + lang_list[0].upper() + '.txt'
newL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'new' + lang_list[0].upper() + '.txt'
oldL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'old' + lang_list[1].upper() + '.txt'
newL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'new' + lang_list[1].upper() + '.txt'
all_path = century_dir
pickle_old = pickle_dir + 'old_close.pkl'
pickle_new = pickle_dir + 'new_close.pkl'

if not os.path.isfile(pickle_old) or not os.path.isfile(pickle_new):
    old_close = ans.find_closest_words(oldL1_path, oldL2_path, neighbours_number = 10, load_txt = True, converted = False)
    new_close = ans.find_closest_words(newL1_path, newL2_path, neighbours_number = 10, load_txt = True, converted = False)
    ans.dump_pickle(pickle_dir + 'old_close', old_close)
    ans.dump_pickle(pickle_dir + 'new_close', new_close) 
else:
    old_close = ans.load_pickle(pickle_dir + 'old_close')
    new_close = ans.load_pickle(pickle_dir + 'new_close') 


for i in np.arange(0.9,1,0.05):
    con = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Convergence', threshold = i) 
    with open('conv.txt', 'a', encoding="utf-8") as o:
        o.write(str(i) + ' ' + str(len(con)) + '\n')
 
for k in np.arange(0,-1,-0.05): 
    div = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Divergence', threshold = k)
    with open('div.txt', 'a', encoding="utf-8") as o:
        o.write(str(k) + ' ' + str(len(div)) + '\n')
       
for j in reversed(np.arange(0.05,0.5,0.05)):
    par = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Parallel Change', threshold = 0, abs_threshold = j)
    with open('par.txt', 'a', encoding="utf-8") as o:
        o.write(str(j) + ' ' + str(len(par)) + '\n')