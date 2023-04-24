from anstools import analysis
from embtools import refine
import os

rf = refine()
ans = analysis()

lang_list = ['eng','ger']
dict_type = 'w-numbers\\extended_dict'

hrd_dir = 'D:\\DL4NLP\\gold_standard\\' + lang_list[0].upper() + '-' + lang_list[1].upper() + '.csv'
emb1_path = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\OLD.txt'
emb2_path = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Split\\' + lang_list[0] + '-' + lang_list[1] + '\\NEW.txt'
century_dir = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\' + lang_list[0] + '-' + lang_list[1] + '.txt'
pickle_dir = 'D:\\DL4NLP\\full_cycle_test\\' + dict_type + '\\AllDecadesPrePost\\Pickled\\' + lang_list[0] + '-' + lang_list[1] + '\\' 
freq_path = 'D:\\DL4NLP\\freq-dicts\\'

if not os.path.isfile(emb1_path) or not os.path.isfile(emb1_path):
    oldL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'old' + lang_list[0].upper() + '.txt'
    newL1_path = emb1_path[:emb1_path.rfind('\\')+1] + 'new' + lang_list[0].upper() + '.txt'
    oldL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'old' + lang_list[1].upper() + '.txt'
    newL2_path = emb2_path[:emb2_path.rfind('\\')+1] + 'new' + lang_list[1].upper() + '.txt'
    
    oldL1 = rf.load_txt(oldL1_path)
    newL1 = rf.load_txt(newL1_path)
    oldL2 = rf.load_txt(oldL2_path)
    newL2 = rf.load_txt(newL2_path)
    
    rf.combine_txt(oldL1, oldL2, emb1_path)
    rf.combine_txt(newL1, newL2, emb2_path)

#corr = ans.corr_check(emb1_path, emb2_path, hrd_dir, threshold = 0, scored = False)
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
    ans.dump_pickle(pickle_old, old_close)
    ans.dump_pickle(pickle_new, new_close) 
else:
    old_close = ans.load_pickle(pickle_old)
    new_close = ans.load_pickle(pickle_new) 
if not os.path.isfile(pickle_dir + 'par2.txt') or not os.path.isfile(pickle_dir + 'conv2.txt') or not os.path.isfile(pickle_dir + 'div2.txt'):
    par = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Parallel Change', lang_list, freq_path, threshold = 0)
    con = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Convergence', lang_list, freq_path, threshold = 0)
    div = ans.find_change_type(oldL1_path, newL1_path, oldL2_path, newL2_path, old_close, new_close, 'Divergence', lang_list, freq_path, threshold = 0) 
    
    ans.save_to_file(par, pickle_dir + 'par2.txt', lang_list, 'Parallel Change', max_lines=10000, par = True)
    ans.save_to_file(con, pickle_dir + 'conv2.txt', lang_list, 'Convergence', max_lines=10000, par = False)
    ans.save_to_file(div, pickle_dir + 'div2.txt', lang_list, 'Divergence', max_lines=10000, par = False)