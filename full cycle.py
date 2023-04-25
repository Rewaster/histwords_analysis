import os
from embtools import refine
from anstools import analysis
from plottools import graphing
import shutil

rf = refine()
ans = analysis()
grph = graphing()


decade_dir = 'D:\\DL4NLP\\full_cycle_test\\IndividualDecades\\'
century_dir = 'D:\\DL4NLP\\full_cycle_test\\AllDecadesPrePost\\'
dicts_dir = 'D:\\DL4NLP\\full_cycle_test\\Dictionaries\\'
vecmap_dir = 'D:\\DL4NLP\\full_cycle_test\\vecmap-master\\'
plot_dir = 'D:\\DL4NLP\\full_cycle_test\\plots\\'
hrd_dir = 'D:\\DL4NLP\\all_dicts\\'


lang_list = ['eng','ger'] #two languages of interest in the analysis - currently out of 'eng','ger','fre','cmn' ('cmn' will not work with centuries)
l1_files = []
l2_files = []
word_1 = 'bourgeoisie'
word_2 = 'bourgeoisie'
#word_copy = 'proletariat'
WORDS = [word_1, word_2] #can string together more words for decade plotting, century plotting can only show two at the same time
year_1 = 1800
year_2 = 2000
split = 10
dia = range(year_1, year_2, split)
dec_analysis = True

comb_dict = hrd_dir + 'seed_dict_' + lang_list[1] + '_' + lang_list[0] + '-filtered.txt'

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if dec_analysis:
    full_path_l1 = decade_dir + lang_list[0] + '-all_sgns\\'
    full_path_l2 = decade_dir + lang_list[1] + '-all_sgns\\'
    temp_path_l1 = full_path_l1 + 'temp\\' 
    check_dir(temp_path_l1)
    temp_path_l2 = full_path_l2 + 'temp\\' 
    check_dir(temp_path_l2)
    
    aligned_dir = decade_dir + 'Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\'
    check_dir(aligned_dir)
    
    temp_dir = decade_dir + 'Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\temp\\'
    check_dir(temp_dir)
    
else:
    full_path_l1 = century_dir + lang_list[0] + '-all_sgns\\'
    full_path_l2 = century_dir + lang_list[1] + '-all_sgns\\'
    temp_path_l1 = full_path_l1 + 'temp\\' 
    check_dir(temp_path_l1)
    temp_path_l2 = full_path_l2 + 'temp\\' 
    check_dir(temp_path_l2)
    
    aligned_dir = century_dir + 'Aligned\\' + lang_list[0] + '-' + lang_list[1] + '\\'
    check_dir(aligned_dir)
    
    split_dir = century_dir + 'Split\\' + lang_list[0] + '-' + lang_list[1] + '\\'
    check_dir(split_dir)
    
    pickle_dir = century_dir + 'Pickled\\' + lang_list[0] + '-' + lang_list[1] + '\\'
    check_dir(pickle_dir)
    
for dirpath, dirnames, filenames in os.walk(full_path_l1):
    for file in filenames:
            l1_files.append(file)

for dirpath, dirnames, filenames in os.walk(full_path_l2):
    for file in filenames:
            l2_files.append(file)
   
#decade
if dec_analysis:
    zipped_files = zip(l1_files, l2_files)
    for i in zipped_files:
        current_year = i[0][:-5]
        if int(current_year) not in dia:
            continue
        dict_path = dicts_dir + lang_list[0] + '-' + lang_list[1] + '_' + current_year + '_dict.txt'
        aligned_dir_l1 = aligned_dir + i[0][:-4] + '-' + lang_list[0] + '.txt'
        aligned_dir_l2 = aligned_dir + i[1][:-4] + '-' + lang_list[1] + '.txt'
        if not os.path.isfile(dict_path):
            rf.decade_dicts(full_path_l1 + i[0], full_path_l2 + i[1], dicts_dir, lang_list, current_year, hrd_path = comb_dict, words = True, common_numbers = True)
        if not os.path.isfile(aligned_dir_l1) or not os.path.isfile(aligned_dir_l2):
            if not os.path.isfile(aligned_dir + current_year + '-w.npy') or not os.path.isfile(aligned_dir + current_year + '-vocab.pkl'):
                rf.full_prep(temp_dir, dict_path, full_path_l1 + i[0], full_path_l2 + i[1], 
                             temp_path_l1 + i[0], temp_path_l2 + i[1], vecmap_dir,
                             aligned_dir,  lang_list,
                             decades = dec_analysis, tagging = False)
    #cleanup
    l1_temp_files = os.listdir(temp_path_l1)
    l2_temp_files = os.listdir(temp_path_l2)
    aligned_temp_files = os.listdir(aligned_dir + 'temp\\')
    #dict_files = os.listdir(dicts_dir)
    if l1_temp_files or l2_temp_files or aligned_temp_files:
        for f in l1_temp_files:
            os.remove(temp_path_l1 + f)
        for f in l2_temp_files:
            os.remove(temp_path_l2 + f)
        for f in aligned_temp_files:
            os.remove(aligned_dir + 'temp\\' + f)
    check_dir(plot_dir)
    grph.decade_plotting(WORDS, aligned_dir, plot_dir, year_1, year_2, split)
else:
    #TODO: simplify century structure, current one is clunky
    oldL1_path = full_path_l1 + l1_files[1]
    newL1_path = full_path_l1 + l1_files[0]
    oldL2_path = full_path_l2 + l2_files[1]
    newL2_path = full_path_l2 + l2_files[0]
    
    oldL1_path_aligned = temp_path_l1 + l1_files[1][:-4] + '-aligned.txt'
    newL1_path_aligned = temp_path_l1 + l1_files[0][:-4] + '-aligned.txt'
    oldL2_path_aligned = temp_path_l2 + l2_files[1][:-4] + '-aligned.txt'
    newL2_path_aligned = temp_path_l2 + l2_files[0][:-4] + '-aligned.txt'
    
    L1_combined = temp_path_l1 + lang_list[0] + '-combined.txt'
    L2_combined = temp_path_l2 + lang_list[1] + '-combined.txt'
    L1_aligned = temp_path_l1 + lang_list[0] + '-aligned.txt'
    L2_aligned = temp_path_l2 + lang_list[1] + '-aligned.txt'
    L1_L2_combined = aligned_dir + lang_list[0] + '-' + lang_list[1] + '.txt'
    
    file_path_l1_dict = dicts_dir + lang_list[0] + '_dict.txt'
    file_path_l2_dict = dicts_dir + lang_list[1] + '_dict.txt'
    file_path_l2_l1_dict = dicts_dir + lang_list[0] + '-' + lang_list[1] + '_tagged_dict.txt'
    
    
    if not os.path.isfile(file_path_l1_dict) or not os.path.isfile(file_path_l2_dict) or not os.path.isfile(file_path_l2_l1_dict):
        file_path_l1_dict, file_path_l2_dict, file_path_l2_l1_dict = rf.century_dicts(oldL1_path, newL1_path, oldL2_path, newL2_path, dicts_dir, lang_list, hrd_path = comb_dict, words = True, common_numbers = False)
        
    align_order = [[temp_path_l1, file_path_l1_dict, oldL1_path, newL1_path, oldL1_path_aligned, newL1_path_aligned, L1_combined, lang_list[0]],
                   [temp_path_l2, file_path_l2_dict, oldL2_path, newL2_path, oldL2_path_aligned, newL2_path_aligned, L2_combined, lang_list[1]],
                   [aligned_dir, file_path_l2_l1_dict, L1_combined, L2_combined, L1_aligned, L2_aligned, L1_L2_combined]]
    for i in align_order:
        if not os.path.isfile(i[4]) or not os.path.isfile(i[5]) or not os.path.isfile(i[6]):
            if len(i)==8:
                tags = True
                lng = i[7]
            else:
                tags = False
                lng = ''
            rf.full_prep(i[0], i[1], i[3], i[2], i[5], i[4], vecmap_dir, i[6], lng, decades = dec_analysis, tagging = tags)
    if not os.listdir(split_dir) or not os.listdir(pickle_dir):
        oldL1_embedding, newL1_embedding, oldL2_embedding, newL2_embedding = ans.sort_embeddings(L1_L2_combined, lang_list)
        rf.save_txt(oldL1_embedding, split_dir + 'old' + lang_list[0].upper() + '.txt')
        rf.save_txt(newL1_embedding, split_dir + 'new' + lang_list[0].upper() + '.txt')
        rf.save_txt(oldL2_embedding, split_dir + 'old' + lang_list[1].upper() + '.txt')
        rf.save_txt(newL2_embedding, split_dir + 'new' + lang_list[1].upper() + '.txt')
        ans.dump_pickle(pickle_dir + 'old' + lang_list[0].upper() + '.pkl', oldL1_embedding)
        ans.dump_pickle(pickle_dir + 'new' + lang_list[0].upper() + '.pkl', newL1_embedding)
        ans.dump_pickle(pickle_dir + 'old' + lang_list[1].upper() + '.pkl', oldL2_embedding)
        ans.dump_pickle(pickle_dir + 'new' + lang_list[1].upper() + '.pkl', newL2_embedding)
    else:
        oldL1_embedding = ans.load_pickle(pickle_dir + 'old' + lang_list[0].upper() + '.pkl')
        newL1_embedding = ans.load_pickle(pickle_dir + 'new' + lang_list[0].upper() + '.pkl')
        oldL2_embedding = ans.load_pickle(pickle_dir + 'old' + lang_list[0].upper() + '.pkl')
        newL2_embedding = ans.load_pickle(pickle_dir + 'new' + lang_list[0].upper() + '.pkl')
        '''
        Навести порядок в селф-клс-ничего:селф когда ток в классе,
        клс чтобы вызывать из класса но он имеет доступ к другим вещам в классе, а они к нему,
        а ничего когда сам по себе 
        '''
    check_dir(plot_dir)
    orig_files = [oldL1_path, newL1_path, oldL2_path, newL2_path]
    for i in os.listdir(split_dir):
        chk_dir = aligned_dir + i
        if not os.path.isfile(chk_dir):
            shutil.copy2(split_dir + i, aligned_dir)
    #cleanup
    l1_temp_files = os.listdir(temp_path_l1)
    l2_temp_files = os.listdir(temp_path_l2)
    #dict_files = os.listdir(dicts_dir)
    if l1_temp_files or l2_temp_files:
        for f in l1_temp_files:
            os.remove(temp_path_l1 + f)
        for f in l2_temp_files:
            os.remove(temp_path_l2 + f)
    grph.century_plotting(aligned_dir, lang_list, word_1, word_2, plot_dir)
    