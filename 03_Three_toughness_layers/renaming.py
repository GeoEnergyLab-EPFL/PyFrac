import os
import numpy as np

#globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv'
#globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/10space_inv_coarse'
#globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/11mtoK'
globalpath= '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/6mtoK_break_kratio_3p5/'
globalpath= '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/07mtoK_break_k2k1_6p5/'
globalpath= '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/6mtoK_break/'
date_ext = '2022-09-19__18_01_05'
date_ext = '2022-09-20__18_23_05'
date_ext = '2022-09-09__19_36_02'
basename = '/simulation__'+date_ext+'_file_'
basenameNew = '/simulation__'+date_ext+'_file1_'
final_number = 162
initial_new = 463
for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basename+ext_old,r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new)

for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new,r'/'+globalpath+'/simulation__'+date_ext+basename+ext_new)


