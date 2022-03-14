import os
import numpy as np

#globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/10space_inv'
#globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/10space_inv_coarse'
globalpath = '/home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/11mtoK'
date_ext = '2022-02-01__16_05_38'
basename = '/simulation__'+date_ext+'_file_'
basenameNew = '/simulation__'+date_ext+'_file1_'
final_number = 805
initial_new = 320
for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basename+ext_old,r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new)

for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new,r'/'+globalpath+'/simulation__'+date_ext+basename+ext_new)


