import os
import numpy as np

globalpath = '/home/peruzzo/PycharmProjects/PyFrac/03_Three_toughness_layers/Data_final/09mtoK'
date_ext = '2021-12-10__12_19_49'
basename = '/simulation__'+date_ext+'_file_'
basenameNew = '/simulation__'+date_ext+'_file1_'
final_number = 193
initial_new = 362
for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basename+ext_old,r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new)

for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new,r'/'+globalpath+'/simulation__'+date_ext+basename+ext_new)


