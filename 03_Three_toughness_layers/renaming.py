import os
import numpy as np

globalpath = 'home/carlo/Desktop/PyFrac/03_Three_toughness_layers/Data_final/08mtoK'
date_ext = '2021-12-08__18_53_44'
basename = '/simulation__'+date_ext+'_file_'
basenameNew = '/simulation__'+date_ext+'_file1_'
final_number = 170 # final number of the new set of results
initial_new = 234   # last number of the old set of results
for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basename+ext_old,r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new)

for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new,r'/'+globalpath+'/simulation__'+date_ext+basename+ext_new)


