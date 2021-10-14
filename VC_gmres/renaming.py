import os
import numpy as np

globalpath = 'home/carlo/Desktop/PyFrac/VC_gmres/kk'
date_ext = '2021-08-18__19_12_04'
basename = '/simulation__'+date_ext+'_file_'
basenameNew = '/simulation__'+date_ext+'_file1_'
final_number = 251
initial_new = 49
for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basename+ext_old,r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new)

for i in range(final_number+1):
    ext_old = str(i)
    ext_new = str(i+initial_new)
    os.rename(r'/'+globalpath+'/simulation__'+date_ext+basenameNew+ext_new,r'/'+globalpath+'/simulation__'+date_ext+basename+ext_new)


