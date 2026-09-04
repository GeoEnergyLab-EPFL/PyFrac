import os
for i in range(1,149,1):
    prefix='simulation__2020-12-14__20_33_47_file_'
    root='/home/carlo/Desktop/PyFrac/toughness_jump/Data/toughness_jump_0027/simulation__2020-12-14__20_33_47_sim1/'
    ending=str(i)
    oldname=root+prefix+ending
    newending=str(i+219)
    newname=root+prefix+newending
    os.rename(r''+oldname,r''+newname)