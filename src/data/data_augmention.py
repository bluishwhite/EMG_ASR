import os 


def data_augmen(emg_path,txt_path):
    emg_list = readfille(emg_path)
    txt_list = readfille(txt_path)
    npy_list = []
    senten_list = []
    for emg,txt in zip(emg_list,txt_list):
        ss_npy,no_ss_npy = get_npy(emg)
        npy_list.append(ss_npy)
        npy_list.append(no_ss_npy)
        senten_list.append(txt)
        senten_list.append(txt)

    file_info = os.path.split(emg_path)
    augem_name = file_info[1].split('.')[0]+"_npy"
    write_to_file(file_info[0]+"/",augem_name,npy_list,senten_list)


def move_channel_data_augmen(emg_path,txt_path,channel):
    emg_list = readfille(emg_path)
    txt_list = readfille(txt_path)
    npy_list = []
    senten_list = []
    move_str = ''
    for item in channel:
        move_str += str(item)  
    channel_str = move_str
    for emg,txt in zip(emg_list,txt_list):
        ss_npy,no_ss_npy = get_npy_move_channel(emg,channel_str)
        npy_list.append(ss_npy)
        npy_list.append(no_ss_npy)
        senten_list.append(txt)
        senten_list.append(txt)

    file_info = os.path.split(emg_path)
    augem_name = file_info[1].split('.')[0]+"_channel"+str(channel_str)+"_npy"
    write_to_file('debug/move_node'+"/",augem_name,npy_list,senten_list)




def readfille(path):
    texts = open(path,encoding='utf-8')
    line = texts.readline().strip()
    contents = []
    while line:
        contents.append(line)
        line = texts.readline().strip()
    return contents

def get_npy_move_channel(file_path,move_channel):
    
    file_info = os.path.split(file_path)
    file_name = file_info[1].split('.')[0]
    ss_npy = file_info[0]+"/"+file_name+"_256_25_ss"+"_mv"+str(move_channel)+".npy"
    noss_npy = file_info[0]+"/"+file_name+"_256_25_ce"+"_mv"+str(move_channel)+".npy"
    
    return ss_npy,noss_npy


def get_npy(file_path):
    file_info = os.path.split(file_path)
    file_name = file_info[1].split('.')[0]
    ss_npy = file_info[0]+"/"+file_name+'_256_25_ss.npy'
    noss_npy = file_info[0]+"/"+file_name+'_256_25_ce.npy'
    
    return ss_npy,noss_npy

def write_to_file(directory,dir_name,en_list,de_list):

    en_file = open(directory+dir_name+".en",'w', encoding='utf-8',newline='')
    de_file = open(directory+dir_name+".de",'w', encoding='utf-8',newline='')

    for emg,text in zip(en_list,de_list):
        en_file.write(emg+'\n')  
        de_file.write(text+'\n')

    en_file.close()
    de_file.close()
    # Todo

# data_augmen('debug/1238sentences/train.en','debug/1238sentences/train.de')
move_channel_data_augmen('debug/1238sentences/train.en','debug/1238sentences/train.de',channel=[6,7,8])