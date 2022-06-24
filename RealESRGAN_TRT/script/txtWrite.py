import os

def txt(path, name, msg):
    full_path = path + '/' + name + '.txt'
    file = open(full_path, 'a+')
    file.writelinde([str(msg), '\n'])
