# -*- coding: utf-8 -*-

from tqdm import tqdm, trange
from time import sleep

#bar = trange(10)
#for i in bar:
#    # Print using tqdm class method .write()
#    sleep(0.1)
#    if not (i % 3):
#        tqdm.write("Done task %i" % i)
#    # Can also use bar.write()
    
for epoch in trange(10,desc='epoches'):
    for step in trange(30,desc='step'):
        tqdm.write('acc= %0.2f'%(step/30))
        tqdm.write('miou= %0.2f'%(epoch/10))
        sleep(0.1)