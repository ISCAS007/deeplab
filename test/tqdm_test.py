# -*- coding: utf-8 -*-

from tqdm import tqdm, trange
from time import sleep
import sys
from random import random,randint

#bar = trange(10)
#for i in bar:
#    # Print using tqdm class method .write()
#    sleep(0.1)
#    if not (i % 3):
#        tqdm.write("Done task %i" % i)
#    # Can also use bar.write()
    
for epoch in trange(5,desc='epoches',leave=False):
#    with trange(5,desc='step',leave=False) as t:
    t=trange(5,desc='step',leave=False) 
    for idx,step in enumerate(t):
        t.set_postfix(acc=(step/5),miou=(epoch/5),idx=idx)
#            tqdm.write('acc= %0.2f'%(step/5))
#            tqdm.write('miou= %0.2f'%(epoch/5))
#            tqdm.write('idx=%d'%idx)
        sleep(0.1)
        
for i in range(21):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    sleep(0.25)
    
with trange(100) as t:
    for i in t:
        # Description will be displayed on the left
        t.set_description('GEN %i' % i)
        # Postfix will be displayed on the right,
        # formatted automatically based on argument's datatype
        t.set_postfix(loss=random(), gen=randint(1,999), str='h',
                      lst=[1, 2])
        sleep(0.1)

with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
          postfix=["Batch", dict(value=0)]) as t:
    for i in range(10):
        sleep(0.1)
        t.postfix[1]["value"] = i / 2
        t.update()