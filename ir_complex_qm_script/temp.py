from glob import glob
import json

files = glob('./iter.init/02.fp/conf.*/input.com')
#print(files)
with open('machine.json','r') as fp:
	mdata = json.load(fp)
print(mdata)
