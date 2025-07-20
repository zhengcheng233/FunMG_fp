import os
import json

def read_kr(log):
    with open(log, "r") as fa:
        lines = fa.readlines()
    for iline in lines:
        iline = iline.split(" ")
        if "radiative" in iline:
            iline = [im for im in iline if im != ""]
            return iline[4]
def read_knr(log):
    with open(log, "r") as fa:
        lines = fa.readlines()
    for iline in lines:
        iline = iline.split(" ")
        if ("Intersystem" in iline) and ("Reverse" not in iline):
            iline = [im for im in iline if im != ""]
            return iline[9]
data = {}
data['0.0'] = [float(read_kr("./kr/spec.tvcf.log")),float(read_knr("./knr/isc.tvcf.log"))]

with open("plqy.json", "w") as fa:
    json.dump(data, fa)
