import sys
from os import listdir, makedirs
from os.path import isfile, join
import re
import numpy as np
from util.util import save3Dimage_numpy
from util.util import save3D_3slices_numpy
import nibabel as nib

def sortByNum(txt):
    num = re.findall(r'\d+' ,txt) 
    return int(num[0])

if __name__ == '__main__':
    dataroot = sys.argv[1]
    output = sys.argv[2]
    for dir in ["train", "test"]:
        for folder in ["{}ct".format(dir), "{}ct_labels".format(dir), "{}mr".format(dir), "{}mr_labels".format(dir)]:
            makedirs(join('output' ,folder), exist_ok = True)
            path = join(dataroot, folder)
            onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
            if len(onlyfiles) > 16:
                onlyfiles = sorted(onlyfiles, key=sortByNum)[:16]
            for file in onlyfiles:
                full_path = join(dataroot, folder, file)
                if ".nii" in file:
                    data = nib.load(full_path)
                    print("File: ", full_path, "Orientation: ", nib.aff2axcodes(data.affine), "Shape: ", data.shape)
                else:
                    data = np.load(full_path)['arr_0']
                    output_path = join(output, folder, file.replace("npz", "png").replace("nii.gz", "png"))
                    save3D_3slices_numpy(data.squeeze(), output_path)
                    print("{} Printed".format(output_path))