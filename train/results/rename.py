import glob
import os
import shutil





img_dir = "./balloon1"
imgs = glob.glob(img_dir + "/*.png")
os.makedirs(os.path.join(img_dir,"Balloon1"), exist_ok=True)
for img in imgs:
    file_name = os.path.basename(img)
    file_name, file_ext = os.path.splitext(file_name)
    t = int(file_name[:3])
    shutil.copyfile(img, os.path.join(img_dir,"Balloon1", "v000_t%03d.png" % t))
