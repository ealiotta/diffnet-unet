import numpy as np
from keras.models import load_model
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import argparse
from skimage.transform import resize
import nibabel as nib

parser = argparse.ArgumentParser(description='Supply neural net location (folder containing .h5 files) and DTI image data (folder containing dicom) to fit tensors.')
parser.add_argument('NeuralNetLoc', help='location of the keras neural network files')
parser.add_argument('ImageLoc', help='DTI image data location (.nii.gz format)')
parser.add_argument('OutLoc', help='Location to save the fitted tensors')

args = parser.parse_args()

NeuralNetLoc = args.NeuralNetLoc
ImageLoc = args.ImageLoc
OutLoc = args.OutLoc

niidata = nib.load(ImageLoc)
IM = niidata.get_fdata()

directions = IM.shape[3]-1

print('Found %d x %d x %d image with %d directions (+b0 image)' % (IM.shape[0], IM.shape[1], IM.shape[2], directions))

# load pretrained neural network
NeuralNetFile = NeuralNetLoc + '/UNET_FA_' + str(directions) + 'dir.h5'
print('Running model: %s' % (NeuralNetFile))
model_fa = load_model(NeuralNetFile)
NeuralNetFile = NeuralNetLoc + '/UNET_MD_' + str(directions) + 'dir.h5'
print('Running model: %s' % (NeuralNetFile))
model_md = load_model(NeuralNetFile)

yres = IM.shape[0]
xres = IM.shape[1]
zres = IM.shape[2]
ndir = IM.shape[3]

# normalize and resize (if needed) image
signals = np.zeros((zres, yres, xres, ndir))

for k in range(0, zres):
    for q in range(0, ndir):
        tmpIM = np.squeeze(IM[:, :, k, q])
        b0IM = np.squeeze(IM[:, :, k, 0])

        signals[k, :, :, q] = np.divide(tmpIM, b0IM.max())

if signals.shape[1] != 128 or signals.shape[2] != 128:
    print('Resizing to 128x128...')
    signals = resize(signals, [zres, 128, 128, ndir])

# the x and y directions do matter (brain should be oriented in this fashion)
signals = np.transpose(signals,[0,2,1,3])

# reconstruct with diffNet
FA = model_fa.predict(signals)
MD = model_md.predict(signals)/1e3

if not os.path.exists(OutLoc):
    os.makedirs(OutLoc)

FA = np.transpose(FA,(2, 1, 0, 3))
MD = np.transpose(MD,(2, 1, 0, 3))

# save data as .mat
FILE = OutLoc + '/dFA.mat'
sio.savemat(FILE, mdict={'FA_unet': FA})
print('FA map saved to', FILE, '\n')
FILE = OutLoc + '/dMD.mat'
sio.savemat(FILE, mdict={'MD_unet': MD})
print('MD map saved to', FILE, '\n')

# also save as numpy array
FILE = OutLoc + '/dFA.npy'
np.save(FILE, FA)
print('Saved to', FILE)
FILE = OutLoc + '/dMD.npy'
np.save(FILE, MD)
print('Saved to', FILE)

plt.subplot(121)
plt.imshow(np.squeeze(FA[:, :, np.int(zres/2)]), vmin=0, vmax=1, cmap='gray')
plt.title('Unet FA')

plt.subplot(122)
plt.imshow(np.squeeze(MD[:,:,np.int(zres/2)]),vmin=0,vmax=3e-3,cmap='gray')
plt.title('Unet MD')
plt.show()
