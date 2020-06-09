# diffnet-unet

applyDiffNet_unet.py 

simple implementation of pretained convolutional neural networks which reconstruct fractional anisotrpy
(FA) and mean diffusivity (MD) maps from diffusion tensor imaging (DTI) scans.

This code reads in a dti image set in .nii.gz format. These dicom files should contain
DWI with b=1000s/mm2 and 3, 6, or 20 diffusion encoding directions plus one b=0 reference image. 
This code assumes that data contains b=0 images first followed by all DWI.

Data should be formatted as: [Nx,Ny,Nslice,Ndir]

** these networks are trained on data with b=1000s/mm2. This will not work properly for other b-values **

Code outputs dFA and dMD maps (i.e. FA and MD estimates) in matlab (.mat) and numpy (.npy) formats.

Requirements:	python with the following libraries
                numpy
                scipy
                nibabel
                matplotlib
                keras (https://keras.io/)

  Usage: python <pathToCode>\applyDiffNet_unet.py <NeuralNetLoc> <ImageLoc> <OutLoc>

	where: 	NeuralNetLoc: 	path to folder containing .h5 keras neural network files (provided)
		ImageLoc: 	path to image file (.nii.gz)
		OutLoc:		path for desired output

Example dicom files are provided with 3, 6, and 20 diffusion encoding directions. After successful
completion, you should see one slice of the reconstructed dFA and dMD maps in a dialog box.

Eric Aliotta, PhD
University of Virginia, Department of Radiation Oncology
06.09.2020

eric.aliotta@virginia.edu
