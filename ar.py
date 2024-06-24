import os
from ml import download_nifti
from niiSeg import niiSegCode
from niiToGLB import niiToGLB
import nibabel as nib
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import time

def handleAR(flairObj, filename):
    # load brain nii file
    brainFile = f'./tmp/{flairObj["public_id"]}'
    brainNII = nib.load(brainFile)
    brainData = brainNII.get_fdata()

    # create nii file for tumor segmentation
    tumorNIIPath = f'./tmp/{filename}_tumor.nii'
    niiSegCode(brainNII, brainData, tumorNIIPath)

    # convert nii brain file to glb file
    brainGLBPath = f"./results/{filename}_brain.glb"
    tumorGLBPath = f"./results/{filename}_tumor.glb"
    niiToGLB(brainData, brainGLBPath)

    # load nii tumor file
    tumorNII = nib.load(tumorNIIPath)
    tumorData = tumorNII.get_fdata()

    # convert nii tumor file to glb file
    niiToGLB(tumorData, tumorGLBPath)

    # upload both brain glb and tumor glb then return the responses
    brainGLB = cloudinary.uploader.upload(brainGLBPath)
    tumorGLB = cloudinary.uploader.upload(tumorGLBPath)

    paths = [tumorNIIPath, brainGLBPath, tumorGLBPath]

    # added the response of both brainGLB and tumorGLB to result object to return it
    result = {
        "brainGLB": {"secure_url": brainGLB["secure_url"], "public_id": brainGLB["public_id"]},
        "tumorGLB": {"secure_url": tumorGLB["secure_url"], "public_id": tumorGLB["public_id"]},
        "paths":paths
    }

    

    # return response
    return result