import nibabel as nib

def niiSegCode(brainNII, brainData, filePath):
    optimal_threshold = 350
    tumor_mask = (brainData > optimal_threshold)

    tumor_img = nib.Nifti1Image(tumor_mask.astype(float), affine=brainNII.affine)
    nib.save(tumor_img, filePath)
    print('Tumor image saved')