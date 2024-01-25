from cleanfid import fid


real_images_dir = 'real_images'
generated_images_dir = 'generated_images'

# compute FID
score = fid.compute_fid(real_images_dir, generated_images_dir, mode="clean")
print(f"FID score: {score}")