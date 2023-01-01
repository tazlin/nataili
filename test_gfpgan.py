import PIL
import time

from nataili import gfpgan, ModelManager, logger

image = PIL.Image.open("01.png").convert("RGB")

mm = ModelManager()

mm.gfpgan.load("GFPGAN")

facefixer = gfpgan(mm.gfpgan.loaded_models["GFPGAN"])

tick = time.time()
results = facefixer(input_image=image, strength=1.0)
logger.init_ok(f"Job Completed. Took {time.time() - tick} seconds", status="Success")