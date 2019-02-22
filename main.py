from model import *
from data import *
import matplotlib as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/brain/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_brain.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=7,callbacks=[model_checkpoint])

testGene = testGenerator("data/brain/test")
results = model.predict_generator(testGene,6,verbose=1) #steps per epoch is 6. This value should be less the no.of test images
saveResult("data/brain/test",results)