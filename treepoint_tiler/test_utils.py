from utils import *
import numpy as np

rand_arr = np.random.randint(0, 255, size = 16384).reshape((128,128))
stk_arr = [rand_arr, rand_arr, rand_arr]
fake_img = np.dstack(stk_arr)
x,y = fake_img.shape[:-1]
fake_ann = np.zeros((128,128))
fake_ann1 = fake_ann[63:64] == 1

def test_get_crop_dims():
    assert 64 == get_crop_dims(64,32)
    
def test_crop_center():
    assert np.array_equiv(fake_img, crop_center(fake_img,x,y))
    
def test_crop_trainset():
    assert np.array_equiv(fake_img[51:128,0:128], crop_trainset(fake_img,0.2))
    
def test_crop_testset():
    assert np.array_equiv(fake_img[0:25,0:128], crop_testset(fake_img,0.2))

def test_crop_valset():
    assert np.array_equiv(fake_img[25:51,0:128], crop_valset(fake_img,0.2))
    
def test_prep_data():
    assert None != prep_data(fake_img,0.2,0,0)
    
## to be continued...
    
