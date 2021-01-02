import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import argparse


TRAIN_EPOCHS = 1000

im_sz = 512
mp_sz = 128

warp_scale = 0.2
mult_scale = 0.4
add_scale = 0.4


@tf.function 
def warp(origins, targets, preds_org, preds_trg):
    res_targets = tfa.image.dense_image_warp(origins * (1 + preds_org[:,:,:,0:3] * mult_scale) + preds_org[:,:,:,3:6] * 2 * add_scale, preds_org[:,:,:,6:8] * im_sz * warp_scale )
    res_origins = tfa.image.dense_image_warp(targets * (1 + preds_trg[:,:,:,0:3] * mult_scale) + preds_trg[:,:,:,3:6] * 2 * add_scale, preds_trg[:,:,:,6:8] * im_sz * warp_scale )
    return res_targets, res_origins

def create_grid(scale):
    grid = np.mgrid[0:scale,0:scale] / (scale - 1) * 2 -1
    grid = np.swapaxes(grid, 0, 2)
    grid = np.expand_dims(grid, axis=0)
    return grid


def produce_warp_maps(origins, targets):
    class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
            self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
            self.conv3 = tf.keras.layers.Conv2D((3 + 3 + 2) * 2, (5, 5))

        def call(self, maps):
            x = tf.image.resize(maps, [mp_sz, mp_sz])
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x
        

    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(maps, origins, targets):
      with tf.GradientTape() as tape:
        preds = model(maps)
        preds = tf.image.resize(preds, [im_sz, im_sz])
        
        res_targets, res_origins = warp(origins, targets, preds[...,:8], preds[...,8:])
        res_targets_half, res_origins_half = warp(origins, targets, preds[...,:8] * 0.5, preds[...,8:] * 0.5)

        loss =  (loss_object(origins, res_origins) + loss_object(targets, res_targets)) + loss_object(res_targets_half, res_origins_half) 
        
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)
      
    maps = create_grid(im_sz)
    maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1)
  
    
    template = 'Epoch {}, Loss: {}'
    for i in range(TRAIN_EPOCHS):
        epoch = i + 1  
        train_step(maps, origins, targets)

        if epoch % 100 == 0:
            print (template.format(epoch, train_loss.result()))  

        
        if (epoch < 100 and epoch % 10 == 0) or\
           (epoch < 1000 and epoch % 100 == 0) or\
           (epoch % 1000 == 0):
            preds = model(maps, training=False)[:1]
            preds = tf.image.resize(preds, [im_sz, im_sz])
            
            res_targets, res_origins = warp(origins, targets, preds[...,:8], preds[...,8:])
            
            res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
            res_img = ((res_targets.numpy() + 1) * 127.5).astype(np.uint8)
            cv2.imwrite("train/a_to_b_%d.jpg" % epoch, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
            
            res_origins = tf.clip_by_value(res_origins, -1, 1)[0]
            res_img = ((res_origins.numpy() + 1) * 127.5).astype(np.uint8)
            cv2.imwrite("train/b_to_a_%d.jpg" % epoch, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
            
            np.save('preds.npy', preds.numpy())
        
def use_warp_maps(origins, targets):
    STEPS = 100
  
    preds = np.load('preds.npy')
    
    org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (STEPS - 1) 
    trg_strength = tf.reverse(org_strength, axis = [0])
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('morph/morph.mp4', fourcc, 48, (im_sz, im_sz))
    for i in range(STEPS):
        preds_org = preds * org_strength[i]
        preds_trg = preds * trg_strength[i]
    
        res_targets, res_origins = warp(origins, targets, preds_org[...,:8], preds_trg[...,8:])
        
        results = res_targets * trg_strength + res_origins * org_strength
        results = tf.clip_by_value(results, -1, 1)
        res_numpy = results.numpy()
    
        res_img = ((res_numpy[i] + 1) * 127.5).astype(np.uint8)
        cv2.imwrite("morph/%d.jpg" % (i+1), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

        video.write(cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
        
        if (i+1) % 10 == 0: print ('Image #%d saved.' % (i + 1))

    cv2.destroyAllWindows()
    video.release()   
    print ('Result video saved.')    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", help="Source file name", default = None)
    parser.add_argument("-t", "--target", help="Target file name", default = None)
    parser.add_argument("-e", "--train_epochs", help="Number of epochs to train network", default = TRAIN_EPOCHS, type=int)
    parser.add_argument("-a", "--add_scale", help="Scaler for addition map", default = add_scale, type=float)
    parser.add_argument("-m", "--mult_scale", help="Scaler for multiplication map", default = mult_scale, type=float)
    parser.add_argument("-w", "--warp_scale", help="Scaler for warping map", default = warp_scale, type=float)

    args = parser.parse_args()
    
    if not args.source: 
        print("No source file provided!")
        exit()
        
    if not args.target: 
        print("No target file provided!")
        exit()    
        
    
    TRAIN_EPOCHS = args.train_epochs
    add_scale = args.add_scale
    mult_scale = args.mult_scale
    warp_scale = args.warp_scale
    
    
    dom_a = cv2.imread(args.source, cv2.IMREAD_COLOR)
    dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
    dom_a = cv2.resize(dom_a, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_a = dom_a / 127.5 - 1

    dom_b = cv2.imread(args.target, cv2.IMREAD_COLOR)
    dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    dom_b = cv2.resize(dom_b, (im_sz, im_sz), interpolation = cv2.INTER_AREA)
    dom_b = dom_b / 127.5 - 1

    origins = dom_a.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    targets = dom_b.reshape(1, im_sz, im_sz, 3).astype(np.float32)

    produce_warp_maps(origins, targets)
    use_warp_maps(origins, targets)