import Alexnet
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import time
import sys
from torch.autograd import Variable

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output
        #self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

class FilterVisualizer():
    def __init__(self, size=128, upscaling_steps=12, upscaling_factor=1.2):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = Alexnet.alexnet(pretrained=True).eval()

    def visualize(self, layer, filter,img, lr=0.05, opt_steps=1, blur=None, save=False):
        sz = self.size
        img=cv2.resize(img/255, (sz, sz), interpolation = cv2.INTER_CUBIC)
        activations = SaveFeatures(list(self.model.children())[0][layer])  # register hook
        print(list(self.model.children())[0])
        for step in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            img=img.transpose(2,0,1)
            img_var = Variable(torch.from_numpy(img).unsqueeze_(0).float(), requires_grad=True)  # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img = img_var.data.cpu().numpy()[0].transpose(1,2,0)
            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            
            img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
            
        activations.close()
        return cv2.resize(np.clip(self.output, 0, 1) ,(1600, 900), interpolation = cv2.INTER_CUBIC)
        

def stream(input_channel=0):
    Visualizer=FilterVisualizer()

    window_name = "window"
    interframe_wait_ms = 30

    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(input_channel)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    start=time.time()
    frame_count=0
    filter=0
    while (True):
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            print("Reached end of video, exiting.")
            break
        if frame_count%100==0:
            if filter==190:
                filter=0
            filter+=1
        frame=Visualizer.visualize(6,filter,frame,lr=(frame_count%100)/100.0*0.05,blur=2)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(interframe_wait_ms) & 0x7F == ord('q'):
            print("Exit requested.")
            break
        frame_count+=1

    cap.release()
    cv2.destroyAllWindows()
    print((time.time()-start)/frame_count)

def image(path):
    img=cv2.imread(path)
    Visualizer=FilterVisualizer()
    
    #cv2.imwrite('view.png', img)
    filter=20
    while True:
        img=Visualizer.visualize(10,filter,img,lr=0.002,opt_steps=10,blur=2)
        filter+=1
        cv2.imshow('test', img)
        if cv2.waitKey(30) & 0x7F == ord('q'):
            print("Exit requested.")
            break

def main():
    mode=sys.argv[1]
    if mode=='stream':
        stream(sys.argv[2])
    elif mode=='image':
        image(sys.argv[2])
    else:
        print('got unexpected key word, use stream [input_channel(int)] or image [image_path]')
    
    

if __name__ == '__main__':
    main()