#!/usr/bin/env python
"""
Aligner is an image aligner specialization of Net.
"""

import numpy as np
import caffe
import cv2
import pdb

class Aligner(caffe.Net):
    """
    Aligner extends for image align
    """
    def __init__(self, model_dir):
        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709
        
        self.PNet = caffe.Net(model_dir+"/det1.prototxt", model_dir+"/det1.caffemodel", caffe.TEST)
        self.RNet = caffe.Net(model_dir+"/det2.prototxt", model_dir+"/det2.caffemodel", caffe.TEST)
        self.ONet = caffe.Net(model_dir+"/det3.prototxt", model_dir+"/det3.caffemodel", caffe.TEST)
        print "================"
        print "Finish init"


    def bbreg(self, boundingbox, reg):
        reg = reg.T 
        
        # calibrate bouding boxes
        if reg.shape[1] == 1:
            print "reshape of reg"
            pass # reshape of reg
        w = boundingbox[:,2] - boundingbox[:,0] + 1
        h = boundingbox[:,3] - boundingbox[:,1] + 1

        bb0 = boundingbox[:,0] + reg[:,0]*w
        bb1 = boundingbox[:,1] + reg[:,1]*h
        bb2 = boundingbox[:,2] + reg[:,2]*w
        bb3 = boundingbox[:,3] + reg[:,3]*h
        
        boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
        return boundingbox


    def pad(self, boxesA, w, h):
        boxes = boxesA.copy() # shit, value parameter!!!
        
        tmph = boxes[:,3] - boxes[:,1] + 1
        tmpw = boxes[:,2] - boxes[:,0] + 1
        numbox = boxes.shape[0]

        dx = np.ones(numbox)
        dy = np.ones(numbox)
        edx = tmpw 
        edy = tmph

        x = boxes[:,0:1][:,0]
        y = boxes[:,1:2][:,0]
        ex = boxes[:,2:3][:,0]
        ey = boxes[:,3:4][:,0]
       
        tmp = np.where(ex > w)[0]
        if tmp.shape[0] != 0:
            edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
            ex[tmp] = w-1

        tmp = np.where(ey > h)[0]
        if tmp.shape[0] != 0:
            edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
            ey[tmp] = h-1

        tmp = np.where(x < 1)[0]
        if tmp.shape[0] != 0:
            dx[tmp] = 2 - x[tmp]
            x[tmp] = np.ones_like(x[tmp])

        tmp = np.where(y < 1)[0]
        if tmp.shape[0] != 0:
            dy[tmp] = 2 - y[tmp]
            y[tmp] = np.ones_like(y[tmp])
        
        # for python index from 0, while matlab from 1
        dy = np.maximum(0, dy-1)
        dx = np.maximum(0, dx-1)
        y = np.maximum(0, y-1)
        x = np.maximum(0, x-1)
        edy = np.maximum(0, edy-1)
        edx = np.maximum(0, edx-1)
        ey = np.maximum(0, ey-1)
        ex = np.maximum(0, ex-1)
        
        return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]


    def rerec(self, bboxA):
        # convert bboxA to square
        w = bboxA[:,2] - bboxA[:,0]
        h = bboxA[:,3] - bboxA[:,1]
        l = np.maximum(w,h).T
        
        bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
        bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
        return bboxA


    def nms(self, boxes, threshold, type):
        """nms
        :boxes: [:,0:5]
        :threshold: 0.5 like
        :type: 'Min' or others
        :returns: TODO
        """
        if boxes.shape[0] == 0:
            return np.array([])
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,4]
        area = np.multiply(x2-x1+1, y2-y1+1)
        I = np.array(s.argsort()) # read s using I
        
        pick = [];
        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type == 'Min':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where( o <= threshold)[0]]
        return pick


    def generateBoundingBox(self, map, reg, scale, t):
        stride = 2
        cellsize = 12
        cell_w= 32
        cell_h= 12
        stride_w = 8
        stride_h = 2

        dx1 = reg[0,:,:]
        dy1 = reg[1,:,:]
        dx2 = reg[2,:,:]
        dy2 = reg[3,:,:]
        (y, x) = np.where(map >= t)

        yy = y
        xx = x
        
        score = map[y,x]
        reg = np.array([dx1[y,x], dy1[y,x], dx2[y,x], dy2[y,x]])

        if reg.shape[0] == 0:
            pass
        boundingbox = np.array([yy, xx]).T

        #bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
        #bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
        bb1 = np.fix((np.array([stride_h, stride_w]) * (boundingbox) + 1) / scale).T
        bb2 = np.fix((np.array([stride_h, stride_w]) * (boundingbox) + np.array([cell_h, cell_w]) - 1 + 1) / scale).T

        score = np.array([score])

        #pdb.set_trace()
        boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

        return boundingbox_out.T


    def drawBoxes(self, im, boxes):
        for bx in boxes:
            cv2.rectangle(im, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (0,255,0), 1)
        return im


    def drawPoints(self, im, points):
        for ps in points:
            for p in ps:
                cv2.circle(im, (int(p[0]), int(p[1])), 5, (0,255,0), -1)
        return im


    def align(self, input, fastresize=False):
        """
        align image
        """
        img = input.copy()
        tmp = img[:,:,2].copy()
        img[:,:,2] = img[:,:,0]
        img[:,:,0] = tmp
        
        img2 = img.copy()

        factor_count = 0
        total_boxes = np.zeros((0,9), np.float)
        points = []
        h = img.shape[0]
        w = img.shape[1]
#        minl = min(h, w)
        if w*12 > h*32:
            minl = h
            m = 12. / self.minsize
            min_lb = 12
        else:
            minl = w
            m = 32. / self.minsize
            min_lb = 32

        img = img.astype(float)
#        m = 12.0/self.minsize
        minl = minl*m
        print "original: {} {}".format(w,h)
        print "minl: {}\nm: {}\nminsize: {}".format(minl, m, self.minsize)

        # create scale pyramid
        scales = []
        while minl >= min_lb:
            scales.append(m * pow(self.factor, factor_count))
            minl *= self.factor
            factor_count += 1
        
        # first stage
        for scale in scales:
            hs = int(np.ceil(h*scale))
            ws = int(np.ceil(w*scale))

            if fastresize:
                im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
                im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
            else: 
                im_data = cv2.resize(img, (ws,hs)) # default is bilinear
                im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]

            #im_data = np.swapaxes(im_data, 0, 2)
            im_data = np.transpose(im_data, (2,0,1))
            im_data = np.array([im_data], dtype = np.float)
            self.PNet.blobs['data'].reshape(1, 3, hs, ws)
            self.PNet.blobs['data'].data[...] = im_data
            #print "forward"
            #print im_data.shape
            out = self.PNet.forward()
        
            boxes = self.generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, self.threshold[0])
            if boxes.shape[0] != 0:
                pick = self.nms(boxes, 0.5, 'Union')
                if len(pick) > 0 :
                    boxes = boxes[pick, :]

            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)
             
        #####
        # 1 #
        #####
        import cPickle

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # nms
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            # revise and convert to square
            #pdb.set_trace()
            regw = total_boxes[:,3] - total_boxes[:,1]
            regh = total_boxes[:,2] - total_boxes[:,0]
            t1 = total_boxes[:,0] + total_boxes[:,5]*regh
            t2 = total_boxes[:,1] + total_boxes[:,6]*regw
            t3 = total_boxes[:,2] + total_boxes[:,7]*regh
            t4 = total_boxes[:,3] + total_boxes[:,8]*regw
            t5 = total_boxes[:,4]
            #total_boxes = np.array([t1,t2,t3,t4,t5]).T
            total_boxes = np.array([t2,t1,t4,t3,t5]).T

            return total_boxes
            stage1_out = "/data/image_server/user/ouxiaotian/MTCNN/mtcnn-caffe/Experiments/plates/stage1_out.pkl"
            with open(stage1_out, "wb") as f:
                cPickle.dump(total_boxes, f, -1)

            total_boxes = self.rerec(total_boxes) # convert box to square
            
            total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # construct input for RNet
            tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

            # RNet
            tempimg = np.swapaxes(tempimg, 1, 3)
            
            self.RNet.blobs['data'].reshape(numbox, 3, 24, 24)
            self.RNet.blobs['data'].data[...] = tempimg
            out = self.RNet.forward()

            score = out['prob1'][:,1]
            pass_t = np.where(score>self.threshold[1])[0]
            
            score =  np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)

            mv = out['conv5-2'][pass_t, :].T
            if total_boxes.shape[0] > 0:
                pick = self.nms(total_boxes, 0.7, 'Union')
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    total_boxes = self.bbreg(total_boxes, mv[:, pick])
                    total_boxes = self.rerec(total_boxes)
                
            #####
            # 2 #
            #####
            numbox = total_boxes.shape[0]
            if numbox > 0:
                # third stage
                total_boxes = np.fix(total_boxes)
                [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(total_boxes, w, h)

                tempimg = np.zeros((numbox, 48, 48, 3))
                for k in range(numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                    tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                    tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
                tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                    
                # ONet
                tempimg = np.swapaxes(tempimg, 1, 3)
                self.ONet.blobs['data'].reshape(numbox, 3, 48, 48)
                self.ONet.blobs['data'].data[...] = tempimg
                out = self.ONet.forward()
                
                score = out['prob1'][:,1]
                points = out['conv6-3']
                pass_t = np.where(score>self.threshold[2])[0]
                points = points[pass_t, :]
                score = np.array([score[pass_t]]).T
                total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
                
                mv = out['conv6-2'][pass_t, :].T
                w = total_boxes[:,3] - total_boxes[:,1] + 1
                h = total_boxes[:,2] - total_boxes[:,0] + 1

                points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
                points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

                if total_boxes.shape[0] > 0:
                    total_boxes = self.bbreg(total_boxes, mv[:,:])
                    pick = self.nms(total_boxes, 0.7, 'Min')
                    
                    if len(pick) > 0 :
                        total_boxes = total_boxes[pick, :]
                        points = points[pick, :]

        # sort by box score
        ids = sorted(xrange(len(total_boxes)), key=lambda id: total_boxes[id][-1], reverse=True)
        # boxes
        boxes = map(lambda id: list(total_boxes[id]), ids)
        # points
        points = map(lambda id: map(lambda k: (points[id][k], points[id][5+k]), xrange(5)), ids)

        #input = self.drawBoxes(input, boxes)
        #input = self.drawPoints(input, points)
        #cv2.imwrite("output.jpg", input)
        return boxes, points

