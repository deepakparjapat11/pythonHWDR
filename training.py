import numpy as np
import cv2

image = cv2.imread('digi.png')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small=cv2.pyrDown(image)

#print ("image.shape")
cv2.imshow('Digits Image',small)
cv2.waitKey(0)
cv2.destroyAllWindows()

cells=[np.hsplit(row,100)for row in np.vsplit(gray,50)]

x=np.array(cells)
print("The shape of our cells array: "+str(x.shape))


train = x[:,:70].reshape(-1,400).astype(np.float32)
test = x[:,70:100].reshape(-1,400).astype(np.float32)


k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]
global knn
knn = cv2.KNearest()
knn.train(train, train_labels)
ret,result,neighbors,distance=knn.find_nearest(test,k=3)

matches=result==test_labels
correct=np.count_nonzero(matches)
accuracy=correct*(100.0/result.size)
print("Accuracy is = %.2f"%accuracy+"%")
