
##This part will introduce about Staistics and Applications
![Build](https://github.com/ntkme/github-buttons/workflows/build/badge.svg)
![Python Versions](https://img.shields.io/badge/python-v3.8-blue)


<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h2>AI Foundation Tutorial</h2>
<table >
  <tr style="border: 5px solid;">
    <th>Content</th>
  </tr>
  <tr >
    <td>Mean and Median</td>
    
  </tr>
  <tr>
    <td>Range and Histogram</td>
    
  </tr>
  <tr>
    <td>Variance</td>
   
  </tr>
  <tr>
    <td>Correlation Coefficient</td>
    
  </tr>
</table>

</body>
</html>

____________
#***Mean***

![](https://lh3.googleusercontent.com/_XcsP7xqrEg4GYDLj_i5fdu62vkazbtaz6P8lWeddyljwqU4uF0_Wq4CtQxr3RDqHeahlJN117zDpj8Q=w302-h206-rw)

![](https://lh3.googleusercontent.com/Hu8Zj8a18jiYNYvUFwh08YgGPE6nv6ID4Xr5AS41u7IoNm3yPkcW9JwGUtYtp-QMT9NUr66xO_gfVJNr=w396-h330-rw)

**Example**

![](https://lh3.googleusercontent.com/DDaUbG9Fl6GuDVtiMuBm0Ql3xatqsAOIVhamHlanu2snEnupEp0x6OZg9_hrmMNEXAMCCw1jl23v4dE9=w506-h330-rw)

![](https://lh3.googleusercontent.com/7P-sDxJCbDsnPG2ntwtmAWxnkC60VYUOSndp33EV2y_5WAKw-ONJcrwBGlZT0nnyNsT7NuIyAfcx36A_=w908-h330-rw)
### Example code:
```python
def calculate_mean(numbers):    #1
    s = sum(numbers)            #2
    N = len(numbers)            #3
    mean = s/N                  #4
    return mean                 #5
    
# Tạo mảng donations đại diện cho số tiền quyên góp trong 12 ngày
donations = [100, 60, 70, 900, 100, 200, 500, 500, 503, 600, 1000, 1200]

mean_value = calculate_mean(donations)
print('Trung bình số tiền quyên góp là: ', mean_value)
```
> Trung bình số tiền quyên góp là:  477.75



*  *#1* : Đặt tên là calculate_mean(), hàm này sẽ nhận đối số number, là chuỗi các số cần tính trung bình.  
* *#2* : Sử dụng hàm sum() để tính tổng dãy số cho trước.  
* *#3* : Sử dụng hàm len() tính chiều dài dãy số cần tính.  
* *#4* : Tính trung bình của dãy số bằng cách lấy tổng chia cho chiều dài.  
* *#5* : Cuối cùng ta cho hàm trả về giá trị mean tính được


## Application:
-Smooth or blur pictures, photos

-Correlation(~Convolution)  

![convolution](https://ars.els-cdn.com/content/image/1-s2.0-S016971611830021X-f09-24p1-9780444640420.jpg)

![convolution](https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png?w=736)



+-------------+--------------------------+----------------+
| Numpy       | Scipy                    | OpenCV         |
+=============+==========================+================+
| np.einsum() | scipy.signal.convole2D() | cv2.filter2D() |
+-------------+--------------------------+----------------+

**Kernels for computing mean**

*3x3 kernel*   
<img src = https://3.bp.blogspot.com/-Ue37dDpyfyE/WpYEF1GV-sI/AAAAAAAAI6Q/rL8gRmrQHF475-0A3t_JDBPcgtiZUCstACPcBGAYYCw/s1600/3x3%2BNormalized%2Bbox%2Bfilter.png alt = '3x3' height = 150 width = 190>

*5x5 kernel*  
<img src = https://4.bp.blogspot.com/-wWR_tPCRMvg/Uj8b0ahrYrI/AAAAAAAAAfQ/fzyASxHr55UFiZeIoeHSdALZm9shydjHgCPcBGAYYCw/s1600/Homogeneous%2BKernel.png alt = '5x5' height = 150 width = 190>



```python
output_image = c2.filter2D(input_image, cv2.CV_8U, kernel)
```

```python
#load image and blurring

import numpy as np
import cv2

# load image in grayscale mode
image = cv2.imread('stair.jpg', 0)

# create kernel
kernel = np.ones((5,5), np.float32) / 25.0

# compute mean for each pixel
dst = cv2.filter2D(image, cv2.CV_8U, kernel)

#show images
cv2.imshow('image', image)
cv2.imshow('dst', dst)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img src = https://srome.github.io/images/heatblur/output_5_1.png)>



Để hiểu rõ hơn hãy nhìn gif dưới đây
(Convolution gif)
![](https://topdev.vn/blog/wp-content/uploads/2019/08/Convolution_schematic.gif)




## Image Blurring
```python
# load image and blurring using mask-simple

import numpy as np
import cv2

# load image in grayscale mode
image = cv2.imread('beauty.jpg', 0)

# create kernel
kernel = np.ones((5,5), np.float32) / 25.0

# Select ROI (top_y, top_x, height, width)
roi = image[40:140, 150:280]

# compute mean for each pixel
roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

image[40:140, 150:280] = roi

# Show image
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img src = https://i.pinimg.com/originals/84/12/71/8412715b792dc8e26f384ce8d26e8304.jpg alt ='original_face' height = 250 width = 188>
<img src = https://lh3.googleusercontent.com/lliBxDa9-Fge2jlimlnahroHuBlAb9qGlrFyOALT2ls-P2mRTy6tlncV0aHY213YYIQDx8olPLt_Vm3e=w242-h330-rw alt = 'blur_face' heigt = 220 width = 188>

```python
# load image and blurring using face detection

import numpy as np
import cv2

# face detection setup
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image in grayscale mode
image = cv2.imread('confused.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#face detection
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    roi = image[y:y+h, x:x+w]
    
    #compute mean for each pixel
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    roi = cv2.filter2D(roi, cv2.CV_8U, kernel)
    
    #update
    image[y:y+h, x:x+w] = roi
    
# Show image
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img src = https://www10.lunapic.com/editor/working/162100434264841959?5898521098 alt = 'detect_Face' height = 250 width = 250>

<img src = https://www2.lunapic.com/editor/working/162104673826449510?2983037628 alt ='blur' height = 250 width = 250>

```python
# load image and blurring using face detection

import numpy as np
import cv2

# face detection setup
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# load image in grayscale mode
image = cv2.imread('mr.bean.jpg', 0)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# face detection
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# create kernel
kernel = np.ones((7,7), np.float32) / 49.0

# Draw the rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    

# Show image
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src = >

##Numpy review
```python
# numpy review
import numpy as np

arr = np.ones((5,5))
print(arr)

roi = arr[1:4, 1:4]
roi = roi + 1
print(roi)

arr[1:4, 1:4] = roi
print(arr)
```

# Stereo matching

##Tracking and bluring image:
code example
____________
# ***Median***
>Data
X={X1,X2,...}

>Formula:
Step 1:Sort X-> S
Step 2:
  if N is odd, then m=
  If N is even, then m=

>***Exmaple:*** Given the data
X = {4,1,6,2,8}
N=5

>Step 1:
   S={4,1,6,2,8}

>Step 2: N=5
   k=
(code)
##Median
```python
def calculate_median(numbers):
    N = len(numbers)
    numbers.sort()
    if N%2 == 0:
        m1 = N/2
        m2 = (N/2) + 1
        m1 = int(m1) - 1
        m2 = int(m2) - 1
        median = (numbers[m1] + numbers[m2]) / 2
    else:
        m = (N+1)/2
        m = int(m)-1
        median = numbers[m]
    return median
```
### *Application*: Noise reduction
####*Image Denoising
(Image example)
(Code)
```python
import numpy as np
import cv2

img1 = cv2.imread('mrbean_noise.jpg')
img2 = cv2.meadianBlur(img1, 3)

# Show image
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```            

##*Mean and Median
###*Comparison
>Data:
X={}

>Formula
Equation:
Step 1:
Step 2:
____________
#***Range***
##Procedure
(code)
```python
def find_range(numbers):        #1
    lowest = min(numbers)       #2
    highest = max(numbers)      #3
    r = highest = lowest        #4
    print('Lowest: {0}\tHighest: {1}\tRange: {2}'.format(lowest, highest, r))

# data
points = [7, 8, 9, 2, 10, 9, 9, 9, 9, 4, 5, 6, 1, 5, 6, 1, 5, 6, 7, 8, 6, 1, 10, 6, 6]
find_range(points)
```


____________
#***Histogram for grayscale***
(image)
-Subtitile N is sum of pixel in pictures and (nb) is number of (bth) pixel
-Density histogram value at (bth) is calculated such as:(equation)
-Cumulative histogram at (bth) is calculated such as:
(image example)

##Histogram equalization

Formula: (here)

This formula is used for increasing contrast of pictures
(image)

(image example)
____________
#***Variance***
##Definition:
>***Formula:***
**mean**
**variance**
**Standard deviation**

>***Example***:

(code)
```python
# variance
def calculate_mean(numbers):                        #1
    s = sum(numbers)
    N = len(numbers)
    mean = s/N
    retrun mean
    
def calculate_variance(numbers):                    #2
    mean = calculate_mean(numbers)                  #3
    
    diff = []                                       #4
    for num in numbers:
        diff.append(num-mean)
        
    squared_diff = []                               #5
    for d in diff:
        squared_diff.append(d**2)
        
    sum sqared_diff = sum(squared_diff)
    variance - sum_squared_diff/len(numbers)
    
    return variance
```
(image)

***Application:*** Variance(~Standard deviation) used to find **texture** for a image

##Implementation:
(image)
(code)
```python
import numpy as np
import cvv2
import math
form scipy.ndimage.filters import generic_filter

img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('edge_s1.jpg', gray)

x = gray.astype('float')
x_flit = generic_filter(x, np.std, size = 7)
cv2.imwrite('edge_ss2.jpg', x_filt)

x_filt[x_filt < 20] = 0
cv2.imwrite('edge_s3.jpg', x_filt)

maxv = np.max(x_filt)
print(maxv)

x_filt = x_filt*2.5
cv2.imwrite('edge_s4.jpg', x_filt)
```
____________
#Correlation Coefficient
##Formula: 
**Subtitule:** x, y are 2 random variables
(equtaion/property)
(c0de)
```python
def find_corr_x_y(x,y):
    n = len(x)
    prod = []
    for xi, yi in zip(x,y):
        prod.append(xi*yi)
        
    sum_prod_x_y = sum(prod)
    
    sum_x = sum(x)
    sum_y = sum(y)
    
    squared_sum_x = sum_x**2
    squared_sum_y = sum_x**2
    
    x_square = []
    for xi in x:
        x_square.append(xi**2)
    x_square_sum = sum(x_square)
    
    y_square = []
    for yi in y:
        y_square.append(yi**2)
    y_square_sum = sum(y_square)
    
    # Use formula to calculate correlation
    numerator = n*sum_prod_x_y - sum_x*sum_y
    denomiator_term1 = n*x_square_sum - squared_sum_x
    denominator_term2 = n*y_square_sum - squared_sum_y
    denominator = (denominator_term1*denominator_term2)**0.5
    correlation = numerator/denominator
```

##Application:
###For patch maching:
(image and code)
```python
import numpy as np
from PIL import Image

# load ảnh và chuyển về kiểu list
image1 = Image.open('image/img1.png')
image2 = Image.open('image/img2.png')
image3 = Image.open('image/img3.png')
image4 = Image.open('image/img4.png')

image1_lit = np.aâay(image1).flatten().tolist()
image2_list = np.asarray(image2).flatten().tolist()
image3_list = np.asarray(image3).flatten().tolist()
image4_list = np.asarray(image4).flatten().tolist()

# tính correlation coefficient
corr_1_2 = find_corr_x_y(image1_list, image2_list)
corr_1_3 = find_corr_x_y(image1_list, image3_list)
corr_1_4 = find_corr_x_y(image1_list, image4_list)

print('corr_1_2:', corr_1_2)
print('corr_1_3:', corr_1_3)
print('corr_1_4:' corr_1_4)
```

###For template matching
(image and code)
```python
grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
output = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
```

```python
#template matching

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to grayscale
image = cv.imread('image.png', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print(type(gray[0][0]))

template = cv.imread('teamplate.png', 0)
w, h = template.shape[::-1]

# Apply template matching
corr_map = cv2.matchTemplate(gray, template, cv2.TM_COEFF_NORMED)

# Save correlation map
corr_map = (corr_map + 1.0)*127.5
corr_map = corr_map.astype('uint8')
cv.imwrite('corr_map_grayscale.png', corr_map)
```

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to grayscale
image = cv.imread('image.png', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = cv.imread('teamplate.png', 0)
w, h = template.shape[::-1]

# Apply template matching
corr_map = cv2.matchTemplate(gray, template, cv2.TM_COEFF_NORMED)
min_val, max_val, min_loc = cv.minMaxLoc(corr_map)

# take minimum
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# draw
cv.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# save corr map in grayscale
corr_map = (corr_map+1.0)*127.5
corr_map = corr_map.astype('uint8')
cv.imwrite('corr_map_grayscale.png', corr_map)

#applyColorMap
corr_map = cv2.applyColorMap(corr_map, cv2.COLORMAP_JET)

# save results
cv.imwrite('corr_map_color.png', corr_map)
cv.imwrite('result.png', image)
