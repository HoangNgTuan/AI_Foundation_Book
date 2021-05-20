## **This part will introduce about Statistics and Applications**
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
# ***Mean***

<img src ='https://lh3.googleusercontent.com/_XcsP7xqrEg4GYDLj_i5fdu62vkazbtaz6P8lWeddyljwqU4uF0_Wq4CtQxr3RDqHeahlJN117zDpj8Q=w302-h206-rw'>

<img src = 'https://lh3.googleusercontent.com/Hu8Zj8a18jiYNYvUFwh08YgGPE6nv6ID4Xr5AS41u7IoNm3yPkcW9JwGUtYtp-QMT9NUr66xO_gfVJNr=w396-h330-rw' alt="Formula" width=180>

&nbsp;  

**Example**

<img src = 'https://lh3.googleusercontent.com/DDaUbG9Fl6GuDVtiMuBm0Ql3xatqsAOIVhamHlanu2snEnupEp0x6OZg9_hrmMNEXAMCCw1jl23v4dE9=w506-h330-rw' width=220>  

<img src = 'https://lh3.googleusercontent.com/7P-sDxJCbDsnPG2ntwtmAWxnkC60VYUOSndp33EV2y_5WAKw-ONJcrwBGlZT0nnyNsT7NuIyAfcx36A_=w908-h330-rw' width=400>  

&nbsp;  

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
![](https://lh3.googleusercontent.com/sEQFtV8cUXF7kD3bQ47iVCLm5--46lP3RkvfIAnMeuK2-U5bdyHeAK-5CxWea7-0THZSNc7lVC0DbZM9Cg=w767-h72-rw) 

*  *#1* : Đặt tên là calculate_mean(), hàm này sẽ nhận đối số number, là chuỗi các số cần tính trung bình.  
* *#2* : Sử dụng hàm sum() để tính tổng dãy số cho trước.  
* *#3* : Sử dụng hàm len() tính chiều dài dãy số cần tính.  
* *#4* : Tính trung bình của dãy số bằng cách lấy tổng chia cho chiều dài.  
* *#5* : Cuối cùng ta cho hàm trả về giá trị mean tính được

&nbsp;  


## Application:
-Smooth or blur pictures, photos

-Correlation(~Convolution)  

![convolution](https://ars.els-cdn.com/content/image/1-s2.0-S016971611830021X-f09-24p1-9780444640420.jpg)

![convolution](https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png?w=736)

&nbsp;  

<table>
<thead>
	<tr>
		<th style="text-align:center">Numpy</th>
		<th style="text-align:center">Scipy</th>
		<th style="text-align:center">OpenCV</th>
	</tr>
</thead>
<tbody>
	<tr>
		<td style="text-align: center">np.einsum()</td>
		<td style="text-align: center">scipy.signal.convole2D()</td>
		<td style="text-align: center">cv2.filter2D()</td>
	</tr>
</tbody>
</table>

&nbsp;  
&nbsp;  

**Kernels for computing mean**

*3x3 kernel*   
<img src = https://3.bp.blogspot.com/-Ue37dDpyfyE/WpYEF1GV-sI/AAAAAAAAI6Q/rL8gRmrQHF475-0A3t_JDBPcgtiZUCstACPcBGAYYCw/s1600/3x3%2BNormalized%2Bbox%2Bfilter.png alt = '3x3' height = 150 width = 190>

*5x5 kernel*  
<img src = https://4.bp.blogspot.com/-wWR_tPCRMvg/Uj8b0ahrYrI/AAAAAAAAAfQ/fzyASxHr55UFiZeIoeHSdALZm9shydjHgCPcBGAYYCw/s1600/Homogeneous%2BKernel.png alt = '5x5' height = 150 width = 190>

&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
## Image Blurring
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
cv2.imshow(image)
cv2.imshow(dst)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<img src = 'https://lh3.googleusercontent.com/vwtjw-1ckRa9Q0aB83PcZ0nCgQ0MgmGE9g79g0f1Yf_QtSzxpW0H9W1nVxUOghOzCBsIKI28BkzlMdq1yA=w186-h330-rw' alt='gray_steph_curry'>

<img src ='https://lh3.googleusercontent.com/KyKG576GhD5MliREBAH6QxQkRcZ4OSfmxbxY7Rn3nkAjoBhw0dXlVsr-DX1S0O6fEMlXexsttTierGTA1w=w186-h330-rw' alt='gray_blur_steph_curry'>

&nbsp;  


Để hiểu rõ hơn hãy nhìn gif dưới đây:  
![gif](https://topdev.vn/blog/wp-content/uploads/2019/08/Convolution_schematic.gif)
&nbsp;  
&nbsp; 
&nbsp;  
 

```python
# load image and blurring using mask-simple

import numpy as np
import cv2

# load image in grayscale mode   # Why ?? T.T
image = cv2.imread('beauty.jpg', 0)

# create kernel
kernel = np.ones((5,5), np.float32) / 25.0

# Select ROI (top_y, top_x, height, width)
roi = image[185:740, 170:570]

# compute mean for each pixel
roi = cv2.filter2D(roi, cv2.CV_8U, kernel)

image[185:740, 170:570] = roi

# Show image
cv2.imshow('roi', roi)
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```
<table>
    <tbody>
	    <tr>
    	    <td style='text-align:center'>
        	    <img src='https://lh3.googleusercontent.com/J5MS8lvlZE7jg7RcbXVPH0ZGHNVHh-wYIoWVyx7fWK65LWfkJug3F-ZMCBnRdCv48OABdWpd4wbRSYdD=w257-h330-rw' height = 190></td>  
            <td style='text-align:center'><img src = "https://lh3.googleusercontent.com/5ZrItsIhPSJ3SeQBADmrY4IhQpCbYXvXfaoKGVEQEr0_Y980kxz1tx5zSA4tqG7aHOxB8S8qyKK0RiYl=w252-h330-rw" alt = 'imshow' ></td>
        </tr>
        <tr>
    	    <td style='text-align:center'>roi</td>
            <td style='text-align:center'>image</td>
        </tr>
    </tbody>
</table>


```python
# load image and blurring using face detection

import numpy as np
import cv2

# face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load image in grayscale mode
image = cv2.imread('beauty.jpg', 0)

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
cv2.imshow('image', image)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```

<img src = 'https://lh3.googleusercontent.com/RvcxJmrZjX5eBCCMHKylDxVNA-mNSwJr6nRphh6z80L6jGlzu9SgAnHyCyx9rcUxc6KFfqq9I7bcyfly=w242-h330-rw' alt ='rectangle_face'>



```python
# load image and blurring using face detection

import numpy as np
import cv2

# face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load image in grayscale mode
image = cv2.imread('beauty.jpg')

#show image with default format
cv2.imshow(image)

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

<img src = "https://lh3.googleusercontent.com/DLQX6hGEnPkLNqQ6S_FQqlHTVczZ1FBmuqytKMzsfjx8xEKzfIBygLxcEK0cnpHRI3tu8wrNxzaHWXuU=w242-h330-rw" alt = 'imshow' > 
 
<table>
    <tbody>
	    <tr>
    	    <td style='text-align:center'>
        	    <img src = 'https://lh3.googleusercontent.com/4O63kApSW9ueZ9DIafzqfwisEyF_kAZECYchDZLgHSfq633of93lCEsU8s_6U8WZ9O4097SzToZ-6Gui=s330-rw' alt = 'roi' height = 180></td>  
            <td style='text-align:center'>
                <img src = 'https://lh3.googleusercontent.com/CB_OQKVMAksUEM_wE38pDrpq6-z22z1wV9CV6GCjAjIUjiVZw4EQl47JZ0QzOoZagIXzMxfys_ul5yxT=w242-h330-rw' ></td>
        </tr>
        <tr>
    	    <td style='text-align:center'>roi</td>
            <td style='text-align:center'>image</td>
        </tr>
    </tbody>
</table>
    
&nbsp;   
&nbsp;   
&nbsp;   
## Stereo matching
[Source](https://aivietnam.ai/courses/aisummer2019/lessons/stereo-matching/)  
Stereo matching là một lĩnh vực nghiên cứu trong computer vision, nhằm tính khoảng cách từ camera đến các object. Stereo matching dùng hệ thống gồm 2 camera (gọi là stereo camera) để bắt trước cặp mắt của con người trong việc tính khoảng cách.  
<img src='https://ae01.alicdn.com/kf/HTB1.H16OVXXXXXbaXXXq6xXFXXXa/Truy-n-video-kh-ng-d-y-3D-FPV-camera-stereo.jpg' width=300>

- Stereo matching dùng stereo camera, là hệ thống gồm 2 camera.  
- Depth map chứa khoảng cách cho mọi pixel của ảnh input.  
- Stereo matching hiện được ứng dụng rộng rãi, và vẫn được tiếp tục nghiên cứu.  
- Khi tìm hiểu về stereo matching, bạn sẽ gặp thuật ngữ disparity map. Disparity map chính là dạng ‘dữ liệu thô’ của depth map. Hai map này có thể chuyển đổi qua lại bằng một công thức đơn giản.

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt  

imgL = cv.imread('tsukuba_l.png',0)
imgR = cv.imread('tsukuba_r.png',0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
```
![](https://docs.opencv.org/master/disparity_map.jpg)

&nbsp;  

## Numpy review
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

<img src ='https://lh3.googleusercontent.com/v4J1gXzZIzH0t8p8QLBVKk8lp4GJ58S6N807tVHxDlDeJZ5V01PX4-CuKvtQFjUTZ0A2TygMlEI5BASd=w221-h330-rw' alt = 'numpy_review'>


## Tracking blurry image
[Source](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)    

Dataset
![](https://www.pyimagesearch.com/wp-content/uploads/2015/09/detecting_blur_dataset.png)  


Define Algorithm
```python
# import the necessary packages
from imutils import paths
import argparse
import cv2  

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()  

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())
```

Mark the image 'blurry' or 'Not blurry'
```python
# loop over the input images
for imagePath in paths.list_images(args["images"]):
	# load the image, convert it to grayscale, and compute the
	# focus measure of the image using the Variance of Laplacian
	# method
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	text = "Not Blurry"

	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
	if fm < args["threshold"]:
		text = "Blurry"
	# show the image
	cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Image", image)
	key = cv2.waitKey(0)
```   
<img src ='https://www.pyimagesearch.com/wp-content/uploads/2015/09/detecting_blur_result_001-785x1024.jpg' height=250>

<img src='https://www.pyimagesearch.com/wp-content/uploads/2015/09/detecting_blur_result_003.jpg' height = 250>


&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
____________
# ***Median***
<img src='https://lh3.googleusercontent.com/_XcsP7xqrEg4GYDLj_i5fdu62vkazbtaz6P8lWeddyljwqU4uF0_Wq4CtQxr3RDqHeahlJN117zDpj8Q=w302-h206-rw'>    

<img src='https://lh3.googleusercontent.com/kSbePcfROT1I9yiB1UoSiuFGdtgU__8NgmsiUEjR2EcEg--dG2KcNNW6huUQS78TBwRv-x7DB56CzN6q-w=w548-h330-rw'>

&nbsp;  

**Example 1:**  
<img src='https://lh3.googleusercontent.com/_jPqu1LY2k4mP5IpnoZCCwiZMfhTL-h-YAwflJlbQNzILcmGc8kZXO_Wa44xjLjMiZ15YLX7j2RxT4pFaQ=w375-h203-rw' width=200>  
<img src='https://lh3.googleusercontent.com/vdDKkQ9M1pmLHCHMqDPQ4tsZA9Vu1vm3FX3a2a6FfWCMI9bDItfrEM95HY38B0BSh3b58CBrA6Mx4U2Q2Q=w378-h146-rw'>  
<img src='https://lh3.googleusercontent.com/ZqsKJvHO6RWUhNMwKC-0MgazZa8OunxzcuAF6-NTxudszJiCONrjCDhgRwFl0Lo-fv0ZXmGJ51wSxZni9g=w374-h242-rw'>

&nbsp;  
**Example 2:**  
<img src='https://lh3.googleusercontent.com/DDaUbG9Fl6GuDVtiMuBm0Ql3xatqsAOIVhamHlanu2snEnupEp0x6OZg9_hrmMNEXAMCCw1jl23v4dE9=w506-h330-rw' width=200>  
<img src='https://lh3.googleusercontent.com/Y89tMmuVRYM8vh3K-8mO7oZGDA-GU8bO5_R5mnm_61ERjQPP8z2Fz0VwkRZGwb4sMPFDm8k8hqpQD0_P0Q=w366-h146-rw'>   
<img src='https://lh3.googleusercontent.com/9MGamxsNR_bS73U8L7y4WAmTD0xkEPqEUpgTfQpIkcdUY_sEseWTrPNeE3W0GLxBtFcGeBLheyzJ6rNSRw=w372-h234-rw'>

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
#### *Image Denoising*
```python
import numpy as np
import cv2

img1 = cv2.imread('beauty_noise.jpg')
img2 = cv2.medianBlur(img1, 3)

# Show image
cv2.imshow('image 1', img1)
cv2.imshow('image 2', img2)

# waiting for any keys pressed and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
```            

<img src = 'https://lh3.googleusercontent.com/mp41Rwwp7cWoNc3oxuFSy9kWS3cpdZtKI3k9Nz0aXcXyGIWQdhiwDbHRwQ9ypEqoOuFapY4fOQ1IeMjYcg=w225-h330-rw' alt = 'beauty_2'>
<img src = 'https://lh3.googleusercontent.com/W04B-0BEZs7NFVlFywyaJRSKiPrQxf6JYchMovbkvkIM3pIxjHpEdZ9jHhfBX3ZYBGhkbNlzpO9NrHlYXg=w225-h330-rw' alt ='beauty_noise'>

&nbsp;  
&nbsp;  
## *Mean and Median*  

<a name="Definitions_of_mean_and_median" class="dfnanchor"></a>      </div>
        <div id="comptable"><h2 class="chartHeading"></h2>
        <table id="diffenTable">
            <caption>Bảng so sánh Mean và Median</caption>
            <thead>
                <tr><th class=acol>        </th><th class=vcol scope=col>Mean</th><th class=vcol scope=col>Median</th></tr></thead><tbody><tr  class='comparisonRow diff ' id='row1'><th class='acol' scope='row'>Định nghĩa</th>                    <td id="valtd1_1" class="vcol">Mean là trung bình về mặt đại số học hay trong các bài toán phân phối. Nó thường được sử dụng trong việc tìm giá trị bình quân của một tập hợp số</td>
                    <td id="valtd1_2" class="vcol">Median là một giá trị số tách biệt phần trên của một mẫu, dân số hoặc một phân phối xác suất, với phần dưới của nó</td>
                    </tr><tr  class='comparisonRow diff ' id='row2'><th class='acol' scope='row'>Khả năng áp dụng</th>                    <td id="valtd2_1" class="vcol">Dùng trong phân phối chuẩn</td>
                    <td id="valtd2_2" class="vcol">Thường dùng cho phân phối lệch</td>
                    </tr><tr  class='comparisonRow diff ' id='row3'><th class='acol' scope='row'>Mối liên hệ với tập dữ liệu</th>                    <td id="valtd3_1" class="vcol">Mean không phải là một công cụ mạnh do có thể bị ảnh hưởng bởi yếu tố bên ngoài</td>
                    <td id="valtd3_2" class="vcol">Median phù hợp với các bài toán phân phối lệch để lấy giá trị bình quân, vì vậy nó là một công cụ mạnh và đáng tin cậy</td>
                    </tr><tr  class='comparisonRow diff lastRow' id='row4'><th class='acol' scope='row'>Cách tính</th>                    <td id="valtd4_1" class="vcol">Mean được tính bắng cách cộng tất cả các giá trị và chia cho tổng số dữ liệu</td>
                    <td id="valtd4_2" class="vcol">Median là số ở chính giữa tập giá trị. Median có thể được tính bằng cách sắp xếp các giá trị tăng dần, và rồi tìm kiếm giá trị ở vị trí chính giữa</td>
                    </tr></tbody></table></div>      <div id="essay">


&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  

____________
# ***Range***
## Procedure  

<img src ='https://cdn.corporatefinanceinstitute.com/assets/range1.png' width=500>

```python
def find_range(numbers):        #1
    lowest = min(numbers)       #2
    highest = max(numbers)      #3
    r = highest - lowest        #4
    print('Lowest: {0}\tHighest: {1}\tRange: {2}'.format(lowest, highest, r))

# data
points = [7, 8, 9, 2, 10, 9, 9, 9, 9, 4, 5, 6, 1, 5, 6, 1, 5, 6, 7, 8, 6, 1, 10, 6, 6]
find_range(points)
```
<img src = 'https://lh3.googleusercontent.com/8OKST5Mb2E9FEQwUOSZIqhTm0UKHxGWVYZUr_ha44ji5_GNbbQZjHuSVqPmzfZsdCjg9OPOAZXRQXe7JKw=w762-h78-rw' alt = 'range'>

* #1 Tên hàm  
* #2 Số nhỏ nhất trong dãy  
* #3 Số lớn nhất trong dãy  
* #4 Tìm range

&nbsp;  
&nbsp;  
&nbsp;  
____________
# ***Histogram for grayscale***

Histogram (Đường tần suất) cho chúng ta cái nhìn trực quan về sự phân bố cường độ của pixels trong bức ảnh grayscale.  
Nó là một biểu đồ với trục hoành biểu thị mức xám của pixels (thường là từ 0 -> 255); trục tung biểu thị số pixel trong bức ảnh tương ứng với giá trị mức xám đó.  
* Phần bên trái của đường tần suất thể hiện số lượng pixels có màu tối  
* Phần bên phải của đường tần suất thể hiện số lượng pixels có màu sáng


<img src ='https://media.geeksforgeeks.org/wp-content/uploads/OpenCV-Python-Program-to-analyze-an-image-using-Histogram-3.png' width=400>  

Nhìn bức hình ta có nhận xét: 
* Số lượng pixels ở vùng màu tối nhiều hơn vùng màu sáng
* Vùng màu ở giữa (giá trị pixel quanh 127) có ít  

Một số hình ảnh khác về đường tần suất

<img src = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR8aRtHDBRKIY4zVdXjedZPTmBeBTKu0Ak2Gw&usqp=CAU' width=400>

<img src = 'https://miro.medium.com/max/1582/1*OmPxmzT-ERYhLZpqv3t_GA.png' width=400>

```python
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
```
* images: Bức ảnh đen trắng ở dạng uint8 hoặc float32. Nên đưa ở dạng [img]  
* channels: Cũng đưa ở dạng [], trong ngoặc là thứ tự channel mà ta muốn tính histogram. Ví dụ, với ảnh đen trắng ta để [0], với ảnh màu ta để [0], [1]. [2] tương ứng với channel xanh nước biển, xanh lá hoặc đỏ.
* mask: Khi tìm histogram của cả bức ảnh, mask mặc định bằng None. Nhưng nếu ta chỉ muốn một phần bức ảnh, ta sẽ phải tạo một mask image.
* histSize: Đại diện cho số BIN, đưa ở dạng []. Cho toàn bộ khoảng giá trị pixel, ta chọn [256].
* ranges: Đây là RANGE, tức khoảng mức xám mà ta muốn quan sát. Bình thường là [0, 256].
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image (assume this is BGR image)
image = cv2.imread('beauty_3', 1)

# Just to flex
cv2.imshow(image)

# Convert to grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2

cv2.imshow(image)

hist = cv2.calcHist([gray], [0], None, [256], [0,256])

# With cv2.calcHist
plt.plot(hist)
plt.show()

# Doesn't need cv2.calcHist
plt.hist(gray.ravel(), 256, [0,256])
plt.show()
```  

<img src ='https://lh3.googleusercontent.com/NHPwLxjwGq5dNe8yQkwetS2lh97kBrl9l59aL-bh-8mvv3ikfiWTsHRU_3jSechXYe-ACsXfLRgj0hyyww=w219-h330-rw' alt = 'origin_image_4_histogram'>

<img src ='https://lh3.googleusercontent.com/QJxe6MpdeskgBOJf-yWuLC9tYggXVv0H32-Iej4yxFn4KFGqWPz50H4msziGoNtmQWovk3yD-_3NqyBFSQ=w219-h330-rw' alt = 'gray_origin_img_4_histg'>

> With cv2.calcHist
<img src = 'https://lh3.googleusercontent.com/vTUTcC0y5ujm3Rs_IZJ5U2736sC3WFTq-Ih_VGzaZJ52UZxFQl5gDphw_kkz0_xszCf4B0bkyBJiacHecg=w507-h330-rw'>


> Without cv2.calcHist    
<img src ='https://lh3.googleusercontent.com/BKz7APmWPx9c7m0LNNZTroba1ncjOoYI60xpkTVx8drSu2qPetYbG4SoiEzuzMU7FBp02_NATy5srD72hw=w507-h330-rw'>

## Histogram equalization  

Giải thuật cân bằng sáng:  
* Thống kê histogram cho ảnh: ![](https://latex.codecogs.com/png.image?\dpi{110}%20H(i))
*  Thống kê histogram cho ảnh:  ![](https://latex.codecogs.com/png.image?\dpi{110}%20H%27(i)=\sum_{j=0}^{i}H(j))
*  Chuẩn hóa histogram mới vừa biến đổi ![](https://latex.codecogs.com/png.image?\dpi{110}%20H%27(i)) về [0, 255]
*  Mapping mức sáng ảnh kết quả theo ![](https://latex.codecogs.com/png.image?\dpi{110}%20H%27:I%27(x,y)%20=%20H%27(I(x,y))) Với ![](https://latex.codecogs.com/png.image?\dpi{110}%20I(x,y)) là ảnh gốc, ![](https://latex.codecogs.com/png.image?\dpi{110}%20I%27(x,y)) là ảnh đã cân bằng sáng

Công thức này được dùng để tăng độ tương phản của bức ảnh bằng cách làm giãn giá trị tần suất cao nhất ra    

<img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/Histogrammeinebnung.png'>  

<img src='https://scikit-image.org/docs/0.5/_images/plot_equalize_1.png'>  

```python
import cv2 
import numpy as np

img = cv2.imread('beauty_3.jpg', 1)

# Just to flex
cv2.imshow(img)
# Convert to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
cv2.imshow('Original image', gray)
cv2.imshow('Equalized Image', dst)

cv2.waitKey()
cv2.destroyAllWindows()
```
![](https://lh3.googleusercontent.com/dk4Ej9EsxpcCvakNQm_Ka1W5copCpAlaXh0XEqxihyabU2fPBzLf7eAEV9jWtKF_IuVIDo7e_9M_vP2FZA=w221-h330-rw)

<table>
    <tbody>
	    <tr>
    	    <th style='text-align:center'>
        	    Original Image</th>  
            <th style='text-align:center'>
                Equalized Image</th>
        </tr>
        <tr>
    	    <td style='text-align:center'>
                <img src='https://lh3.googleusercontent.com/net4F2E5jhxPrfgG8WpobKjpFQkekRVdSnDJinK9vv1hCZxYiaXn-Ac1G9Ae6oT3X8kJPa_19AD4Z7Bd0g=w221-h330-rw'>
                <img src='https://lh3.googleusercontent.com/V4t8ciaD8kcTJ0vaYxK9gGKV5mbrYQkok1tIUdvO0xRwyddgS4-Ib7FmtAg5ghlXb4XykP18ZZDkxu_eKA=w515-h330-rw'></td>
            <td style='text-align:center'>
                <img src='https://lh3.googleusercontent.com/ZuieZH95BBA9SMhclf28lmC6TIOnstEwBAWcJIJ7eEMnQL5Wt3IJvbU5hl8xmt1cb5kgs0F7GOhiNQS5YA=w221-h330-rw'>
                <img src='https://lh3.googleusercontent.com/2FytnO2CxdKJrTsK4UBQPPdrs3UED-43tZjdRJbhBAw8Q0958eHggE4nUJR8dwxHloJgnzUBnrs5Jt_yfg=w515-h330-rw'></td>
        </tr>
    </tbody>
</table>

&nbsp;   
&nbsp;  
&nbsp;  
&nbsp;  
____________
# ***Variance***
## Định nghĩa: Là trung bình của bình phương độ lệch của giá trị MEAN
*Formula*
* Tính MEAN (trung bình cộng của dãy số)
* Bình phương từng giá trị MEAN, ta có một dãy số mới.
* Tính trung bình của dãy số mới đó, ta có Variance
[Source](https://www.mathsisfun.com/data/standard-deviation.html)  
Giả sử ta vừa tính chiều cao của các con chó trong một ngôi nhà
![dogs](https://www.mathsisfun.com/data/images/statistics-dogs-graph.gif)  
Chiều cao mỗi con lần lượt là 600, 470, 170, 430 và 300.  
Tính trung bình Mean, ta có Mean = 394  
![mean](https://www.mathsisfun.com/data/images/statistics-dogs-mean.gif)  
Chúng ta tính độ lệch của chiều cao mỗi con chó so với giá trị Mean  
![diff](https://www.mathsisfun.com/data/images/statistics-dogs-deviation.gif)  
Rồi bình phương từng giá trị, ta có dãy số mới:  
[42436, 5776, 50176, 1296, 8836]  
Tiếp tục tính Mean của dãy số này, ta có được Variance của đề bài = 21704
**Standard Deviation** ![sigma](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma)
Bằng căn bậc hai của Variance  
Như trong ví dụ trên, ![equat](https://latex.codecogs.com/png.image?\dpi{110}%20\sigma%20=%20\sqrt{21704}%20=%20147,32..%20\approx%20147)  
![](https://www.mathsisfun.com/data/images/statistics-standard-deviation.gif)  
Từ hình trên, từ việc sử dụng Standard Deviation ta đã có cái nhìn về chiều cao trung bình của loài chó (con đầu cao, 3 con cao trung bình, con thứ 3 lùn..)  

&nbsp;  
Trong xử lí ảnh, variance, standard deviation cùng với histogram là cơ sở cho các bài toán phân đoạn ảnh phức tạp, dựa trên việc lấy edge và xác định region trong bức ảnh.
[Otsu's threshold](https://nerophung.github.io/2019/09/26/otsu-threshold)   

```python  

# variance
def calculate_mean(numbers):                        #1
    s = sum(numbers)
    N = len(numbers)
    mean = s/N
    return mean
    
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

**Application:** Về cơ bản, Variance(~Standard deviation) được sử dụng để tìm  **texture** (kết cấu) của một bức ảnh

## Implementation:

```python
import numpy as np
import cv2
import math
from scipy.ndimage.filters import generic_filter
import glob

img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/file/edge_s1.jpg', gray)

x = gray.astype('float')
x_flit = generic_filter(x, np.std, size = 7)
cv2.imwrite('/file/edge_s2.jpg', x_filt)

x_filt[x_filt < 20] = 0
cv2.imwrite('/file/edge_s3.jpg', x_filt)

maxv = np.max(x_filt)
print(maxv)

x_filt = x_filt*2.5
cv2.imwrite('/file/edge_s4.jpg', x_filt)

for i in glob.glob('/file/*.jpg'):
    edge = cv2.imread(i)
    cv2.imshow(edge)
```
Original image:  
<img src='https://lh3.googleusercontent.com/QjgMy7VoTtLAZIRhV-yBiPMiACY3fEvL-9yE19d8USj1TNy6bcyPfKaely7QJecUqWAw-34Mn-aZM0ScXw=w221-h330-rw'>
<html>
    <table>
        <tr>
            <th style='text-align:center'>edge_s1</th>
            <th style='text-align:center'>edge_s2</th>
            <th style='text-align:center'>edge_s3</th>
            <th style='text-align:center'>edge_s4</th>
            <th style='text-align:center'>maxv</th>
        </tr>
        <tr>
            <td style='text-align:center'><img src='https://lh3.googleusercontent.com/eUtrxwWWAqaiqejbsWEW6cGfiEIIcBFFpUZy5SL9kdwBr5O7hbf-KGuTXzk32T_BgEEEsq4ab4SBDp_KnQ=w221-h330-rw'></td>
            <td style='text-align:center'><img src='https://lh3.googleusercontent.com/nMaE5egpouCYveUVcN6Duzo8eD3QR9ekgQBq1GIeVM_6dBWdLvbvtPXsAi-5rQJAV_M1hY6FX3tVqWREkw=w221-h330-rw'></td>
            <td style='text-align:center'><img src='https://lh3.googleusercontent.com/53_6ANrzJVE4nTjp7ojM2G3N0ImtccEP-aeudKYf3iE1uzjE2ukgKh4LZFYpTy2CB_9EhT94Q3CV_jxGUw=w221-h330-rw'></td>
            <td style='text-aalign:center'><img src='https://lh3.googleusercontent.com/em-UGUMXiqR9-3YBVXx53feTodbSGsLV-LZiRinMZnEit0mvNX52MnoV5oHlVbE45w7wVUc2pgotY5nMYQ=w221-h330-rw'></td>
            <td style='text-align:center'><img src='https://lh3.googleusercontent.com/2lEGYfKqpcpS81eWBYHDAkCaN7nd9HJYA1Ht8hRtMuZDvvyp3XWBX5kyHsMbZRpwYjvuUxiaU5c_DT_-vA=w350-h33-rw'></td>
    </table>
</html>

&nbsp;  
&nbsp;  
&nbsp;  
____________
# Correlation Coefficient
## Definition  
**Correlation coefficients** (hệ số tương quan) được sử dụng để đo lường mức độ mạnh yếu của mối quan hệ giữa hai biến số  
Hệ số tương quan có giá trị từ -1.0 đến 1.0. Kết quả được tính ra lớn hơn 1.0 hoặc nhỏ hơn -1 có nghĩa là có lỗi trong phép đo tương quan.

- Hệ số tương quan có giá trị âm cho thấy hai biến có mối quan hệ nghịch biến hoặc tương quan âm (nghịch biến tuyệt đối khi giá trị bằng -1)

- Hệ số tương quan có giá trị dương cho thấy mối quan hệ đồng biến hoặc tương quan dương (đồng biến tuyệt đối khi giá trị bằng 1)

- Tương quan bằng 0 cho hai biến độc lập với nhau

Cách tính hệ số tương quan Pearson
Có nhiều loại hệ số tương quan, nhưng loại phổ biến nhất là tương quan Pearson. Chỉ số này đo lường sức mạnh và mối quan hệ tuyến tính giữa hai biến. Nó không thể đo lường các mối quan hệ phi tuyến giữa hai biến và không thể phân biệt giữa các biến phụ thuộc và biến độc lập.


![](https://d138zd1ktt9iqe.cloudfront.net/media/seo_landing_files/diksha-q-how-to-calculate-correlation-coefficient-01-1609233340.png)
## Formula: 
![](https://www.statisticshowto.com/wp-content/uploads/2012/10/pearson.gif)  

**Subtitule:** x, y are 2 random variables

**Example 1**  
<img src='https://lh3.googleusercontent.com/DIEA36dN3sVDo3j1-XGFLslMu0JXErBekGiQkxNnfFpA1WRu-DwUt691yG30XdjCfojN5fczJmaGQPyJBA=w606-h269-rw'>    

**Example 2**
<img src='https://lh3.googleusercontent.com/Wj92GM6ZD69HlJz5v0p6m2vPr2OkWCi8ZXcbag4_LCDLQONVj6v9XjslsABhClfp6EDM3YbVsv10ESwHfg=w633-h282-rw'>  


**Property 1**
<img src='https://lh3.googleusercontent.com/htDtZokBOoWlXdVShCX7qQf6okcCgMqS00LWrVtYD_s5xK9rhNtRnQ4xz22qYOnjFL5ol3qvy6vkS_x9qQ=w308-h60-rw'>  
<img src='https://lh3.googleusercontent.com/K-OtmY2kHxbwRHkj4kBq89BeftSRpBaLfYYqbWsV1rql_yM_bEDYvlUx60c4s-UIOSb1GxKas2UZTkroJg=w398-h330-rw'>  

**Property 2**  
<img src='https://lh3.googleusercontent.com/xRfKAecaG-LrWHqGTdVzUslZDMIlAcQSJn8nKrtEtdYyc8yBJKI73jU1l3-7w7D2fppN7HzFOW_Q0aN9rw=w377-h65-rw'>  
<img src='https://lh3.googleusercontent.com/E23ZHj0J-rloJAg2hElB0ktkER9F4GXXm0rZGivHqfrXM5XQYnu2XnTb1Gijz8ZyA3FQx3hvyyDrzh_a0Q=w728-h330-rw'>  


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
    denominator_term1 = n*x_square_sum - squared_sum_x
    denominator_term2 = n*y_square_sum - squared_sum_y
    denominator = (denominator_term1*denominator_term2)**0.5
    correlation = numerator/denominator

    return correlation
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%5E%7B%7D%20x_%7Bi%7Dy_%7Bi%7D'></p>
```python
prod = []
for xi, yi in zip(x,y):
    prod.append(xi*yi)
        
sum_prod_x_y = sum(prod)
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%5E%7B%7D%20x_%7Bi%7D'></p>
```python 
    sum_x = sum(x)
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%5E%7B%7D%20y_%7Bi%7D'></p>
```python  
    sum_y = sum(y)
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%28%5Csum_%7Bi%7D%5E%7B%7Dx_%7Bi%7D%29%5E%7B%5E%7B2%7D%7D'></p>
```python
squared_sum_x = sum_x**2
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%28%5Csum_%7Bi%7D%5E%7B%7Dy_%7Bi%7D%29%5E%7B%5E%7B2%7D%7D'></p>
```python
squared_sum_y = sum_x**2
```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%5E%7B%7Dx_i%5E%7B%5E%7B2%7D%7D'></p>
```python
x_square = []
    for xi in x:
        x_square.append(xi**2)
    x_square_sum = sum(x_square)

```

&nbsp;  
<p style='text-align:center'><img src='https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%7D%5E%7B%7Dy_i%5E%7B%5E%7B2%7D%7D'></p>
```python
y_square = []
    for yi in y:
        y_square.append(yi**2)
    y_square_sum = sum(y_square)
```
## Application:
### For patch maching:

```python
import numpy as np
from PIL import Image

# load ảnh và chuyển về kiểu list
image1 = Image.open('image/img1.png')
image2 = Image.open('image/img2.png')
image3 = Image.open('image/img3.png')
image4 = Image.open('image/img4.png')

image1_list = np.asarray(image1).flatten().tolist()
image2_list = np.asarray(image2).flatten().tolist()
image3_list = np.asarray(image3).flatten().tolist()
image4_list = np.asarray(image4).flatten().tolist()

# tính correlation coefficient
corr_1_2 = find_corr_x_y(image1_list, image2_list)
corr_1_3 = find_corr_x_y(image1_list, image3_list)
corr_1_4 = find_corr_x_y(image1_list, image4_list)

# Flex
image1.show()
image2.show()
image3.show()
image4.show()

print('corr_1_2:', corr_1_2)
print('corr_1_3:', corr_1_3)
print('corr_1_4:', corr_1_4)
```
<img src='https://lh3.googleusercontent.com/DZ9IzAhE-kvSZvbCgk5ro7hzE-4ztFApff5b95yCSog9sBTsv0hyumJyBOiTmdP-vLJUkIy8eoKSGf2Hrw=w221-h330-rw' height=300>
<img src='https://lh3.googleusercontent.com/mFHWaFTH9QFqlqET-MzoFVaN07wIDM4gLuri90Sx9Hhbcea52BAzjZHfIjwW-3hIXKgZAMEYJwTwjFbRvw=w221-h330-rw' height=300>
<img src='https://lh3.googleusercontent.com/zNKSVHrmJXJjKBGv3TK3_XAkWoSmrwWTCylI4bHs0_ge4qHik4BJENe3IZxMgb821tx8nUNjinNldkNgXQ=w221-h330-rw' height=300>
<img src='https://lh3.googleusercontent.com/XQTFZBKRme4f8NBzbHw3jB1DW3EHIy_2LOFLL2fEIz7arcW04XS4_v9YsLhD2NcAbCVqxfg4jnLvPmHV0w=w248-h330-rw' height=300>  

&nbsp;  
<img src='https://lh3.googleusercontent.com/D3JYrCmnbmbpxLlqUBQI2nmAs5xoHR4jBsU9hGAL1-H0DkeVCf32JS9EEwD7FYKvOM3hP0N-XSJaoVOewg=w999-h144-rw'>

Nhận xét từ giá trị corr
* Ảnh 1 với 2 không giống nhau chút nào
* Cặp ảnh 1 và 4 là giống nhau nhất trong 3 cặp ảnh mình so sánh

&nbsp;  
###For template matching
(image and code)
![](https://lh3.googleusercontent.com/UjhyDjj0Iw-wNuYwjikjvlTsrtEYEL2MDsOD_ZtxWXokSuoRoCLYIcHpuhqSeKqd2V_B2thVnsB99cl7DQ=w927-h330-rw)  
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

template = cv2.imread('teamplate.png', 0)
w, h = template.shape[::-1]

# Apply template matching
corr_map = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

# Save correlation map
corr_map = (corr_map + 1.0)*127.5
corr_map = corr_map.astype('uint8')
cv2.imwrite('corr_map_grayscale.png', corr_map)

img1 = cv2.imread('image.png')
img2 = cv2.imread('template.png')
img3 = cv2.imread('corr_map_grayscale.png)
cv2.imshow(img1)
cv2.imshow(img2)
cv2.imshow(img3)
```

<img src ='https://lh3.googleusercontent.com/SrfdD9_BtKuEzVDGFt0_ao-YeAkiT49d9NG8W_I3ArkrrRtsDfegeaNFeA_NKBjCHpz7iPJ93m0b3wCOZA=w219-h330-rw'>
<img src='https://lh3.googleusercontent.com/SZg-YJHUtUh7ijmot4W1dktyrcpbiizkjYB1mmOEM4rnPt2MqD9Ql6myr2uLugr2MJNbnZZzC4VCZryfIA=w191-h269-rw'>  
<img src='https://lh3.googleusercontent.com/sTzcRtetkpUWJB2qR1NGCfBV723n1ETZh_VB1-TM0RMgcO70leWb4R-doLrr-WLm_7wAze183NMOv8Yy1Q=w216-h330-rw'>
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image and convert to grayscale
image = cv2.imread('image.png', 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

template = cv2.imread('teamplate.png', 0)
w, h = template.shape[::-1]

# Apply template matching
corr_map = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr_map)

# take minimum
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# draw
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# save corr map in grayscale
corr_map = (corr_map+1.0)*127.5
corr_map = corr_map.astype('uint8')
cv2.imwrite('corr_map_grayscale.png', corr_map)

#applyColorMap
corr_map = cv2.applyColorMap(corr_map, cv2.COLORMAP_JET)

# save results
cv2.imwrite('corr_map_color.png', corr_map)
cv2.imwrite('result.png', image)
```

<img src='https://lh3.googleusercontent.com/sTzcRtetkpUWJB2qR1NGCfBV723n1ETZh_VB1-TM0RMgcO70leWb4R-doLrr-WLm_7wAze183NMOv8Yy1Q=w216-h330-rw'>  
<img src='https://lh3.googleusercontent.com/pIAagswKPn0335pGL-3afBlCxOPiQ5MCpKtPzDABtXY5eHgsjvjDNxswc3BOhgc1qXrryHDrVkQ1ZjTMAg=w216-h330-rw'>  
<img src='https://lh3.googleusercontent.com/vG9PIOcfBFuhFUYseEboeAz2LM6wKGfkD5izH7A7bxUhJRdrdINYA8Uv4Tb-RcOpmg7RvTzy4Zq3u-CMiw=w219-h330-rw'>  
