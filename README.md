# Signature Extractor

(c) Vlad Zat 2017

# Introduction:
Extract the signature from an image by firstly extracting the page in it (if any)
then extracting a signature block, and in the end use thresholding to extract only the
signature in the correct colour. Allow the user to import an image from file or capture
from the camera. The result can be viewed on the screen or exported to file.

# Structure:
1. Extract the Page

   a. Convert the image to Grayscale
   
       * Using only one channel is necessary for both edge detection and segmentation
       
   b. Find the edges of the image using Canny
   
       * Edge detection is used to be able to find the main objects in the image
       
       * The range for Canny is approximated to be from half the treshhold to
         the value of the treshhold. This threshold is calculated using Otsu's Algorithm.
         This provides consident results and is recommended by Mei Fang (et al.) in
         "The Study on An Application of Otsu Method in Canny Operator" [1]
         
   c. Getting the contours of the objects in the image
   
   d. Finding the biggest contour with 4 edges
   
       * The perimeter of the contour is calculated and then used to approximate a
         polygon around it. To decrese the ammound of edges detected, the permited
         is multiplied with 0.1 as recommended in the OpenCV Documentation [2]
         
   e. Detecting if the biggest contour has any contours in it
   
       * If the biggest contour does not have any other contours in it (such as words
         or the signature) then it's a false alarm and there is no complete page in the image
         so the whole image is used in the next step
         
2. Extracting the Signature

   a. Convert the image to Grayscale
   
   b. Find the edges of the image using Canny
   
   c. Getting the contours of the objects in the image
   
   d. Closing the image so the signatures is more filled and pronounced
   
   e. Finding the contour with the most number of edges
   
       * Opposed to the method from the first step, here the permimeter of the
         signature is multipled with 0.01 to give it a better shape of the signature [2]
         
       * Signatures have more edges than normal text in a page so the contour with
         the maximum ammount of edges should be the signature
         
   f. Pad the image so the signature can be viewed more easily
   
3. Remove the background from the signature

   a. Convert the image to Grayscale
   
   b. Use adaptive thresholding to extract only the signature
   
       * A block of 1/8 of the width of the image is used as it provides
         consistent results
         
       * If the image is too small then the width of the image is used instead

# Experiments:
   * Blur the image using Median Blur and Bilateral Blur to maintain the edges
     but smooth the objects. It requires hardcoded values for the kernel size used
     and it doesn't provide good enough results
     
   * Use thresholding to extract the page. While it does provide similar results to
     the edge detection method, it requires hard coded values
     
   * Try to use Otsu's Algorithm for the final step. It does not provide good results,
     even for small parts of the image
     
   * Use Hough Line to get the lines in the image and extracting zones where they are
     intersecting. It does not detect lines in the images
     
   * Use a feature detection algorithm such as AKAZE to get the the features of the image.
     It doesn't provide better results than using Canny
     
   * Use different colour spaces to better extract the paper. The luminance or saturation
     provide similar results to grayscale
     
   * Make the image smaller if the image is too big as the signature extraction does not
     work properly with large images. It causes problems for some images.

   References:
   
       [1] M.Fang, GX.Yue1, QC.Yu, 'The Study on An Application of Otsu Method in Canny Operator',
           International Symposium on Information Processing, Huangshan, P. R. China,
           August 21-23, 2009, pp. 109-112
           
       [2] OpenCV, 'Contour Approximation', 2015. [Online].
           Available: http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
           [Accessed: 2017-10-05]
