# live-cam-video-filter

Apply basic filters to video in real time.

# features and GIFs

<details>

<summary><b>Edge Detection</b>: Variations of Canny, Sobel, and Laplacian </b>(click to see example)</b><br></summary>

<img src="screen_shots/edge_detect.gif" width="66%">

</details>
<br>
<details>

<summary><b>Face Detection</b>: Uses face and eye detection to apply wearable mask filters that adjust size and position to cover the target.<br></summary>

<img src="screen_shots/face_detect.gif" width="66%">

</details>
<br>
<details>

<summary><b>Color Filters</b>: Hue Saturation Value (HSV), L\*a\*b\*, Gray Scale, Red, Green, and Blue<br></summary>

<img src="screen_shots/color.gif" width="66%">

</details>
<br>
<details>

<summary><b>Blur</b>: Averages neighboring pixels to blur the image. Kernel size is adjustable.<br></summary>

<img src="screen_shots/blur.gif" width="66%">

</details>
<br>
<details>

<summary><b>Foreground Scenery</b>: Add opaque or transparent foreground images to the video.</summary>

<img src="screen_shots/foreground.gif" width="66%">

</details>
<br>
<details>

<summary><b>Geometric Transformations</b>: Translate the video along the x or y axis, rotate the video about it's center or increase / decrease the scale of the video.</summary>

<img src="screen_shots/geometric_transform.gif" width="66%">

</details>
<br>

# controls

<b>ctrl + c</b>: copy current frame to clipboard<br>
<b>ctrl + s</b>: save current frame<br>
<b>q</b>: quit<br>
<b>f</b>: on / off face detection<br>
<b>h</b>: Cycle the thickness of the face detection ring (none, 1, 3, 5, solid)<br>
<b>g</b>: on / off for foreground opacity<br>
<b>r</b>: resets all modifications to initial state<br>
<b>left / right-arrow</b>: Change between features<br>
<b>up / down-arrow</b>: Change the current feature's settings<br>

# stackable filters

The filters are applied sequentially to the video frame such that filteres of different types can be used together.<br><br>

For example, an eye mask filter that uses face detection can be used at the same time as an image rotation and a Canny edge detection filter are applied:

<br>
<p align="center">
<img src="screen_shots/rotation_canny_mask.png" width="66%">
</p>
<br>

Or a color filter can be applied on top of a Laplacian edge detection filter with moderate image blur:

<br>
<p align="center">
<img src="screen_shots/laplacian_blur_green.png" width="66%">
</p>
<br>

# credits

Please read the LICENSE file as well as the Intel licence located in the header of video_editor.py

The latter is included because the haar_cascade face and eye classifiers were obtained here: https://github.com/opencv/opencv/tree/master/data

All images were labeled free for personal use and the original artists are listed in images/artist_credits.txt.