<h1>Image Processing Lab</h1>
<h2>Description</h2>
In this lab, <br />

<h2>Languages and Environments Used</h2>

- <b>Python</b> 
- <b>VS code</b>

<h2>Program walk-through PART 1</h2>

<p align="left">
Create useful FUNCTIONS:<br/>

1. get_pixel: returns the value at an index of the image<br/>
2. set_pixel: changes the colour of a pixel at an index of the image<br/>
3. apply_per_pixel: applies the input function to each pixel<br/>
4. round_and_clip_image: ensures that the values in the pixels list are integers in range [0, 255]. Values > 255 should have value 255, and < 0 should have value 0.<br/>
5. box: returns an nxn box kernel of identical values sum to 1

<p align="left">
Create useful FEATURES:<br/>
1. Inverted: inverts the image <br/>

Consider this image:<br/>
<img src= "https://imgur.com/BBa8uLg.png" height="50%" width="50%"/>

After inversion:<br/>
<img src= "https://imgur.com/BUWNojR.png" height="50%" width="50%"/>

2. Correlate: computes the result of correlating an image with the kernel given, treating pixels that are out of the image bounds as having value 0, nearest edge value, or wrapped around the edge value. <br/>

Consider this image:<br/>
<img src= "https://imgur.com/bdbZwuy.png" height="50%" width="50%"/>

After "zero" correlation:<br/>
<img src= "https://imgur.com/CD6QVjO.png" height="50%" width="50%"/>

After "extend" correlation:<br/>
<img src= "https://imgur.com/2OSUUfy.png" height="50%" width="50%"/>

After "wrap" correlation:<br/>
<img src= "https://imgur.com/JMMGsVp.png" height="50%" width="50%"/>



<p align="left">
Create FILTERS:<br/>
1. Blur: blurs the image <br/>

Consider this image:<br/>
<img src= "https://imgur.com/O43pWQN.png" height="40%" width="40%"/>

After applying the 'blur' filter:<br/>
<img src= "https://imgur.com/iNF3Qg3.png" height="40%" width="40%"/>

2. Sharpen: sharpens image using formula s=2*I-B, where I is the image and B is blur.<br/>

Consider this image:<br/>
<img src= "https://imgur.com/NbNkOjF.png" height="40%" width="40%"/>

After applying the 'sharpen' filter:<br/>
<img src= "https://imgur.com/zNVVapb.png" height="40%" width="40%"/>

3. Edges: returns new image with each corresponding index with Sobel operator. <br/>

Consider this image:<br/>
<img src= "https://imgur.com/PXXTAXs.png" height="40%" width="40%"/>

After applying the 'edges' filter:<br/>
<img src= "https://imgur.com/CjP324u.png" height="40%" width="40%"/>


<h2>Program walk-through PART 2</h2>
<p align="left">
More useful FILTERS & FUNCTIONS:<br/>

1. split_color: splits image into red, green, blue<br/>
2. recombine_greyscale: returns a new image with pixels of all input images combined<br/>
3. filter_cascade: returns a new single filter such that applying that filter to an image produces the same
output as applying each of the individual input filters in list in turn.<br/>

Example:<br/>
<img src= "https://imgur.com/byfEJsc.png" height="40%" width="40%"/>

<p align="left">
Implement SEAM CARVING:<br/>

This function uses the seam carving technique to remove n columns from an image.<br/>
Here are some helper functions involved in this process:<br/>
1. 
