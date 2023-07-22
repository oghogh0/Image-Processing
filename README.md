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

This function uses the seam carving technique to remove n columns from an image. The goal is to scale down an image while preserving the perceptually important parts, such as removing the background but preserving the subjects.<br/>

Here are some helper functions involved in this process:<br/>
1. greyscale_image_from_color_image: computes and returns a corresponding greyscale image of a coloured image. A colour pixel's equivalent greyscale value ‘v’ is computed using v = round(.299*r  +  .587*g  +  .114*b) <br/>
2. compute_energy: computes a measure of "energy" of a greyscale image using the edges function.<br/>
3. cumulative_energy_map: computes a "cumulative energy map" from the measure of energy. <br/>
4. minimum_energy_seam: returns a list of the indices that correspond to pixels contained in the minimum-energy seam, given a cumulative energy map. This is found by backtracing from the bottom to the top of the cumulative energy map. The minimum value pixel in the bottom row of the cumulative energy map is the bottom pixel of the minimum seam. Then, the seam is traced back up to the top row of the cumulative energy map by following the adjacent pixels with the smallest cumulative energies.<br/>
5. image_without_seam: return a new image containing all the pixels from the original image except those corresponding to the locations
in the given list of indices of a coloured image. In other words, it removes the computed path.<br/>

Consider this image:<br/>
<img src= "https://imgur.com/hbXBREZ.png" height="40%" width="40%"/>

After seam carving:<br/>
<img src= "https://imgur.com/b3iHOkS.png" height="40%" width="40%"/>

<h2>Program walk-through PART 3</h2>
<p align="left">
Create CUSTOM feature:<br/>
This is a custom feature I made that draws multiple circles with varying radius above a right-angle triangle<br/>
<img src= "https://imgur.com/HYW6yap.png" height="40%" width="40%"/>

<img src= "https://imgur.com/CboRT4n.png" height="40%" width="40%"/>

<h2>More Images</h2>
<img src= "https://imgur.com/3Gcmshq.png" height="40%" width="40%"/>

<img src= "https://imgur.com/8rr9wz4.png" height="40%" width="40%"/>

<img src= "https://imgur.com/19U9OOj.png" height="40%" width="40%"/>

<img src= "https://imgur.com/uby5Zjk.png" height="40%" width="40%"/>
