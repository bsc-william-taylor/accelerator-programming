
<img src='preview.gif'  />

<img src='icon.png' height='150' width='150' align='right' />

# GPGPU and Accelerator Programming

The GPGPU and Accelerator programming module focused on GPU APIs and parallel compute available through GPUs. It was assessed in two parts both of which were programming assignments. This was a really fun module as it was a crash course in GPUs and GPU programming which was great to fill in gaps on any pre-existing knowledge. It also covered new subjects in this space so there was much to learn as well.

## Assignment

So as mentioned the assignment was in two parts.

#### Part 1

Part one was to output a set of complex numbers from the Mandelbrot set. We were given some basic C++ code that did this and were told to port it to CUDA to make it more efficient. We then had to write a report setting out the improvements made. We were also expected to record data timings as well for referencing in the report.

#### Part 2

Part two was to perform an unsharp filter on a given image. Code which accomplished this was given and we were meant to port it to a GPU API, any GPU API apart from CUDA. However in this part we were given a list of additional tasks we could implement to gain extra marks these were:

* Utilize shared/local memory as a cache for data-reuse  
* Exploit ﬁxed-function graphics hardware via OpenCL images and bilinear ﬁltering 
* Interactive graphical visualization of varying blur radii

## Submission

I met all the requirements for both assignments and received really high marks as well.

#### Part 1
 
I managed to complete part 1 rather easily with a 5x speed improvement with a 16k image over the previous version. I also added unit tests to test for an identical output to the original code. Note when benchmarking I benchmarked the entire program so GPU memory transfers were considered. If I were to remove the memory transfers and the writing of the image to disk from the program the speed up would have been even larger. 

#### Part 2

For part 2 I implemented the assignment using OpenCL as unlike CUDA I could run it on any computer and not just NVidia approved hardware. In additional to implementing the unsharp mask with a 1100x speed up with a blur radius of 35 I also developed two additional applications. A PPM viewer which allowed me to view the output of the program. And a Visualizer which allowed me to view the changes in real time when altering blur radii this was done through texture sharing between OpenCL and OpenGL. I also met the fixed function graphics hardware requirement by using OpenCL images rather than OpenCL vector types for data caching and filtering options.

## License 

Apache 2.0