# Image Processing (MC920) - Convolution Filters

- [Requirements](papers/enunciado.pdf)
- [Report](papers/entrega.pdf)

This project is a simple sandbox for spatial convolution. A tiny CLI takes a grayscale PNG, applies one or more preset kernels, and shows or saves the output. The available catalogue spans eleven classic kernels:

- Gaussian and box blurs
- two Laplacian edge detectors
- Sobel $x$, $y$, and $mag$
- Prewitt
- two diagonal line detectors
- a 45° motion blur
- a simple sharpening mask

Below are three representative results on the familiar **butterfly** test image.

![Butterfly convoluted with a Sobel X](resultados/butterfly_h3.png "Sobel X filter")

![Butterfly convoluted with a Diagonal Line Detector](resultados/butterfly_h8.png "Diagonal Line Detector filter")

![Butterfly convoluted with a Prewitt filter](resultados/butterfly_h11.png "Prewitt 45° filter")
