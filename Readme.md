First pair of G/D = 5 layer
DISCRIMONATOR
  self.main = nn.Sequential(
            # First layer: a convolution that takes an image (3 channels) as input and outputs feature maps of size 64 x 32 x 32
            nn.Conv2d(3, 64, kernel = 4, stide = 2, padding = 1, bias=False),
            # LeakyReLU activation allows a small gradient when the unit is not active
            nn.LeakyReLU(0.2, inplace=True),
            # Size after this layer: (64) x 32 x 32

            # Second layer: another convolution that reduces the spatial dimensions by half and increases depth
            nn.Conv2d(64, 128, kernel = 4, stide = 2, padding = 1, bias=False),
            # Batch normalization to stabilize learning by normalizing the output of the previous layer
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Size after this layer: (128) x 16 x 16

            # Third layer: similar to the second, further reducing spatial dimensions and increasing depth
            nn.Conv2d(128, 256, kernel = 4, stide = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Size after this layer: (256) x 8 x 8

            # Fourth layer: continues the pattern of halving spatial dimensions and increasing depth
            nn.Conv2d(256, 512, kernel = 4, stide = 2, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Size after this layer: (512) x 4 x 4

            # Final layer: a convolution that reduces the feature maps to a single value, the discriminator's prediction
            nn.Conv2d(512, 1, kernel = 4, stide = 2, padding = 1, bias=False),
            # Sigmoid activation function to output a probability between 0 and 1
            nn.Sigmoid()
            # Final output size: (1), representing the probability that the input image is real
        )

GENERATOR
self.main = nn.Sequential(
            # First layer: a transposed convolution that takes the latent vector as input and outputs feature maps of size 512 x 4 x 4
            nn.ConvTranspose2d(100, 512, kernel = 4, stide = 1, padding = 0, bias=False),
            # Batch normalization to stabilize learning by normalizing the output of the previous layer
            nn.BatchNorm2d(512),
            # ReLU activation to introduce non-linearity into the model
            nn.ReLU(True),

            # Second layer: another transposed convolution that doubles the width and height dimensions of the feature maps
            nn.ConvTranspose2d(512, 256, kernel = 4, stide = 2, padding = 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Third layer: similar to the second, it further doubles the dimensions and reduces the depth
            nn.ConvTranspose2d(256, 128, kernel = 4, stide = 2, padding = 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Size after this layer: (128) x 16 x 16

            # Fourth layer: continues the pattern, doubling dimensions and reducing depth
            nn.ConvTranspose2d(128, 64, kernel = 4, stide = 2, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Size after this layer: (64) x 32 x 32

            # Output layer: final transposed convolution that outputs the 3-channel image of the target size 64 x 64
            nn.ConvTranspose2d(64, 3, kernel = 4, stide = 2, padding = 1, bias=False),
            # Tanh activation function to scale the output pixel values to the range [-1, 1]
            nn.Tanh()
            # Final output size: (3) x 64 x 64
        )


Second pair of D/G = 6 layer
DISCRIMONATOR

self.main = nn.Sequential(
            # First layer: a convolution that takes an image (3 channels) as input and outputs feature maps of size 64 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Second layer: another convolution that reduces the spatial dimensions by half and increases depth
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Third layer: similar to the second, further reducing spatial dimensions and increasing depth
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Additional layers for more complex feature extraction
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # State size. (1024) x 2 x 2

            # Final layer to produce a single scalar output (probability)
            nn.Conv2d(1024, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
            # Final output size: (1), representing the probability that the input image is real
        )

GENERATOR
      self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # Second layer: another transposed convolution that doubles the width and height dimensions of the feature maps
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Third layer: similar to the second, it further doubles the dimensions and reduces the depth
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Fourth layer: continues the pattern, doubling dimensions and reducing depth
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Additional layer to refine the features before the final layer
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Output layer, reducing the feature depth to match the image channels (e.g., 3 for RGB)
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh()
            # Final output size. (3) x 64 x 64
        )


Thirf pair of d/G = 3 LAYERS
Discriminator
self.main = nn.Sequential(
            # Input layer: taking in 3-channel images.
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (64) x 16 x 16

            # Middle layer: increasing depth while reducing size.
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (128) x 8 x 8

            # Final layer: producing a single scalar output (probability).
            nn.Conv2d(128, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
            # Final output size: (1)
        ) 
Generator
self.main = nn.Sequential(
            # Input is the latent vector Z.
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: (256) x 4 x 4

            # Middle layer, expanding the size.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: (128) x 8 x 8

            # Output layer, producing the image.
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Final image size: (3) x 16 x 16
        )


