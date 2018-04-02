Convolutional neural network

The task is to recognize text font.
Under developing...
-mini batch learning;
-stabilize network
-providing custom path for generating data set 
-compatible with windows
-custom input for testing

The network is ready to 'memorize' relatively small input (about 30 images) with
 100% accuracy, however unstable with validation data tests.

Before generating the data, a text file with the names of fonts needs to be provided to work directory.
My custom text file consists of 10 more less distinctive text fonts:

Andale Mono,Apple Chancery, Arial, Chalkduster, Courier New, Brush Script, Luminari, Times New Roman,Trebuchet MS Italic, Trebuchet

To generate the data run:
- python3 data_generator.py <mode> <numb of instances per class>
For this moment valid only for macOS

Two data sets need to be created with mode: <training> and <test>, number of instances is arbitrary.
 
