Convolutional neural network

The task is to recognize text font.

Under developing:

-create custom input for testing
-providing custom path for generating data set 
-compatible with windows
-bug fix


The network is ready to 'memorize' relatively small input (about 30 images) with
100% accuracy, however accuracy can vary within 10-15% for same epochs working on validation data tests.


Manual:
1)Provide text file with font names separated with commas.
In repo there is my custom text file consists of 10 more less distinctive text fonts.
2)Run in terminal: python3 generate_data.py <mode> <numb of instances per class>
mode - training, test
3) Run in terminal: python3 conv_net.py <mode> <batch_size> <epochs>
	1)First train mode
	2)For now, batch size is number of picturer per class.
	It means if in a class 50 pics, and there are 10 classess 
	that batch of 5 will be 50 * 10 = 50 out of 450
	
	3)test mode in pogress.
, then test  
For this moment valid only for macOS
