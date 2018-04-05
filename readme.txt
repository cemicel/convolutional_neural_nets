Convolutional neural network

The task is to recognize text font.

Under developing:

-compatible with windows
-bug fix
-impruve interface


The network is ready to 'memorize' relatively small input (about 30 images) with
100% accuracy, however accuracy can vary within 10-15% for same epochs working on validation data tests.


Manual:
1)Provide text file with font names separated with commas.
In repo there is my custom text file consists of 10 more less distinctive text fonts.
2)Run in terminal: python3 generate_data.py <mode> <numb of instances per class>
mode - training, test
3) Run in terminal: python3 conv_net.py <mode> <batch_size> <epochs>
	1)First train mode
	2)For now, batch size is number of pictures per class.
	It means if in a class 50 pics, and there are 10 classes 
	that batch of 5 will be 50 * 10 = 50 out of 450
	
	3)Two type of test available:
	a) ‘test_custom’ + space + type custom font from given list
	example: conv_net.py test_custom Arial
	
	b) ‘test’ - tests validation data
Some times, macOS creates .DS_Store system hidden files. 
befor test network training/dest directoris need to be cheked with ls -a command. If you notice this file, delete it.
For this moment valid only for macOS
