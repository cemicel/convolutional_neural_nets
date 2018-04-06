Convolutional neural network

The task is to recognize text font.

Under developing:

-bug fix
-impruve interface


The network is ready to 'memorize' relatively small input (about 30 images) with
100% accuracy, however accuracy can vary within 10-15% for same epochs working on validation data tests.

After training the network, in work directory ‘save’ directory is created with network adjustments. After you can test the network.


Manual:
1)Provide directory with fonts (.ttf);
In repo there is my custom directory of 10, more less distinctive text fonts.

2)Run in terminal: python3 generate_data.py <mode> <numb of instances per class>
mode - training, test
IF there is no parameters passed, training and test data sets will be created with 20 and 1  respectively picture/s per class.

3) Run in terminal: python3 conv_net.py <mode> <batch_size> <epochs>
	<mode> - training/test_custom/test
	
	1)First train mode

	a)For now, batch size is number of pictures per class.
	It means if there are 50 pics in a class, and there are 10 classes 
	then batch of 5 will be 50 * 10 = 50 out of 450
	
	b)test_custom - checks given font after mode;
	Example:
	-test_custom Arial
	After, You will be asked to provide a string to recognze or press enter, 	and word 'textToRecognize' will be checked. 
		

	3) ‘test’ - tests validation data

	4)If all parameters are empty, training mode is set with batch: 5 and epochs: 50

Some times, macOS creates .DS_Store system hidden files. 
befor test network training/dest directoris need to be cheked with ls -a command. If you notice this file, delete it.

