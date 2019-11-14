# dialectautoencoder
Performs dialect analysis using autoencoders. Generates output files that can be ported to GabMap or visualized using SplitsTree.

## Please consider citing the following paper if you use the code:
Taraka Rama and Çağrı Çöltekin. LSTM Autoencoders for Dialect Analysis. Proceedings of Third Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial3), COLING 2016, Osaka, Japan, 2016.	http://www.aclweb.org/anthology/W/W16/W16-48.pdf#page=37

## Requirements:
  - Need Keras (https://keras.io/) with Tensorflow (https://www.tensorflow.org/) as  a backend for running the code.

## Running the program:
  - The main program is in src folder as ae.py
  - Run the program as ```python3 src/ae.py data/Germany40.utf8```
  - The program generates a nexus file (.nex extension) and a vectors file in the   data folder
  
## Contact:
In case of any questions, contact taraka-rama.kasicheyanula@uni-tuebingen.de
