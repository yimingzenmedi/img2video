# Links

** Due to an update to the hosting platform, the sharing link cannot be created at the moment. Will update later. **


# Training

To train the model, download the datasets and unzip to ./Dataset.
Run train_center.py to train the model for center frame.
Run train_F35.py to train the model for the 3rd and 5th frame.
Run train_F26_F17.py to train the model for the 1st, 2nd, 4th and 7th frame.


# Testing

To test on image, put the blurred image into ./test_dir, an run test.py.
Parameter "index" is required to specify which frame to generate.
