# DDWP-DA
Integrating data assimilation with deep learning. Find details in this [paper](https://gmd.copernicus.org/preprints/gmd-2021-71/) \
Background forecast model is U-STNx. Model training is performed through the jupyter notebook. \ 
Key points: 
1. Replace Convolution2D with CConv2D custom function if circular convolution is needed. No major performace improvement 
2. Ensure training and autoregressive prediction uses same convolution function. 
3. U-STN1 +SPEnKF for regular 24 hrs DA and 1hr forecast is given in EnKF_DD_all_time.py
4. U-STN1 + SPEnKF with virtual observations from U-STN12 is given in EnKF_DD_all_time_2DAv2.py
5. weights and biases for U-STN12 and U-STN1 is provided in the repository. 
6. Change value of "lead" and run training for any "x" in U-STNx or U-NETx
7. Baseline U-NET model can be trained (and tested autoregressively) with Unet_noSTN.py

Finally, the SPEnKF algorithm implementation is inspired from Tyrus Berry's presentation [here](http://math.gmu.edu/~berry/Presentations/KalmanPractical.pdf).
For any questions , please reach out to me at akc6@rice.edu
