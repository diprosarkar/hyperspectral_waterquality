# hyperspectral_waterquality
This code is designed to create a spatial map that displays varying concentrations of pollutants from in-situ data. The code is to be supplied by 3 types of dataset:
1. Concentration of pollutants from Water sample
2. Reflectivity data from Spectroradiometer
3. Hyperspectral imagecubes from UAV

The codeblock takes the concentration values and the reflectivity values, interpolate the intermediate reflectivity values from the spectroradiometer data and matches the bads of the hyperspectral cube, then trains and generate an AI model. The AI model is then used to generate a spatial map of the respective pollutants. 
