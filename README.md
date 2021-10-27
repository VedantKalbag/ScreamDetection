# Scream Detection and Classification for Heavy Metal vocals

The objective of this project is to label blocks of audio based on the detected vocal content, and classify it based on the type of scream (or growl) present

Work has been divided into the following stages:
- Data fetch and cleaning 
  - Dataset is built off youtube audio releases from the artist's official channels
- Data annotation (done using SonicVisualizer)
- Preprocessing:
  - Apply Spleeter source separation to extract vocal track
  - Extract VGGish features
- Feature Extraction:
  - Zero Crossing Rate
  - Spectral Centroid
  - Spectral Contrast
  - Spectral Flatness
  - Spectral Roll-off
- kNN classifier using MFCC and delta MFCCs
- SVM and RF classifiers using the extracted features listed above features along with MFCCs and delta MFCCs

Next steps and alternate ideas:
- Apply image processing to the spectrogram to classify the vocal type, since there are clear patterns visible
