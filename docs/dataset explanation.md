# Dataset Explanation

## anatomy_segmentation_labels

There are two files: `train_label_summary.xlsx` and `valid_label_summary.xlsx`

These two files contain anatomical structure segmentation label data for the training set (`train`) and validation set (`valid`), respectively.

| Column Name   | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| `Patient ID`  | Patient identifier.                                          |
| Other columns | Each column represents an anatomical structure. The value indicates whether the structure is present in the CT scan. |

Value meanings:

- `1.0`: The structure is present in the scan.
- `NaN`: The structure is not labeled. This may mean it is outside the scan range or hasn't been annotated yet.
- `True/False`: For some structures, boolean values are used to indicate presence.

| English Label                  | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `Spleen`                       | One of the major immune organs                |
| `Kidney R / Kidney L`          | Kidneys are responsible for filtering blood   |
| `Gallbladder`                  | Organ that stores bile                        |
| `Liver`                        | The largest internal organ, metabolism center |
| `Stomach`                      | Initial part of food digestion                |
| `Aorta`                        | Main artery in systemic circulation           |
| `Inferior vena cava`           | Main vein returning blood to the heart        |
| `Portal vein and splenic vein` | Blood return channels from digestive organs   |
| `Trachea`                      | Respiratory pathway                           |
| `Lung R / Lung L`              | Site of gas exchange in respiration           |
| `Esophagus`                    | Connects throat to stomach                    |
| `Heart`                        | Organ that pumps blood                        |
| `Subcutaneous tissue`          | Tissue layer outside muscles and bones        |
| `Muscle`                       | Part of the muscular system                   |
| `Bones`                        | Supporting structures                         |
| `Abdominal cavity`             | Region containing abdominal organs            |
| `Thoracic cavity`              | Contains the lungs and heart                  |
| `Gland structure`              | Includes thyroid, thymus, etc.                |
| `Pericardium`                  | Membrane around the heart                     |
| `Prosthetic breast implant`    | Annotation of prosthetic implant              |
| `Mediastinum`                  | Central part of thoracic cavity               |
| `Spinal cord`                  | Part of the central nervous system            |

## metadata

There are three files: `Metadata_Attributes.xlsx`, `train_metadata.csv`, and `validation_metadata.csv`

The file `Metadata_Attributes.xlsx` explains each field in the other two files.

| Attribute                           | Explanation                                     | Chinese Explanation                   |
| ----------------------------------- | ----------------------------------------------- | ------------------------------------- |
| `VolumeName`                        | The file name.                                  | CT volume image file name             |
| `Manufacturer`                      | Equipment manufacturer.                         | Manufacturer of imaging device        |
| `SeriesDescription`                 | Description of the series.                      | Description of image series           |
| `ManufacturerModelName`             | Model name of the equipment.                    | Device model                          |
| `PatientSex`                        | Sex of the patient.                             | Patient sex (M: male, F: female)      |
| `PatientAge`                        | Age of the patient (e.g., "045Y").              | Patient age in years                  |
| `ReconstructionDiameter`            | Diameter of reconstruction area (in mm).        | Size of reconstructed image area      |
| `DistanceSourceToDetector`          | Distance from X-ray source to detector (mm).    | Source-to-detector distance           |
| `DistanceSourceToPatient`           | Distance from X-ray source to patient (mm).     | Source-to-patient distance            |
| `GantryDetectorTilt`                | Tilt angle of the gantry/detector.              | Detector tilt angle                   |
| `SliceThickness`                    | Thickness of each CT slice (mm).                | Slice thickness                       |
| `SpacingBetweenSlices`              | Gap between slices (mm).                        | Inter-slice spacing                   |
| `KVP`                               | Kilovolt peak during scan.                      | Voltage during scan (kV)              |
| `TableHeight`                       | Height of the CT table (mm).                    | CT table height                       |
| `ConvolutionKernel`                 | Filter used for image reconstruction.           | Reconstruction convolution kernel     |
| `ImageType`                         | Image classification flags.                     | Image type (e.g., original, enhanced) |
| `PixelSpacing` / `XYSpacing`        | Pixel spacing in x and y (mm).                  | Pixel size in x and y directions      |
| `Rows` / `Columns`                  | Image dimensions (pixels).                      | Image width and height                |
| `RescaleIntercept` / `RescaleSlope` | Coefficients for converting pixel values to HU. | Linear transformation to HU values    |
| `NumberofSlices`                    | Total number of slices in the volume.           | Number of CT slices                   |
| `ZSpacing`                          | Spacing between slices in the z-axis (mm).      | Spacing in the longitudinal direction |
| `StudyDate`                         | Date of the CT scan (format: YYYYMMDD).         | Date of examination                   |

`train_metadata.csv` and `validation_metadata.csv` record metadata for each CT image in the training and validation sets. These help understand the acquisition conditions, spatial resolution, voxel density, etc.

## multi_abnormality_labels

Two files: `train_predicted_labels.csv` and `valid_predicted_labels.csv`

These two files contain **predicted labels of clinical abnormalities** for each CT image in the training and validation sets.

Each row represents the multi-label prediction for a single CT image (i.e., multiple abnormalities may exist):

| Field         | Meaning                                                      |
| ------------- | ------------------------------------------------------------ |
| `VolumeName`  | CT image file name (e.g., `train_1_a_1.nii.gz`)              |
| Other columns | Each column is a clinical abnormality. Values are `0` or `1`, meaning absent or present |

| Label                                | Description                                 |
| ------------------------------------ | ------------------------------------------- |
| `Medical material`                   | Presence of implants, stents, etc.          |
| `Arterial wall calcification`        | Sign of artery hardening                    |
| `Cardiomegaly`                       | Enlarged heart size                         |
| `Pericardial effusion`               | Fluid in the pericardial sac                |
| `Coronary artery wall calcification` | Related to coronary artery disease          |
| `Hiatal hernia`                      | Stomach structure moves into chest          |
| `Lymphadenopathy`                    | May indicate infection or tumor             |
| `Emphysema`                          | Common in COPD                              |
| `Atelectasis`                        | Partial lung collapse                       |
| `Lung nodule`                        | Small nodules, possibly benign or malignant |
| `Lung opacity`                       | May indicate lung disease                   |
| `Pulmonary fibrotic sequela`         | Fibrosis from chronic lung damage           |
| `Pleural effusion`                   | Fluid buildup in the pleural space          |
| `Mosaic attenuation pattern`         | Sign of small airway disease                |
| `Peribronchial thickening`           | Seen in bronchitis, asthma                  |
| `Consolidation`                      | Lung filled with fluid or inflammation      |
| `Bronchiectasis`                     | Permanent dilation of bronchi               |
| `Interlobular septal thickening`     | Related to edema, fibrosis, etc.            |

## radiology_text_reports

Two files: `train_reports.csv` and `validation_reports.csv`

These files contain **radiology reports** corresponding to each CT scan in the training and validation sets.

| Field Name               | Meaning                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `VolumeName`             | CT image file name (e.g., `train_1_a_1.nii.gz`)              |
| `ClinicalInformation_EN` | Clinical info (symptoms, past medical history, etc.)         |
| `Technique_EN`           | Imaging technique (scan type, contrast use, slice thickness, etc.) |
| `Findings_EN`            | Radiologist's observations (lungs, heart, mediastinum, etc.) |
| `Impressions_EN`         | Summary and diagnostic impression (include disease names or advice, etc.) |
