This project involves the detection, deskewing, and pixelation of faces and QR/barcodes in images of personal cards. The process is performed using models for card detection, face detection, and QR/barcode detection. The project also includes OCR functionality to identify and parse personal information from the cards and mask it to ensure privacy.
### Personal Cards Datasets

For the personal cards, we didn't rely on a single dataset. Instead, we combined multiple datasets, including publicly available datasets and our own custom-generated dataset. The links for the public datasets we used are:

- [Data Source 1](https://universe.roboflow.com/weedigital-nu572/cc-card/browse?queryText=&pageSize=200&startingIndex=0&browseQuery=true)
- [Data Source 2](https://universe.roboflow.com/bioecosys/card-6i8ys/dataset/1/images?split=train)
- [Data Source 3](https://universe.roboflow.com/documentos-31jir/cardid-j0tid)
- [Custom Created dataset (Not sharing due to privacy reasons)]()

### QR-Barcode Dataset

We used the following public dataset for QR and barcode detection [Data Source](https://universe.roboflow.com/labeler-projects/barcodes-zmxjq/dataset/5)

### Face Detection Dataset

For face detection in the cards, we utilized the following dataset [Data Source](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/24)