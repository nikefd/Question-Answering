# Match-LSTM and Answer Pointer (Wang and Jiang, ICLR 2016)
This repo attempts to reproduce the match-lstm and answer pointer experiments from the 2016 paper on the same. A lot of the preprocessing boiler code is taken from Stanford CS224D. 

The meat of the code is in qa_model.py. I had to modify tensorflow's original attention mechanism implementation for the code to be correct. run train.py to train the model and qa_answer.py to generate answers given a set of paragraphs. Contact me at shikhar.murty@gmail.com for more info.

This code also serves as an example code showing how tensorflow's attention mechanism can be wired together. As of August 13th, 2017, such an example was not available anywhere.


# Requirements
* A computer running macOS or Linux
* Tensorflow 1.2
* For training new models, you'll also need a NVIDIA GPU
* Python version 2.7

# Quick Start

### Data acquisition and preparation

```
python /preprocessing/squad_preprocess.py
python /preprocessing/dwr.py
python qa_data.py #make sure you have squad and dwr under the download folder
```

### Training
```
python train.py
```

### Generation
```
python qa_answer.py input_file output_file
```

### Evaluating
```
python evaluate-v1.1.py groud_truth_file prediction_file
```
