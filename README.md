# EMG_ASR
## File Description
configs: parameter configuration file
src: model files, data processing files
scripts: bulid dictionary


## Run

Train the ctc model
```
python train_emg_ctc.py --config_path configs/un-aguement/transformer_ctc_silent.yaml --model_name TransformerCTC --gpu 0
```
