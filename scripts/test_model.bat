@rem Checkpoints: 43, 58, 73, 85, 88
@set REMOTE_DATA_PATH="F:\\Users\v-yaf\DataTransfer\fairseq\wmt14_en_de_joined_dict"
@set REMOTE_MODEL_PATH="\\GCRAZGDW140\ckpts\WMT14_EN_DE_CYCLE_4.5k"
python generate.py %REMOTE_DATA_PATH%\wmt14_en_de_joined_dict --path %REMOTE_MODEL_PATH%\checkpoint%1.pt --batch-size 128 ^
    --beam 5 --lenpen 0.6 --remove-bpe ^
    --quiet --no-progress-bar
