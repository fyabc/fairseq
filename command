# from Myle Ott
python train.py data-bin/iwslt14.tokenized.de-en --max-update 50000 --arch transformer_iwslt_de_en --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --lr 0.0005 --min-lr '1e-09' --clip-norm 0.0 --dropout 0.3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 --seed 1 --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-layers 6 --decoder-embed-dim 512 --decoder-ffn-embed-dim 1024 --decoder-layers 6 --no-progress-bar --log-interval 100

# from Michael Auli
python train.py data-bin/iwslt14.tokenized.de-en --source-lang de --target-lang en \
  --save-dir checkpoints/auli \
  --max-update 50000 --arch transformer_iwslt_de_en \
  --encoder-embed-dim 512 --encoder-ffn-embed-dim 1024 --encoder-attention-heads 4 --encoder-layers 6 \
  --decoder-embed-dim 512 --decoder-ffn-embed-dim 1024 --decoder-attention-heads 4 --decoder-layers 6 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.0005 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0 --dropout 0.3 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4000 --no-progress-bar --log-interval 100 --seed 1

# test
python generate.py data-bin/iwslt14.tokenized.de-en \
  --task diverse_translation --path $CKPT/checkpoint_best.pt \
  --batch-size 128 --beam 5 --remove-bpe | tee $CKPT/gen.out


# latent variable
python train.py data-bin/iwslt14.tokenized.de-en --save-dir checkpoints/iwslt.latent2 \
  --criterion latent_var --latent-category 2 --latent-token\
  --max-update 50000 --arch transformer_iwslt_de_en \
  --optimizer adam --adam-betas '(0.9, 0.98)' --lr 0.0005 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --clip-norm 0 --dropout 0.3 --label-smoothing 0.0 \
  --max-tokens 4000 --no-progress-bar --log-interval 100 --seed 1


# wmt14 en-de
python train.py data-bin/wmt14_en_de_joined_dict \
  --task diverse_translation --criterion latent_var --latent-category 2\
  --max-epoch 50 --arch transformer_vaswani_wmt_en_de_big \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler fixed --lr 0.0001 \
  --clip-norm 0.0 --dropout 0.3 --weight-decay 0.0 --label-smoothing 0.0 \
  --max-tokens 4000 --no-progress-bar --log-interval 10 --seed 2


# wmt17 zh-en
python train.py data-bin/wmt17_zh_en_small --save-dir checkpoints/wmt17_zh_en_small \
 -a transformer_iwslt_de_en --max-tokens 4000 --max-update 20000 \
 --optimizer adam --adam-betas '(0.9, 0.98)' \
 --lr-scheduler inverse_sqrt --lr 0.0005 --min-lr '1e-09' --warmup-init-lr '1e-07' --warmup-updates 4000 \
 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
 --clip-norm 0 --dropout 0.3 --attention-dropout 0.3 --weight-decay 0.0001 \
 --no-progress-bar --log-interval 100 --seed 1

python train.py data-bin/wmt17_zh_en_full --save-dir checkpoints/wmt17_zh_en_full \
-a transformer_vaswani_wmt_en_de_big --clip-norm 0.0 --lr 0.0005 \
--label-smoothing 0.1 --attention-dropout 0.2 \
--dropout 0.2 --max-tokens 3584 --no-progress-bar --log-interval 100 \
--weight-decay 0.0 --criterion label_smoothed_cross_entropy --fp16 \
--max-update 40000 --seed 3 \
--optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 --warmup-updates 4000 --min-lr 1e-09
