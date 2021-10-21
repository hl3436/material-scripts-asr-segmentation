LANG=$1

python /src/main.py \
	--test_path /work/bin/asr \
	--vocab_file /models/${LANG}_tokenizer.json \
	--model lstm --bidirectional --batch_first \
	--precision 16 --num_workers 0 \
	--load_from_checkpoint /models/${LANG}.ckpt --gpus 1
