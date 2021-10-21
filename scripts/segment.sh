IN_DIR=/input
OUT_DIR=/output
LANG=$1

mkdir -p $OUT_DIR

case $LANG in
    lt | bg | fa)
        echo "Audio Segmentation for $LANG"
            cp -r $IN_DIR/* $OUT_DIR

            mkdir -p /work /work/raw /work/bin /work/split
        
            # transform input
            echo "Transforming to input"
            python /scripts/transform_input.py $IN_DIR
            # tokenize
            python /scripts/preprocessing.py /work/raw/raw.txt /work/raw/tok.txt $LANG
            # combine lines
            python /scripts/combine_lines.py /work/raw

            # bpe
            python /scripts/apply_bpe.py /work/raw $LANG

            # to datasets
            python /scripts/to_dataset.py /work/raw /work

            # to test copus
            python /scripts/binarize.py /work asr /models/${LANG}_tokenizer.json

            # segment
            bash /scripts/eval.sh $LANG

            # to files and combine
            python /scripts/to_asr_files.py /work/split $LANG
            python /scripts/combine_asr.py /work/split $IN_DIR $OUT_DIR

            chmod -R 777 $OUT_DIR
        ;;

    sw | so | ps | tl )
        echo "Language $LANG currently not supported. Copying the files over..."
            cp -r $IN_DIR/* $OUT_DIR
        ;;

    *)
        echo "Language $LANG unknown"
        ;;
esac
