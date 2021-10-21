RAW_FILE=$1
OUT_FILE=$2
l=$3

SCRIPTS_PATH=/scripts/mosesdecoder/scripts
TOKENIZER=$SCRIPTS_PATH/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS_PATH/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS_PATH/tokenizer/remove-non-printing-char.perl
LOWERCASE=$SCRIPTS_PATH/tokenizer/lowercase.perl

cat $RAW_FILE | \
    perl $LOWERCASE | \
    perl $NORM_PUNC $l | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l $l > $OUT_FILE
