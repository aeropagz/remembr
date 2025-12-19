
for i in 3 4 6 16;
do
    python scripts/preprocess_captions.py --seq_id $i --seconds_per_caption 3 --data_dir /mnt/d/coda-data --out_path data/captions/$i/captions
done

