
for i in 4 6 16;
do
    uv run scripts/preprocess_captions.py --seq_id $i --seconds_per_caption 3 --data_path /mnt/d/coda_data/ --out_path data/captions/$i/captions
done

