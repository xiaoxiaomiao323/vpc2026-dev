bash create_mls.sh > mls_out.txt 
python  build_cn_data.py > cn_out.txt 
python build_ja_data.py > jav_parallel100.txt
python build_ja_data.py --subset nonpara30 > jav_nonpara30.txt
s