wiki_data=data/wiki_data
# wiki
python wiki.py $wiki_data data/wiki
python t2s.py data/wiki data/wiki.s
python cut.py wiki.s wiki.cut
# resume
python libs/pre_train.py -x resumes/ -o resumes.txt
python cut.py resumes.txt resume.cut
echo resume.cut >> wiki.cut
mv wiki.cut data/summary
python train.py data/summary
# clean
rm -rf data/wiki.s
rm -rf data/resumes.txt
rm -rf data/resumes.cut
