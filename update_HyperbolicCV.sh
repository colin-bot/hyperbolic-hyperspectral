# Script used to update the HyperbolicCV library to be compatible with our hyperspectral data

rm ../HyperbolicCV/code/classification/models/classifier.py
cp update_HyperbolicCV/classifier.py ../HyperbolicCV/code/classification/models/classifier.py

rm ../HyperbolicCV/code/lib/models/resnet.py
cp update_HyperbolicCV/resnet.py ../HyperbolicCV/code/lib/models/resnet.py

echo 'Updated HyperbolicCV to work with hyperspectral images.'
