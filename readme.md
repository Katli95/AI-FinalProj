# AI-Final Project Spring 2019
The model comes preloaded with weights trained on eight images:
- 022dbb19-2f9c-4fea-bfd1-292260878db0.jpg
- 0300e3d2-b074-4b2e-8a88-360548b60259.jpg
- 04aaf572-3fbc-4d62-b770-8250bf0f8256.jpg
- 06f363fb-b6da-446c-9016-7543265aa794.jpg
- 07328a88-4581-4f94-9fe5-652b4ae513ac.jpg
- 078391d8-d6d5-4cc2-b0d3-44df191ee5cd.jpg
- 08292e21-53fe-4bd3-8f94-3610b9d271ee.jpg

## How to run 
```python
#Test model
test.testImg("./data/img/022dbb19-2f9c-4fea-bfd1-292260878db0.jpg")
# The network will output the resulting image to /data/output/022dbb19-2f9c-4fea-bfd1-292260878db0_detected.jpg

#Train model
from yolo import *
test = YOLO()
test.train(2, 1, 20, 10**-4, 0) # params are: 
#  (
#  number of times to run training data in an epoch, 
#  number of times to run validation data in an epoch,
#  learning rate,
#  Debug (boolean)
#  )

```
