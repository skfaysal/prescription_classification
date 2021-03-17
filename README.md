# prescription_classification

1.Create a folder *"TrainData"* inside *"train and test"* folder\
2.Create 3 folder named "test" ,"train","validation" inside "TrainData"\
3.Upload dataset as "No Pres" and "Pres" folder name in 3 of the folder,Here validation is potional\
4.Create a folder named "misclass_image" inside "train nas test" folder.All the missclassified images after running test.py will be stored here\
5.For training : python3 train_resnet50.py --img_row 224 --img_col 224 --epochs 25 --batch_size 32 --val_split 0.25\
6.For testing: python3 test.py
