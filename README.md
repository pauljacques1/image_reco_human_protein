# image_reco_human_protein
Initial data can be found here: https://www.kaggle.com/c/human-protein-atlas-image-classification/data

merge_testset.py is used to merge the files together in one RGB image (can be used to merge them into RGBY).
train_with_resnet.py is the script to train a binary classifier for each class (28 in total), using the RESNET50 architecture. The script uses a keras data generator.
prediction_script.py  predicts on the test set, also uses a data generator.
