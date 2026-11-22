"""Live ASL letter recognition from the webcam (Phases 3–4, owner: Ayla).

Planned: capture frame -> common.crop_box -> common.preprocess_hog / preprocess_cnn
-> model -> show letter, confidence and ms/frame. Switch HOG+SVM / ResNet18 live.
"""
