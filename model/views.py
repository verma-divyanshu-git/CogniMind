from django.shortcuts import render
from django.http import HttpResponseRedirect
from .forms import YourFileUploadForm
import os
from django.conf import settings
from django.contrib import messages
import joblib
from glob import glob
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
from keras.models import load_model
import pickle
from scipy import signal
from scipy.fftpack import fft, ifft
import statsmodels.api as sm
import antropy as ant
from scipy.stats import kurtosis
from scipy.stats import skew
import tensorflow as tf
import pywt

modelEO1 = joblib.load(r"./models/knn_EO.pkl")
modelEO2 = joblib.load(r"./models/xgb_EO.pkl")
modelEO3 = joblib.load(r"./models/svm_EO.pkl")
modelEO4 = joblib.load(r"./models/rf_EO.pkl")
modelEO5 = joblib.load(r"./models/nb_EO.pkl")
modelEO6 = joblib.load(r"./models/lr_EO.pkl")
modelEO7 = joblib.load(r"./models/dt_EO.pkl")

modelEC1 = joblib.load(r"./models/knn_EC.pkl")
modelEC2 = joblib.load(r"./models/xgb_EC.pkl")
modelEC3 = joblib.load(r"./models/svm_EC.pkl")
modelEC4 = joblib.load(r"./models/rf_EC.pkl")
modelEC5 = joblib.load(r"./models/nb_EC.pkl")
modelEC6 = joblib.load(r"./models/lr_EC.pkl")
modelEC7 = joblib.load(r"./models/dt_EC.pkl")

modelT1 = joblib.load(r"./models/knn_T.pkl")
modelT2 = joblib.load(r"./models/xgb_T.pkl")
modelT3 = joblib.load(r"./models/svm_T.pkl")
modelT4 = joblib.load(r"./models/rf_T.pkl")
modelT5 = joblib.load(r"./models/nb_T.pkl")
modelT6 = joblib.load(r"./models/lr_T.pkl")
modelT7 = joblib.load(r"./models/dt_T.pkl")


def index(request):
    if request.method == "POST":
        form = YourFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            eeg_files_path = os.path.join(settings.MEDIA_ROOT, "eeg_files")
            existing_files = glob(os.path.join(eeg_files_path, "*"))
            for file in existing_files:
                os.remove(file)

            fdt_file = form.cleaned_data["csvFile"]

            with open(
                os.path.join(settings.MEDIA_ROOT, "eeg_files", fdt_file.name), "wb+"
            ) as destination:
                for chunk in fdt_file.chunks():
                    destination.write(chunk)
            file_path = glob(r"pictures\eeg_files\*.csv")
            for file in file_path:
                df = pd.read_csv(file)

            eeg_data = df.iloc[:, 1:].values

            feature_vectors = []

            for channel in range(eeg_data.shape[1]):
                channel_data = eeg_data[:, channel]

                perm_entropy = ant.perm_entropy(channel_data, normalize=True)
                spectral_entropy = ant.spectral_entropy(
                    channel_data, sf=100, method="welch", normalize=True
                )
                svd_entropy = ant.svd_entropy(channel_data, normalize=True)
                hjorth_params = ant.hjorth_params(channel_data)
                petrosian_fd = ant.petrosian_fd(channel_data)
                katz_fd = ant.katz_fd(channel_data)
                higuchi_fd = ant.higuchi_fd(channel_data)
                dfa = ant.detrended_fluctuation(channel_data)
                channel_skewness = skew(channel_data)
                channel_kurtosis = kurtosis(channel_data)

                feature_vector = [
                    perm_entropy,
                    spectral_entropy,
                    svd_entropy,
                    hjorth_params[0],
                    hjorth_params[1],
                    petrosian_fd,
                    katz_fd,
                    higuchi_fd,
                    dfa,
                    channel_skewness,
                    channel_kurtosis,
                ]

                feature_vectors.append(feature_vector)

            x_test = np.array(feature_vectors, dtype=object)

            category = form.cleaned_data["category"]

            models = []

            if category == "EO":
                models = [
                    modelEO1,
                    modelEO2,
                    modelEO3,
                    modelEO4,
                    modelEO5,
                    modelEO6,
                    modelEO7,
                ]
            elif category == "EC":
                models = [
                    modelEC1,
                    modelEC2,
                    modelEC3,
                    modelEC4,
                    modelEC5,
                    modelEC6,
                    modelEC7,
                ]
            elif category == "T":
                models = [modelT1, modelT2, modelT3, modelT4, modelT5, modelT6, modelT7]

            results = []

            for idx, model in enumerate(models, start=1):
                predictions = model.predict(x_test)
                percentage_healthy = np.mean(predictions == 0) * 100
                percentage_mdd = np.mean(predictions == 1) * 100

                result = {
                    "model_name": f"Model {idx}",
                    "diagnosis": (
                        f"Prediction: Healthy with {percentage_healthy:.2f}% confidence"
                        if percentage_healthy > percentage_mdd
                        else f"Prediction: MDD with {percentage_mdd:.2f}% confidence"
                    ),
                }

                results.append(result)

            return render(request, "index.html", {"form": form, "results": results})
        else:
            # If form is invalid, add a message and redirect to the same page
            messages.error(
                request, "Invalid form submission. Please check your inputs."
            )
            return HttpResponseRedirect(request.path_info)

    else:
        form = YourFileUploadForm()

    return render(request, "index.html", {"form": form, "results": None})


def success(request):
    return render(request, "success.html")
