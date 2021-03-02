from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd
import numpy as np


def home(request):
	return render(request, "home.html")

def result(request):
	clf = joblib.load('glass.sav')
	# lis  = pd.DataFrame([request.GET['Ri'],request.GET['Na'],request.GET['Mg'],request.GET['Al'],request.GET['Si'],request.GET['K'], request.GET['Ca'],request.GET['Ba'],request.GET['Fa']])
	lis = []
	 
	lis.append(request.GET['Ri'])
	lis.append(request.GET['Na'])
	lis.append(request.GET['Mg'])
	lis.append(request.GET['Al'])
	lis.append(request.GET['Si'])
	lis.append(request.GET['K'])
	lis.append(request.GET['Ca'])
	lis.append(request.GET['Ba'])
	lis.append(request.GET['Fa'])
	# lis = np.array(lis)
	# lis = lis.reshape(-1,1)
	pred_class = clf.predict([lis])

	return render (request, "result.html",{'pred_class': pred_class})