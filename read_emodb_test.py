import numpy as np
import sys
import librosa
import python_speech_features as psf
import _pickle as cPickle
import os
eps = 1e-5
def read_test_emodb(img_rows=512, img_cols=128):
	rootdir = "D:\\Emodb berlin\\test\\"
	targetdir = "D:\\Emodb berlin\\"
	num = 196
	data = np.empty((num, img_cols*img_rows))
	test_label = np.empty(num, dtype=int)
	test_num=0
	em_test_label= np.empty(7, dtype=int)
	test_data = np.empty((num,300,40,3),dtype = np.float32)
	for filename in os.listdir(rootdir):
		name = "".join(filename)
		if(name.endswith('wav') == True):
			full_name = rootdir+name

			y, fs = librosa.load(full_name, sr=None)

			mel_spec = psf.logfbank(y,fs,nfilt = 40)
			delta1 = psf.delta(mel_spec, 2)
			delta2 = psf.delta(delta1, 2)

			mean1= np.mean(mel_spec, axis=0)
			std1=np.std(mel_spec, axis=0)
			mean2= np.mean(delta1, axis=0)
			std2=np.std(delta1, axis=0)
			mean3= np.mean(delta2, axis=0)
			std3=np.std(delta2, axis=0)

			time = mel_spec.shape[0] 

			if filename[5] == 'W':  # anger
				label = 1
			elif filename[5] == 'L': # boredom
				label = 2
			elif filename[5] == 'E':  # disgust
				label = 3
			elif filename[5] == 'A':  # anxiety/fear
				label = 4
			elif filename[5] == 'F':  # happiness
				label = 5
			elif filename[5] == 'T':  # sadness
				label = 6
			elif filename[5] == 'N':  # neutral
				label = 0

			if(time <= 300):
				part = mel_spec
				delta11 = delta1
				delta21 = delta2
				part = np.pad(part,((0,300 - time),(0,0)),'constant',constant_values = 0)
				delta11 = np.pad(delta11,((0,300 - time),(0,0)),'constant',constant_values = 0)
				delta21 = np.pad(delta21,((0,300 - time),(0,0)),'constant',constant_values = 0)
				test_data[test_num,:,:,0] = (part -mean1)/(std1+eps)
				test_data[test_num,:,:,1] = (delta11 - mean2)/(std2+eps)
				test_data[test_num,:,:,2] = (delta21 - mean3)/(std3+eps)
				test_label[test_num]=label
				em_test_label[label]= em_test_label[label]+1
				test_num = test_num + 1
			else:
				for i in range(2):
					if(i == 0):
						begin = 0
						end = begin + 300
					else:
						begin = time - 300
						end = time
						part = mel_spec[begin:end,:]
						delta11 = delta1[begin:end,:]
						delta21 = delta2[begin:end,:]
						test_data[test_num,:,:,0] = (part -mean1)/(std1+eps)
						test_data[test_num,:,:,1] = (delta11 - mean2)/(std2+eps)
						test_data[test_num,:,:,2] = (delta21 - mean3)/(std3+eps)
						test_label[test_num]=label
						em_test_label[label]= em_test_label[label]+1
						test_num = test_num + 1

	f = open('./emodb_test.pkl', 'wb')
	cPickle.dump((test_data,test_label,em_test_label), f)
	f.close()
if __name__== "__main__":
    read_test_emodb()
    f = open('C:/Users/cpank/Desktop/emodb_test.pkl', 'rb')
    test_data,test_label,em_test_label = cPickle.load(f)
    print(test_data.shape)