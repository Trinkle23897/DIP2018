#!/usr/bin/env python3

import cv2,argparse
import numpy as np

def brightness(img,c):
	return np.clip(img*1.+c,0,255)

def gamma(img,r):
	return np.power(img/255,1/r)*255

def contrast(img,k):
	return np.clip((img-127.)*k+127.,0,255)

def getLUT(img):
	_,LUT=np.unique(img,return_counts=True)
	LUT=np.append(LUT,np.zeros(256-LUT.shape[0]))
	return np.cumsum(LUT/LUT.sum())*255

def meta_hist_equ(img):
	return getLUT(img)[img]

def hist_equ(img):
	# return meta_hist_equ(img)
	if len(img.shape)!=3:
		return meta_hist_equ(img)
	else:
		return np.dstack((meta_hist_equ(img[:,:,0]),
						  meta_hist_equ(img[:,:,1]),
						  meta_hist_equ(img[:,:,2])))

def meta_hist_match(img,style):
	# output=(LUT_style)^(-1)[LUT_img[img]]
	LUT_img=np.round(getLUT(img)).astype(np.uint8)
	LUT_style=getLUT(style)
	LUT_inv=np.zeros(256)
	for i in range(255):
		m0=LUT_style[i]
		m1=LUT_style[i+1]
		for j in range(int(m0+1),int(m1+1)):
			LUT_inv[j]=m0 if m0==m1 else (j-m0)/(m1-m0)+i
	LUT_inv[255]=255
	return np.round(LUT_inv[LUT_img[img]])

def hist_match(img,style):
	if len(img.shape)!=3 or len(style.shape)!=3:
		return meta_hist_match(img,style)
	else:
		return np.dstack((meta_hist_match(img[:,:,0],style[:,:,0]),
						  meta_hist_match(img[:,:,1],style[:,:,1]),
						  meta_hist_match(img[:,:,2],style[:,:,2])))

def write_img(img,name):
	print('Output image into file %s'%name)
	cv2.imwrite(name,img)

def main():
	parser = argparse.ArgumentParser(description='Simple point processing demo')
	parser.add_argument('-i', dest='input', type=str, help='Input image filename')
	parser.add_argument('-o', dest='output', default='try.png', type=str, help='Output image filename')
	parser.add_argument('-b', dest='gain', type=float, help='Brightness parameter')
	parser.add_argument('-c', dest='contrast', type=float, help='Contrast parameter')
	parser.add_argument('-g', dest='gamma', type=float, help='Gamma parameter')
	parser.add_argument('-e', dest='equal', action='store_true', default=False, help='Histogram equalization')
	parser.add_argument('-m', dest='style', type=str, help='Histogram matching, input style image\'s filename')
	args = parser.parse_args()

	if args.input:
		img=cv2.imread(args.input)
	else:
		print('Please specify one input image, use "-i" option or "--help".')
		return
	if args.gain:
		print('Performing brightness transformation ...')
		write_img(brightness(img,args.gain),args.output)
		return
	if args.gamma:
		print('Performing gamma transformation ...')
		write_img(gamma(img,args.gamma),args.output)
		return
	if args.contrast:
		print('Performing contrast transformation ...')
		write_img(contrast(img,args.contrast),args.output)
		return
	if args.equal:
		print('Performing histogram equalization ...')
		write_img(hist_equ(img),args.output)
		return
	if args.style:
		print('Performing histogram matching ...')
		write_img(hist_match(img,cv2.imread(args.style)),args.output)
		return

if __name__=='__main__':
	main()
